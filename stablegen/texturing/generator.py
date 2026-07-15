"""ComfyUI texture generation operator (core pipeline)."""

import os
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import numpy as np
import cv2

import uuid
import json
import urllib.request
import urllib.parse
import socket
import threading
import traceback
import io
from datetime import datetime
import math
import colorsys
from PIL import Image, ImageEnhance

import gpu  # pylint: disable=import-error
import blf  # pylint: disable=import-error
from gpu_extras.batch import batch_for_shader  # pylint: disable=import-error

from ..util.workflow_templates import prompt_text, prompt_text_img2img, prompt_text_qwen_image_edit
from .rendering import export_emit_image, export_visibility, export_canny, bake_texture, prepare_baking, unwrap, export_render, export_viewport, render_edge_feather_mask, apply_uv_inpaint_texture
from ..cameras.geometry import _SGCameraResolution, _get_camera_resolution
from ..cameras.overlays import _sg_restore_square_display, _sg_remove_crop_overlay, _sg_ensure_crop_overlay, _sg_hide_label_overlay, _sg_restore_label_overlay
from .projection import project_image, reinstate_compare_nodes, _bake_visibility_weights
from ..utils import get_last_material_index, get_generation_dirs, get_file_path, get_dir_path, remove_empty_dirs, get_compositor_node_tree, configure_output_node_paths, get_eevee_engine_id, sg_modal_active
from ..util.mirror_color import MirrorReproject, _get_viewport_ref_np, _apply_color_match_to_file
from ..timeout_config import get_timeout
from .._generator_utils import redraw_ui, setup_studio_lighting, _pbr_setup_studio_lights, upload_image_to_comfyui
from .pbr import _PBRMixin
from .gallery import _PreviewGalleryOverlay
from .view_plan import ViewRecord, choose_aligned_resolution, order_cameras
from .quality_masks import (
    apply_binary_context_mask,
    regularize_repair_mask,
    reject_fallback_from_write_mask,
)

import websocket

_ADDON_PKG = __package__.rsplit('.', 1)[0]

class Regenerate(bpy.types.Operator):
    """Regenerate textures for selected cameras / viewpoints
    - Works for sequential and separate generation modes
    - Generates new images for the selected cameras only, keeping existing images for unselected cameras
    - This can be used with different prompts or settings to refine specific viewpoints without affecting others"""
    bl_idname = "object.stablegen_regenerate"
    bl_label = "Regenerate Selected Viewpoints"
    bl_options = {'REGISTER', 'UNDO'}

    _original_method = None
    _original_overwrite_material = None
    _timer = None
    _to_texture = None
    @classmethod
    def poll(cls, context):
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        if not os.path.exists(addon_prefs.output_dir):
            cls.poll_message_set("Output directory not set or does not exist")
            return False
        if not addon_prefs.server_address or not addon_prefs.server_online:
            cls.poll_message_set("ComfyUI server is not connected")
            return False
        if not (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate'):
            cls.poll_message_set("Regenerate is only available in Sequential or Separate mode")
            return False
        if context.scene.output_timestamp == "":
            cls.poll_message_set("No previous generation to regenerate from")
            return False
        if context.scene.generation_status == 'waiting' or sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def execute(self, context):
        """     
        Executes the operator.         
        :param context: Blender context.         
        :return: {'FINISHED'}     
        """
        
        self._original_overwrite_material = context.scene.overwrite_material
        # Set the flag to reproject
        context.scene.generation_mode = 'regenerate_selected'
        # Set the generation method to 'separate' to avoid generating new images
        context.scene.overwrite_material = True
        # Set timer to 1 seconds to give some time for the generate to start
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        # Revert to original discard angle in material nodes in case it was reset after generation
        if context.scene.texture_objects == 'selected':
            self._to_texture = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            # If empty, cancel the operation
            if not self._to_texture:
                self.report({'ERROR'}, "No mesh objects selected. Select mesh objects in the viewport before regenerating.")
                context.scene.generation_status = 'idle'
                ComfyUIGenerate._is_running = False
                return {'CANCELLED'}
        else: # all
            self._to_texture = [obj for obj in bpy.context.view_layer.objects if obj.type == 'MESH' and not obj.hide_get()]
        # Revert discard angle
        new_discard_angle = context.scene.discard_factor
        for obj in self._to_texture:
            if not obj.active_material or not obj.active_material.use_nodes:
                continue
            
            nodes = obj.active_material.node_tree.nodes
            for node in nodes:
                # OSL script nodes (internal or external)
                if node.type == 'SCRIPT':
                    if 'AngleThreshold' in node.inputs:
                        node.inputs['AngleThreshold'].default_value = new_discard_angle
                # Native MATH LESS_THAN nodes (Blender 5.1+ native raycast path)
                elif node.type == 'MATH' and node.operation == 'LESS_THAN' and node.label.startswith('AngleThreshold-'):
                    node.inputs[1].default_value = new_discard_angle
        # Run the generation operator (bypass modal poll guard since we are
        # already registered as a modal operator ourselves)
        from .. import utils as _sg_utils
        _sg_utils._sg_bypass_modal_check = True
        try:
            bpy.ops.object.test_stable('INVOKE_DEFAULT')
        finally:
            _sg_utils._sg_bypass_modal_check = False

        # Switch to modal and wait for completion
        print("[StableGen] Going modal")
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        """     
        Handles modal events.         
        :param context: Blender context.         
        :param event: Blender event.         
        :return: {'PASS_THROUGH'}     
        """
        if event.type == 'TIMER':
            running = False
            if ComfyUIGenerate._is_running:
                running = True
            if not running:
                # Reset the generation method and overwrite material flag
                context.scene.overwrite_material = self._original_overwrite_material
                # Reset the project only flag
                context.scene.generation_mode = 'standard'
                # Remove the modal handler
                context.window_manager.event_timer_remove(self._timer)
                # Report completion
                self.report({'INFO'}, "Regeneration complete.")
                return {'FINISHED'}
        return {'PASS_THROUGH'}

class Reproject(bpy.types.Operator):
    """Rerun projection of existing images
    - Uses the Generate operator to reproject images, new textures will respect new Viewpoint Blending Settings
    - Will not work with textures which used refine mode with the preserve parameter enabled"""
    bl_idname = "object.stablegen_reproject"
    bl_label = "Reproject Images"
    bl_options = {'REGISTER', 'UNDO'}

    _original_method = None
    _original_overwrite_material = None
    _timer = None
    @classmethod
    def poll(cls, context):
        if context.scene.output_timestamp == "":
            cls.poll_message_set("No previous generation to reproject from")
            return False
        if context.scene.generation_status == 'waiting' or sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def execute(self, context):
        """     
        Executes the operator.         
        :param context: Blender context.         
        :return: {'FINISHED'}     
        """
        if context.scene.texture_objects == 'all':
            to_texture = [obj for obj in bpy.context.view_layer.objects if obj.type == 'MESH' and not obj.hide_get()]
        else: # selected
            to_texture = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

        # Search for largest material id
        max_id = -1
        for obj in to_texture:
            mat_id = get_last_material_index(obj)
            if mat_id > max_id:
                max_id = mat_id

        cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
        for i, _ in enumerate(cameras):
            # Check if the camera has a corresponding generated image
            image_path = get_file_path(context, "generated", camera_id=i, material_id=max_id)
            if not os.path.exists(image_path):
                # Try to recover from a packed/embedded Blender image
                # (e.g. after saving and reopening the .blend file when the
                # original output directory no longer exists).
                recovered = self._recover_image_from_blend(
                    image_path, to_texture, max_id)
                if not recovered:
                    self.report(
                        {'ERROR'},
                        f"Camera {i} does not have a corresponding "
                        f"generated image.")
                    print(f"[StableGen] {image_path} does not exist")
                    return {'CANCELLED'}
        
        self._original_method = context.scene.generation_method
        self._original_overwrite_material = context.scene.overwrite_material
        # Set the flag to reproject
        context.scene.generation_mode = 'project_only'
        # Set the generation method to 'separate' to avoid generating new images
        context.scene.generation_method = 'separate'
        context.scene.overwrite_material = True
        # Set timer to 1 seconds to give some time for the generate to start
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        # Run the generation operator (bypass modal poll guard since we are
        # already registered as a modal operator ourselves)
        from .. import utils as _sg_utils
        _sg_utils._sg_bypass_modal_check = True
        try:
            bpy.ops.object.test_stable('INVOKE_DEFAULT')
        finally:
            _sg_utils._sg_bypass_modal_check = False

        # Switch to modal and wait for completion
        print("[StableGen] Going modal")
        return {'RUNNING_MODAL'}

    @staticmethod
    def _recover_image_from_blend(image_path, to_texture, material_id):
        """Try to recover a generated image from bpy.data.images.

        When a .blend file is saved after generation, the generated
        textures are packed/embedded as Blender image data-blocks.  If
        the original output directory no longer exists (e.g. moved PC,
        temp folder cleared), we can find the image in the material's
        node tree and re-save it to the expected path.

        Returns True if the file was successfully recovered.
        """
        target_name = os.path.basename(image_path)

        def _try_save_image(img):
            """Attempt to write an image data-block to *image_path*."""
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            # Case 1: packed file — write raw bytes
            if img.packed_file:
                try:
                    with open(image_path, 'wb') as f:
                        f.write(img.packed_file.data)
                    print(f"[StableGen] Recovered packed image: "
                          f"{target_name}")
                    return True
                except Exception as err:
                    print(f"[StableGen] Packed write failed: {err}")

            # Case 2: image has a valid filepath elsewhere — copy it
            existing_path = bpy.path.abspath(img.filepath_raw)
            if existing_path and os.path.isfile(existing_path):
                try:
                    import shutil
                    shutil.copy2(existing_path, image_path)
                    print(f"[StableGen] Copied image from "
                          f"{existing_path} → {image_path}")
                    return True
                except Exception as err:
                    print(f"[StableGen] File copy failed: {err}")

            # Case 3: pixel data in memory — save via render
            try:
                if img.has_data or len(img.pixels) > 0:
                    img.save_render(image_path)
                    print(f"[StableGen] Recovered image via save_render: "
                          f"{target_name}")
                    return True
            except Exception:
                pass

            return False

        # Strategy 1: scan Image Texture nodes in the target materials
        # (most reliable — images are directly referenced by the object
        #  being reprojected, so we won't accidentally pick up images
        #  from a different model/object that share the same filename).
        for obj in to_texture:
            if not hasattr(obj, 'data') or not obj.data.materials:
                continue
            for mat in obj.data.materials:
                if not mat or not mat.use_nodes:
                    continue
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        node_filename = os.path.basename(
                            node.image.filepath_raw)
                        if node_filename == target_name:
                            if _try_save_image(node.image):
                                return True

        # Strategy 2: direct name lookup in bpy.data.images (fallback)
        for img in bpy.data.images:
            if img.name == target_name or os.path.basename(
                    img.filepath_raw) == target_name:
                if _try_save_image(img):
                    return True

        return False

    def modal(self, context, event):
        if event.type == 'TIMER':
            running = False
            if ComfyUIGenerate._is_running:
                running = True
            if not running:
                # Reset the generation method and overwrite material flag
                context.scene.generation_method = self._original_method
                context.scene.overwrite_material = self._original_overwrite_material
                # Reset the project only flag
                context.scene.generation_mode = 'standard'
                # Remove the modal handler
                context.window_manager.event_timer_remove(self._timer)
                # Report completion
                self.report({'INFO'}, "Reprojection complete.")
                return {'FINISHED'}
        return {'PASS_THROUGH'}
    


class ComfyUIGenerate(_PBRMixin, bpy.types.Operator):
    """Generate textures using ComfyUI (to all mesh objects using all cameras in the scene)
    
    - Multiple modes are available. Choose by setting Generation Mode in the UI.
    - This includes texture generation and projection to the mesh objects.
    - By default, the generated textures will only be visible in the Rendered viewport shading mode."""
    bl_idname = "object.test_stable"
    bl_label = "Generate using ComfyUI"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _progress = 0
    _error = None
    _is_running = False
    _active_ws = None  # WebSocket reference for cancel-time close
    _threads_left = 0
    _cameras = None
    _view_plan = None
    _selected_camera_ids = None
    _grid_width = 0
    _grid_height = 0
    _material_id = -1
    _to_texture = None
    _original_visibility = None
    _generation_method_on_start = None
    _uploaded_images_cache: dict = {}
    workflow_manager: object = None

    # Add properties to track progress
    _progress = 0.0
    _stage =  ""
    _current_image = 0
    _total_images = 0
    _wait_event = None
    # PBR progress (model-level steps across all cameras)
    _pbr_active = False
    _pbr_step = 0
    _pbr_total_steps = 0
    _pbr_cam = 0           # current camera index within current step
    _pbr_cam_total = 1     # total cameras in current step

    # Add new properties at the top of the class
    _object_prompts: dict = {}
    show_prompt_dialog: bpy.props.BoolProperty(default=True)
    current_object_name: bpy.props.StringProperty()
    current_object_prompt: bpy.props.StringProperty(
        name="Object Prompt",
        description="Enter a specific prompt for this object",
        default=""
    ) # type: ignore
    # New properties for prompt collection
    _mesh_objects: list = []
    mesh_index: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_images = 0
        self._current_image = 0
        self._stage = ""
        self._progress = 0
        self._pbr_active = False
        self._pbr_step = 0
        self._pbr_total_steps = 0
        self._pbr_cam = 0
        self._pbr_cam_total = 1
        self._wait_event = threading.Event()
        from ..workflows import WorkflowManager
        self.workflow_manager = WorkflowManager(self)

    def _run_on_main_thread(self, func):
        """Execute *func* on Blender's main thread via a timer callback and
        block until it completes.  Sets ``self._error`` on failure."""
        def _callback():
            try:
                func()
            except Exception as exc:
                self._error = str(exc)
                traceback.print_exc()
            self._wait_event.set()
            return None          # one-shot timer
        bpy.app.timers.register(_callback)
        self._wait_event.wait()
        self._wait_event.clear()
                
    def _get_qwen_context_colors(self, context):
        fallback = (1.0, 0.0, 1.0)
        background = (1.0, 0.0, 1.0)
        if context.scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}:
            fallback = tuple(context.scene.qwen_guidance_fallback_color)
            background = tuple(context.scene.qwen_guidance_background_color)
        return fallback, background

    def _apply_qwen_binary_context_mask(self, context, camera_id, fallback_color):
        """Make the exported sequential mask authoritative for Qwen context."""
        if not getattr(context.scene, "qwen_use_projection_write_masks", True):
            return
        if context.scene.qwen_context_render_mode not in {'REPLACE_STYLE', 'ADDITIONAL'}:
            return

        context_path = get_file_path(
            context, "inpaint", subtype="render", camera_id=camera_id
        )
        mask_path = get_file_path(
            context, "inpaint", subtype="visibility", camera_id=camera_id
        )
        context_image = cv2.imread(context_path, cv2.IMREAD_COLOR)
        repair_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if context_image is None or repair_mask is None:
            print(f"[StableGen] Binary Qwen context skipped; missing context or mask for camera {camera_id}")
            return

        repair_mask = regularize_repair_mask(
            repair_mask,
            threshold=127,
            dilation=context.scene.qwen_context_fallback_dilation,
        )
        context_image = apply_binary_context_mask(
            context_image, repair_mask, fallback_color
        )
        cv2.imwrite(context_path, context_image)
        cv2.imwrite(mask_path, repair_mask)

        # These files are generated immediately before their first upload, but
        # invalidate defensively for regeneration/debug workflows.
        if hasattr(self, '_uploaded_images_cache'):
            self._uploaded_images_cache.pop(context_path, None)
            self._uploaded_images_cache.pop(mask_path, None)

    def _reject_qwen_fallback_writes(self, context, camera_id, image_path, fallback_color):
        """Keep residual sentinel pixels out of the projection material."""
        if not getattr(context.scene, "qwen_use_projection_write_masks", True):
            return
        if context.scene.qwen_context_render_mode not in {'REPLACE_STYLE', 'ADDITIONAL'}:
            return
        mask_path = get_file_path(
            context, "inpaint", subtype="visibility", camera_id=camera_id
        )
        write_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        generated = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if write_mask is None or generated is None:
            return
        write_mask = reject_fallback_from_write_mask(
            write_mask,
            generated,
            fallback_color,
            tolerance=getattr(context.scene, "qwen_residual_fallback_tolerance", 8),
        )
        cv2.imwrite(mask_path, write_mask)

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return True  # Allow cancellation
        # Check for other modal operators
        if context.scene.generation_status == 'waiting' or sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        if not os.path.exists(addon_prefs.output_dir):
            cls.poll_message_set("Output directory not set or does not exist (check addon preferences)")
            return False
        if not addon_prefs.server_address or not addon_prefs.server_online:
            cls.poll_message_set("ComfyUI server is not connected")
            return False
        if bpy.app.online_access == False:
            cls.poll_message_set("Blender's online access is disabled (File → Preferences → System)")
            return False
        return True

    def execute(self, context):
        """     
        Executes the operator.         
        :param context: Blender context.         
        :return: {'RUNNING_MODAL'}     
        """
        if ComfyUIGenerate._is_running:
            self.cancel_generate(context)
            return {'FINISHED'}
        
        self._generation_method_on_start = context.scene.generation_method

        # Clear the upload cache at the start of a new generation
        self._uploaded_images_cache.clear()
        
        # Timestamp for output directory
        if context.scene.generation_mode == 'standard':
            context.scene.output_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        
        # If UV inpainting and we're in prompt collection mode, collect prompts first.
        if context.scene.generation_method == 'uv_inpaint' and self.show_prompt_dialog:
            self._object_prompts[self.current_object_name] = self.current_object_prompt
            if self.mesh_index < len(self._to_texture) - 1:
                self.mesh_index += 1
                self.current_object_name = self._to_texture[self.mesh_index]
                self.current_object_prompt = ""
                return context.window_manager.invoke_props_dialog(self, width=400)
            else:
                self.show_prompt_dialog = False

        
        context.scene.generation_status = 'running'
        context.scene.sg_last_gen_error = False
        ComfyUIGenerate._is_running = True

        print("[StableGen] Executing ComfyUI Generation")

        if context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein') and not context.scene.generation_mode == 'project_only':
            context.scene.generation_method = 'sequential' # Force sequential for edit models

        render = bpy.context.scene.render

        # Force 100% resolution scale so depth/guide map renders match the
        # base resolution_x × resolution_y.  The Qwen workflow derives its
        # output size from the uploaded guidance map, so a non-100% percentage
        # causes it to generate at the wrong resolution.
        self._original_resolution_percentage = render.resolution_percentage
        self._original_resolution_x = render.resolution_x
        self._original_resolution_y = render.resolution_y
        render.resolution_percentage = 100

        resolution_x = render.resolution_x
        resolution_y = render.resolution_y
        total_pixels = resolution_x * resolution_y

        # Qwen Image Edit benefits from 112-aligned resolution (LCM of VAE=8,
        # ViT patch=14, spatial merge=2×14=28, ViT window=112) to avoid
        # subtle pixel shifts between the latent, VAE and CLIP grids.
        # FLUX.2 Klein uses 16x latent downscale, so 16-aligned is needed.
        use_qwen_alignment = (
            context.scene.model_architecture.startswith('qwen')
            and getattr(context.scene, 'qwen_rescale_alignment', False)
        )
        if use_qwen_alignment:
            align_step = 112
        elif context.scene.model_architecture == 'flux2_klein':
            align_step = 16
        else:
            align_step = 8

        target_px = int(getattr(context.scene, 'auto_rescale_target_mp', 1.0) * 1_000_000)
        upper_bound = int(target_px * 1.2)
        lower_bound = int(target_px * 0.8)

        if context.scene.auto_rescale and ((total_pixels > upper_bound or total_pixels < lower_bound) or (resolution_x % align_step != 0 or resolution_y % align_step != 0)):
            render.resolution_x, render.resolution_y = choose_aligned_resolution(
                resolution_x, resolution_y, target_px, align_step
            )
            self.report({'INFO'}, f"Resolution automatically rescaled to {render.resolution_x}x{render.resolution_y}.")

        elif total_pixels > upper_bound:
            target_mp = target_px / 1_000_000
            self.report({'WARNING'}, f"High resolution detected. Resolutions above {target_mp:.1f} MP may reduce performance and quality.")
        
        self._cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
        if not self._cameras:
            self.report({'ERROR'}, "No cameras found in the scene. Use 'Add Cameras' to create them.")
            context.scene.generation_status = 'idle'
            context.scene.sg_last_gen_error = True
            ComfyUIGenerate._is_running = False
            render.resolution_percentage = self._original_resolution_percentage
            render.resolution_x = self._original_resolution_x
            render.resolution_y = self._original_resolution_y
            return {'CANCELLED'}
        order_names = []
        if context.scene.sg_use_custom_camera_order and len(context.scene.sg_camera_order) > 0:
            order_names = [item.name for item in context.scene.sg_camera_order]
        self._cameras = order_cameras(self._cameras, order_names)

        # Hide crop and label overlays during generation
        _sg_remove_crop_overlay()
        _sg_hide_label_overlay()

        # Auto-rescale per-camera resolutions (if any cameras have sg_res_x/y)
        if context.scene.auto_rescale:
            for cam in self._cameras:
                if "sg_res_x" in cam and "sg_res_y" in cam:
                    crx, cry = int(cam["sg_res_x"]), int(cam["sg_res_y"])
                    c_total = crx * cry
                    if (c_total > upper_bound or c_total < lower_bound) or (crx % align_step != 0 or cry % align_step != 0):
                        crx, cry = choose_aligned_resolution(
                            crx, cry, target_px, align_step
                        )
                        cam["sg_res_x"] = crx
                        cam["sg_res_y"] = cry

        self._view_plan = []
        for index, cam in enumerate(self._cameras):
            if "sg_view_uuid" not in cam:
                cam["sg_view_uuid"] = uuid.uuid4().hex
            cam_res_x = int(cam.get("sg_res_x", render.resolution_x))
            cam_res_y = int(cam.get("sg_res_y", render.resolution_y))
            self._view_plan.append(ViewRecord(
                view_id=str(cam["sg_view_uuid"]),
                index=index,
                name=cam.name,
                camera=cam,
                resolution_x=cam_res_x,
                resolution_y=cam_res_y,
            ))

        revision_dir = get_generation_dirs(context)["revision"]
        os.makedirs(revision_dir, exist_ok=True)
        with open(os.path.join(revision_dir, "view_plan.json"), "w", encoding="utf-8") as manifest_file:
            json.dump(
                {"schema": 1, "views": [record.to_manifest() for record in self._view_plan]},
                manifest_file,
                indent=2,
            )
        self._selected_camera_ids = [i for i, cam in enumerate(self._cameras) if cam in bpy.context.selected_objects] #TEST
        if len(self._selected_camera_ids) == 0:
            self._selected_camera_ids = list(range(len(self._cameras))) # All cameras selected if none are selected
        
        # Check if there is at least one ControlNet unit
        controlnet_units = getattr(context.scene, "controlnet_units", [])
        if not controlnet_units and not (context.scene.use_flux_lora and context.scene.model_architecture == 'flux1') and context.scene.model_architecture != 'flux2_klein':
            self.report({'ERROR'}, "At least one ControlNet unit is required. Add one in the Guidance section below.")
            context.scene.generation_status = 'idle'
            context.scene.sg_last_gen_error = True
            ComfyUIGenerate._is_running = False
            render.resolution_percentage = self._original_resolution_percentage
            render.resolution_x = self._original_resolution_x
            render.resolution_y = self._original_resolution_y
            return {'CANCELLED'}
        
        # If there are curves within the scene, warn the user
        if any(obj.type == 'CURVE' for obj in bpy.context.view_layer.objects):
            self.report({'WARNING'}, "Curves detected in the scene. Consider using 'Convert Curves to Mesh' in the Tools section.")
        
        if context.scene.generation_mode == 'project_only':
            print(f"[StableGen] Reprojecting images for {len(self._cameras)} cameras")
        elif context.scene.generation_mode == 'standard':
            print(f"[StableGen] Generating images for {len(self._cameras)} cameras")
        else:
            print(f"[StableGen] Regenerating images for {len(self._selected_camera_ids)} selected cameras")

        if context.scene.texture_objects == 'selected':
            self._to_texture = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            # If empty, cancel the operation
            if not self._to_texture:
                self.report({'ERROR'}, "No mesh objects selected for texturing.")
                context.scene.generation_status = 'idle'
                context.scene.sg_last_gen_error = True
                ComfyUIGenerate._is_running = False
                render.resolution_percentage = self._original_resolution_percentage
                render.resolution_x = self._original_resolution_x
                render.resolution_y = self._original_resolution_y
                return {'CANCELLED'}
        else: # all
            self._to_texture = [obj for obj in bpy.context.view_layer.objects if obj.type == 'MESH' and not obj.hide_get()]

        # Find all mesh objects, check their material ids and store the highest one
        for obj in self._to_texture:
            for slot in obj.material_slots:
                material_id = get_last_material_index(obj)
                if (material_id > self._material_id):
                    self._material_id = material_id
            # Check if there's room for the projection buffer UV map (only 1 slot needed)
            # Projection UV data is stored as attributes (no slot limit), but we need
            # 1 temporary buffer UV slot for the UV Project modifier
            has_buffer = obj.data.uv_layers.get("_SG_ProjectionBuffer") is not None
            if not has_buffer and len(obj.data.uv_layers) >= 8:
                self.report({'ERROR'}, "Not enough UV map slots (max 8). Remove an existing UV map on the object to free a slot for the projection buffer.")
                context.scene.generation_status = 'idle'
                context.scene.sg_last_gen_error = True
                ComfyUIGenerate._is_running = False
                render.resolution_percentage = self._original_resolution_percentage
                render.resolution_x = self._original_resolution_x
                render.resolution_y = self._original_resolution_y
                return {'CANCELLED'}

        if not context.scene.overwrite_material or self._material_id == -1 or (context.scene.generation_method == 'local_edit' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'local_edit')):
            self._material_id += 1

        self._controlnet_units = list(controlnet_units)

        # Prepare for generating
        if context.scene.generation_method == 'grid':
            self._threads_left = 1
        if context.scene.generation_method == 'uv_inpaint':
            self._threads_left = len(self._to_texture)
        else:
            self._threads_left = len(self._cameras)

        self._original_visibility = {}
        if context.scene.texture_objects == 'selected':
            # Hide unselected objects for rendering
            for obj in bpy.context.view_layer.objects:
                if obj.type == 'MESH' and obj not in self._to_texture:
                    # Save original visibility
                    self._original_visibility[obj.name] = obj.hide_render
                    obj.hide_render = True


        # UV inpainting mode preparation
        if context.scene.generation_method == 'uv_inpaint':
            # Check if there are baked textures for all objects
            
            if self.show_prompt_dialog:
                # Start the prompt collection process with the first object
                if not self._object_prompts:  # Only if prompts haven't been collected
                    self.current_object_name = self._to_texture[0].name
                    return context.window_manager.invoke_props_dialog(self, width=400)
                
            # Continue with normal execution if all prompts are collected
            for obj in self._to_texture:
                # Use get_file_path to check for baked texture existence
                baked_texture_path = get_file_path(context, "baked", object_name=obj.name)
                if not os.path.exists(baked_texture_path):
                    # Bake the texture if it doesn't exist
                    self._stage = f"Baking UV Textures ({obj.name})"
                    prepare_baking(context)
                    unwrap(obj, method='pack', overlap_only=True)
                    bake_texture(context, obj, texture_resolution=2048, output_dir=get_dir_path(context, "baked"))
                
                # Check if the material is compatible (uses projection shader)
                active_material = obj.active_material
                if not active_material or not active_material.use_nodes:
                    error = True
                else:
                    # Check if the last node before the output is a color mix node or a bsdf shader node with a color mix node before it
                    output_node = None
                    for node in active_material.node_tree.nodes:
                        if node.type == 'OUTPUT_MATERIAL':
                            output_node = node
                            break
                    if not output_node:
                        error = True
                    else:
                        # Check if the last node before the output is a color mix node or a bsdf shader node with a color mix node before it
                        for link in output_node.inputs[0].links:
                            if link.from_node.type == 'MIX_RGB' or (link.from_node.type == 'BSDF_PRINCIPLED' and any(n.type == 'MIX_RGB' for n in link.from_node.inputs)):
                                error = False
                                break
                        else:
                            error = True
                if error:
                    self.report({'ERROR'}, f"Cannot use UV inpainting with the material of object '{obj.name}'. The generated StableGen material must be the active material.")
                    context.scene.generation_status = 'idle'
                    ComfyUIGenerate._is_running = False
                    render.resolution_percentage = self._original_resolution_percentage
                    render.resolution_x = self._original_resolution_x
                    render.resolution_y = self._original_resolution_y
                    return {'CANCELLED'}
                    
                # Export visibility masks for each object
                self._stage = f"Computing Visibility ({obj.name})"
                export_visibility(context, None, obj)

        if context.scene.view_blend_use_color_match and self._to_texture:
            self._stage = "Matching Colors"
            # Use the first target object as the reference for viewport color
            ref_np = _get_viewport_ref_np(self._to_texture[0])
            if ref_np is not None:
                # Apply color match to ALL generated camera images for this material
                for cam_idx, cam in enumerate(self._cameras):
                    image_path = get_file_path(
                        context,
                        "generated",
                        camera_id=cam_idx,
                        material_id=self._material_id,
                    )
                    _apply_color_match_to_file(
                        image_path=image_path,
                        ref_rgb=ref_np,
                        scene=context.scene,
                    )
        
        self.prompt_text = context.scene.comfyui_prompt

        self._progress = 0.0
        if context.scene.generation_mode == 'project_only':
            self._stage = "Reprojecting"
        else:
            self._stage = "Starting"
        redraw_ui(context)
        self._current_image = 0
        self._total_images = len(self._cameras)
        if context.scene.generation_method == 'grid':
            self._total_images = 1
            if context.scene.refine_images:
                self._total_images += len(self._cameras)  # Add refinement steps
        elif context.scene.generation_method == 'uv_inpaint':
            self._total_images = len(self._to_texture)

        # Regenerate mode preparation
        if context.scene.generation_mode == 'regenerate_selected':
            if context.scene.generation_method == 'sequential':
                # Sequential regeneration: reset all cameras from the first
                # selected onward so the projection sequence replays correctly.
                # Non-selected cameras reuse their existing images but still
                # get reprojected, keeping subsequent cameras' context intact.
                first_selected = min(self._selected_camera_ids)
                ids = [(cid, self._material_id)
                       for cid in range(first_selected, len(self._cameras))]
                reinstate_compare_nodes(context, self._to_texture, ids)
                self._current_image = first_selected
                self._threads_left = len(self._cameras) - first_selected
            else:
                # Non-sequential modes: only reset selected cameras
                ids = [(cid, self._material_id)
                       for cid in self._selected_camera_ids]
                reinstate_compare_nodes(context, self._to_texture, ids)

        # Add modal timer
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)       
        print("[StableGen] Starting thread") 
        if context.scene.generation_method == 'grid':
            self._thread = threading.Thread(target=self.async_generate, args=(context,))
        else:
            _start_cam = self._current_image  # 0 normally, first_selected for sequential regen
            self._thread = threading.Thread(target=self.async_generate, args=(context, _start_cam))
        
        self._thread.start()

        return {'RUNNING_MODAL'}


    def modal(self, context, event):
        """     
        Handles modal events.         
        :param context: Blender context.         
        :param event: Blender event.         
        :return: {'PASS_THROUGH'}     
        """
        if event.type == 'TIMER':
            redraw_ui(context)

            if not self._thread.is_alive():
                context.window_manager.event_timer_remove(self._timer)
                ComfyUIGenerate._is_running = False
                # Restore resolution_percentage that was forced to 100 in execute()
                if hasattr(self, '_original_resolution_percentage'):
                    bpy.context.scene.render.resolution_percentage = self._original_resolution_percentage
                    bpy.context.scene.render.resolution_x = self._original_resolution_x
                    bpy.context.scene.render.resolution_y = self._original_resolution_y
                # Restore original visibility for non-selected objects
                if context.scene.texture_objects == 'selected':
                    for obj in bpy.context.view_layer.objects:
                        if obj.type == 'MESH' and obj.name in self._original_visibility:
                            obj.hide_render = self._original_visibility[obj.name]
                if self._error:
                    if self._error == "'25'" or self._error == "'111'" or self._error == "'5'":
                        # Probably canceled by user, quietly return
                        context.scene.generation_status = 'idle'
                        context.scene.sg_last_gen_error = True
                        self.report({'WARNING'}, "Generation cancelled.")
                        _sg_restore_square_display(context.scene)
                        _sg_ensure_crop_overlay()
                        _sg_restore_label_overlay()
                        remove_empty_dirs(context)
                        return {'CANCELLED'}
                    self.report({'ERROR'}, self._error)
                    _sg_restore_square_display(context.scene)
                    _sg_ensure_crop_overlay()
                    _sg_restore_label_overlay()
                    remove_empty_dirs(context)
                    context.scene.generation_status = 'idle'
                    context.scene.sg_last_gen_error = True
                    return {'CANCELLED'}
                if not context.scene.generation_mode == 'project_only':
                    self.report({'INFO'}, "Generation complete.")
                
                # Reset discard factor if enabled
                if (context.scene.discard_factor_generation_only and
                        (self._generation_method_on_start == 'sequential' or context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein'))):
                    
                    new_discard_angle = context.scene.discard_factor_after_generation
                    print(f"[StableGen] Resetting discard angle in material nodes to {new_discard_angle}...")

                    for obj in self._to_texture:
                        if not obj.active_material or not obj.active_material.use_nodes:
                            continue
                        
                        nodes = obj.active_material.node_tree.nodes
                        for node in nodes:
                            # OSL script nodes (internal or external)
                            if node.type == 'SCRIPT':
                                if 'AngleThreshold' in node.inputs:
                                    node.inputs['AngleThreshold'].default_value = new_discard_angle
                            # Native MATH LESS_THAN nodes (Blender 5.1+ native raycast path)
                            elif node.type == 'MATH' and node.operation == 'LESS_THAN' and node.label.startswith('AngleThreshold-'):
                                node.inputs[1].default_value = new_discard_angle
                    
                    print("[StableGen] Discard angle reset complete.")

                # Reset weight exponent if enabled
                if (context.scene.weight_exponent_generation_only and
                        (self._generation_method_on_start == 'sequential' or context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein'))):
                    
                    new_exponent = context.scene.weight_exponent_after_generation
                    print(f"[StableGen] Resetting weight exponent in material nodes to {new_exponent}...")

                    for obj in self._to_texture:
                        if not obj.active_material or not obj.active_material.use_nodes:
                            continue
                        
                        nodes = obj.active_material.node_tree.nodes
                        has_normalized_sharpness = any(
                            node.type == 'MATH'
                            and node.operation == 'POWER'
                            and node.label.startswith('SharpW-')
                            for node in nodes
                        )
                        for node in nodes:
                            if (node.type == 'MATH'
                                    and node.operation == 'POWER'
                                    and node.label.startswith('SharpW-')):
                                node.inputs[1].default_value = new_exponent
                            elif node.type == 'SCRIPT' and 'Power' in node.inputs:
                                # Normalized materials sharpen at SharpW; their
                                # per-camera source must remain the base weight.
                                node.inputs['Power'].default_value = (
                                    1.0 if has_normalized_sharpness else new_exponent
                                )
                            elif (node.type == 'MATH'
                                    and node.operation == 'POWER'
                                    and node.label == 'power_weight'):
                                node.inputs[1].default_value = (
                                    1.0 if has_normalized_sharpness else new_exponent
                                )
                            elif (node.type == 'MATH'
                                    and node.operation == 'MULTIPLY'
                                    and node.label.startswith('InitialViewPrior-')
                                    and "sg_desired_prior" in node):
                                node.inputs[1].default_value = float(
                                    node["sg_desired_prior"]
                                ) ** (1.0 / max(float(new_exponent), 1e-6))
                    
                    print("[StableGen] Weight exponent reset complete.")

                # Baked visibility stores the angle/exponent numerically in
                # mesh attributes, so changing shader nodes alone cannot reset
                # it. Recompute once using the final display settings.
                reset_angle = (
                    context.scene.discard_factor_after_generation
                    if context.scene.discard_factor_generation_only
                    else context.scene.discard_factor
                )
                reset_exponent = (
                    context.scene.weight_exponent_after_generation
                    if context.scene.weight_exponent_generation_only
                    else context.scene.weight_exponent
                )
                if (context.scene.bake_visibility_weights
                        and (context.scene.discard_factor_generation_only
                             or context.scene.weight_exponent_generation_only)
                        and self._cameras):
                    print("[StableGen] Rebaking visibility weights for final display settings...")
                    _bake_visibility_weights(
                        context,
                        self._to_texture,
                        self._cameras,
                        self._material_id,
                        angle_threshold=reset_angle,
                        power_exponent=reset_exponent,
                    )

                # If viewport rendering mode is 'Rendered' and mode is 'regenerate_selected', switch to 'Solid' and then back to 'Rendered' to refresh the viewport
                if context.scene.generation_mode == 'regenerate_selected' and context.area.spaces.active.shading.type == 'RENDERED':
                    context.area.spaces.active.shading.type = 'SOLID'
                    context.area.spaces.active.shading.type = 'RENDERED'
                context.scene.display_settings.display_device = 'sRGB'
                context.scene.view_settings.view_transform = 'Standard'
                _sg_restore_square_display(context.scene)
                _sg_ensure_crop_overlay()
                _sg_restore_label_overlay()
                context.scene.generation_status = 'idle'
                context.scene.sg_last_gen_error = False
                # Clear output directories which are not needed anymore
                addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
                # Save blend file in the output directory if enabled
                if addon_prefs.save_blend_file:
                    blend_dir = get_dir_path(context, "revision")
                    # Save the current blend file in the output directory
                    scene_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
                    if not scene_name:
                        scene_name = context.scene.name
                    blend_file_path = os.path.join(blend_dir, f"{scene_name}_{context.scene.output_timestamp}.blend")
                    # Clean-up unused data blocks
                    bpy.ops.outliner.orphans_purge(do_recursive=True)
                    # Pack resources and save the blend file
                    bpy.ops.file.pack_all()
                    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path, copy=True)
                remove_empty_dirs(context)
                return {'FINISHED'}
            
            # Handle prompt collection for UV inpainting
            if context.scene.generation_method == 'uv_inpaint' and self.show_prompt_dialog:
                current_index = next((i for i, obj in enumerate(self._to_texture) 
                                    if obj.name == self.current_object_name), -1)
                
                # Store the current prompt
                self._object_prompts[self.current_object_name] = self.current_object_prompt
                
                # Move to next object or finish
                if current_index < len(self._to_texture) - 1:
                    self.current_object_name = self._to_texture[current_index + 1].name
                    self.current_object_prompt = ""
                    return context.window_manager.invoke_props_dialog(self, width=400)
                else:
                    self.show_prompt_dialog = False
                    return self.execute(context)

        return {'PASS_THROUGH'}
    
    def cancel_generate(self, context):
        """     
        Cancels the generation process using api.interupt().    
        :param context: Blender context.         
        :return: None     
        """
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        client_id = str(uuid.uuid4())
        data = json.dumps({"client_id": client_id}).encode('utf-8')
        req =  urllib.request.Request("http://{}/interrupt".format(server_address), data=data)
        context.scene.generation_status = 'waiting'
        ComfyUIGenerate._is_running = False
        urllib.request.urlopen(req)
        # Close active WebSocket to unblock recv() immediately
        ws = getattr(ComfyUIGenerate, '_active_ws', None)
        if ws:
            try:
                ws.close()
            except Exception:
                pass
            ComfyUIGenerate._active_ws = None
        remove_empty_dirs(context)

    # ------------------------------------------------------------------
    # Map preparation — runs from the async thread via timer callbacks
    # so that _stage updates are visible in real time through the modal.
    # ------------------------------------------------------------------
    def _prepare_maps(self, context):
        """Render ControlNet / refine maps.  Called at the start of the async
        thread; every Blender render is dispatched to the main thread via
        ``_run_on_main_thread`` so the progress bar can update between calls."""
        controlnet_units = self._controlnet_units
        cameras = self._cameras

        if context.scene.generation_mode in ('standard', 'regenerate_selected'):
            need_depth = (
                any(u["unit_type"] == "depth" for u in controlnet_units)
                or (context.scene.use_flux_lora and context.scene.model_architecture == 'flux1')
                or (context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein')
                    and context.scene.qwen_guidance_map_type == 'depth')
                or (context.scene.model_architecture.startswith('qwen')
                    and context.scene.qwen_generation_method in ('refine', 'local_edit')
                    and context.scene.qwen_refine_use_depth)
            )
            if need_depth and context.scene.generation_method != 'uv_inpaint':
                for i, camera in enumerate(cameras):
                    self._stage = f"Rendering Depth Maps ({i+1}/{len(cameras)})"
                    _i, _cam = i, camera
                    def _render_depth(_i=_i, _cam=_cam):
                        bpy.context.scene.camera = _cam
                        with _SGCameraResolution(context, _cam):
                            self.export_depthmap(context, camera_id=_i)
                    self._run_on_main_thread(_render_depth)
                    if self._error:
                        return
                if context.scene.generation_method == 'grid':
                    self._run_on_main_thread(
                        lambda: self.combine_maps(context, cameras, type="depth"))
                    if self._error:
                        return

            need_canny = any(u["unit_type"] == "canny" for u in controlnet_units)
            if need_canny and context.scene.generation_method != 'uv_inpaint':
                for i, camera in enumerate(cameras):
                    self._stage = f"Rendering Canny Maps ({i+1}/{len(cameras)})"
                    _i, _cam = i, camera
                    def _render_canny(_i=_i, _cam=_cam):
                        bpy.context.scene.camera = _cam
                        with _SGCameraResolution(context, _cam):
                            export_canny(context, camera_id=_i,
                                         low_threshold=context.scene.canny_threshold_low,
                                         high_threshold=context.scene.canny_threshold_high)
                    self._run_on_main_thread(_render_canny)
                    if self._error:
                        return
                if context.scene.generation_method == 'grid':
                    self._run_on_main_thread(
                        lambda: self.combine_maps(context, cameras, type="canny"))
                    if self._error:
                        return

            need_normal = (
                any(u["unit_type"] == "normal" for u in controlnet_units)
                or (context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein')
                    and context.scene.qwen_guidance_map_type == 'normal')
            )
            if need_normal and context.scene.generation_method != 'uv_inpaint':
                for i, camera in enumerate(cameras):
                    self._stage = f"Rendering Normal Maps ({i+1}/{len(cameras)})"
                    _i, _cam = i, camera
                    def _render_normal(_i=_i, _cam=_cam):
                        bpy.context.scene.camera = _cam
                        with _SGCameraResolution(context, _cam):
                            self.export_normal(context, camera_id=_i)
                    self._run_on_main_thread(_render_normal)
                    if self._error:
                        return
                if context.scene.generation_method == 'grid':
                    self._run_on_main_thread(
                        lambda: self.combine_maps(context, cameras, type="normal"))
                    if self._error:
                        return

            # Qwen guidance using Workbench
            if (context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein')
                    and context.scene.qwen_guidance_map_type == 'workbench'
                    and context.scene.generation_method != 'uv_inpaint'):
                workbench_dir = get_dir_path(context, "controlnet")["workbench"]
                for i, camera in enumerate(cameras):
                    self._stage = f"Rendering Workbench ({i+1}/{len(cameras)})"
                    _i, _cam = i, camera
                    def _render_wb(_i=_i, _cam=_cam):
                        bpy.context.scene.camera = _cam
                        with _SGCameraResolution(context, _cam):
                            export_render(context, camera_id=_i,
                                          output_dir=workbench_dir, filename=f"render{_i}")
                    self._run_on_main_thread(_render_wb)
                    if self._error:
                        return
                if context.scene.generation_method == 'grid':
                    self._run_on_main_thread(
                        lambda: self.combine_maps(context, cameras, type="workbench"))
                    if self._error:
                        return

            # Qwen guidance using Viewport
            elif (context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein')
                  and context.scene.qwen_guidance_map_type == 'viewport'
                  and context.scene.generation_method != 'uv_inpaint'):
                viewport_dir = get_dir_path(context, "controlnet")["viewport"]
                for i, camera in enumerate(cameras):
                    self._stage = f"Rendering Viewport ({i+1}/{len(cameras)})"
                    _i, _cam = i, camera
                    def _render_vp(_i=_i, _cam=_cam):
                        bpy.context.scene.camera = _cam
                        with _SGCameraResolution(context, _cam):
                            export_viewport(context, camera_id=_i,
                                            output_dir=viewport_dir, filename=f"viewport{_i}")
                    self._run_on_main_thread(_render_vp)
                    if self._error:
                        return
                if context.scene.generation_method == 'grid':
                    self._run_on_main_thread(
                        lambda: self.combine_maps(context, cameras, type="viewport"))
                    if self._error:
                        return

        # Refine / Local Edit mode — emit images + edge feather masks
        is_refine = (
            context.scene.generation_method in ('refine', 'local_edit')
            or (context.scene.model_architecture.startswith('qwen')
                and context.scene.qwen_generation_method in ('refine', 'local_edit'))
        )
        if is_refine:
            need_feather = (
                context.scene.refine_edge_feather_projection
                and (context.scene.generation_method == 'local_edit'
                     or (context.scene.model_architecture.startswith('qwen')
                         and context.scene.qwen_generation_method == 'local_edit'))
            )
            for i, camera in enumerate(cameras):
                self._stage = f"Preparing Refinement Maps ({i+1}/{len(cameras)})"
                _i, _cam = i, camera
                def _render_refine(_i=_i, _cam=_cam):
                    bpy.context.scene.camera = _cam
                    with _SGCameraResolution(context, _cam):
                        export_emit_image(context, self._to_texture, camera_id=_i)
                        if need_feather:
                            render_edge_feather_mask(
                                context, self._to_texture, _cam, _i,
                                feather_width=context.scene.refine_edge_feather_width,
                                softness=context.scene.refine_edge_feather_softness)
                self._run_on_main_thread(_render_refine)
                if self._error:
                    return

    def async_generate(self, context, camera_id = None):
        """     
        Asynchronously generates the image using ComfyUI.         
        :param context: Blender context.         
        :return: None     
        """
        self._error = None
        self._pbr_maps = {}  # camera_key → {map_name: file_path}
        try:
            # --- Render ControlNet / refine maps with live progress ---
            self._prepare_maps(context)
            if self._error:
                return

            while self._threads_left > 0 and ComfyUIGenerate._is_running and not context.scene.generation_mode == 'project_only':
                # Swap scene resolution to per-camera values if stored.
                # Must use a timer callback so the write happens on the
                # main thread; writing RNA from a background thread would
                # trigger DEG_id_tag_update on a NULL depsgraph and crash.
                if camera_id is not None and camera_id < len(self._cameras):
                    _cam = self._cameras[camera_id]
                    _rx, _ry = _get_camera_resolution(_cam, context.scene)
                    def _swap_resolution():
                        try:
                            context.scene.render.resolution_x = _rx
                            context.scene.render.resolution_y = _ry
                        except Exception as e:
                            self._error = str(e)
                            traceback.print_exc()
                        self._wait_event.set()
                        return None
                    bpy.app.timers.register(_swap_resolution)
                    self._wait_event.wait()
                    self._wait_event.clear()
                    if self._error:
                        return

                # Sequential regeneration: non-selected cameras skip AI
                # generation but still reproject their existing image so
                # subsequent cameras see the correct incremental texture state.
                _is_seq_reproject = (
                    context.scene.generation_mode == 'regenerate_selected'
                    and camera_id not in self._selected_camera_ids
                    and context.scene.generation_method == 'sequential'
                )

                if _is_seq_reproject:
                    self._stage = "Reprojecting Image"
                    self._progress = 0
                    # project_image patches the material tree for this camera,
                    # loading the existing generated image from disk.
                    def image_reproject_callback():
                        try:
                            redraw_ui(context)
                            project_image(
                                context, self._to_texture, self._material_id,
                                stop_index=self._current_image, cameras=self._cameras,
                            )
                        except Exception as e:
                            self._error = str(e)
                            traceback.print_exc()
                        self._wait_event.set()
                        return None
                    bpy.app.timers.register(image_reproject_callback)
                    self._wait_event.wait()
                    self._wait_event.clear()
                    if self._error:
                        return

                elif context.scene.steps != 0 and not (context.scene.generation_mode == 'regenerate_selected' and camera_id not in self._selected_camera_ids):
                    # Prepare Image Info for Upload
                    controlnet_info = {}
                    mask_info = None
                    render_info = None
                    ipadapter_ref_info = None

                    # Get info for controlnet images for the current camera or grid
                    if context.scene.generation_method != 'uv_inpaint':
                        controlnet_info["depth"] = self._get_uploaded_image_info(context, "controlnet", subtype="depth", camera_id=camera_id)
                        controlnet_info["canny"] = self._get_uploaded_image_info(context, "controlnet", subtype="canny", camera_id=camera_id)
                        controlnet_info["normal"] = self._get_uploaded_image_info(context, "controlnet", subtype="normal", camera_id=camera_id)
                    else: # UV Inpainting
                        current_obj_name = self._to_texture[self._current_image].name
                        mask_info = self._get_uploaded_image_info(context, "uv_inpaint", subtype="visibility", object_name=current_obj_name)
                        render_info = self._get_uploaded_image_info(context, "baked", object_name=current_obj_name)

                    # Get info for refine/sequential render/mask inputs
                    if context.scene.generation_method in ('refine', 'local_edit'):
                        render_info = self._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=camera_id)
                    elif context.scene.generation_method == 'sequential' and self._current_image > 0:
                        render_info = self._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=self._current_image)
                        mask_info = self._get_uploaded_image_info(context, "inpaint", subtype="visibility", camera_id=self._current_image)

                    # Get info for IPAdapter reference image
                    if context.scene.use_ipadapter:
                        ipadapter_ref_info = self._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(context.scene.ipadapter_image))
                    elif context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_mode == 'trellis2_input':
                        # Use the TRELLIS.2 input image as IPAdapter reference
                        t2_path = getattr(context.scene, 'trellis2_last_input_image', '')
                        if t2_path and os.path.exists(bpy.path.abspath(t2_path)):
                            ipadapter_ref_info = self._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(t2_path))
                    elif context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_mode == 'original_render' and context.scene.generation_method == 'local_edit':
                        # Use the existing texture render from this camera's viewpoint as IPAdapter reference
                        ipadapter_ref_info = self._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=camera_id)
                    elif context.scene.sequential_ipadapter and self._current_image > 0:
                        cam_id = 0 if context.scene.sequential_ipadapter_mode == 'first' else self._current_image - 1
                        ipadapter_ref_info = self._get_uploaded_image_info(context, "generated", camera_id=cam_id, material_id=self._material_id)

                    # Filter out None values from controlnet_info
                    controlnet_info = {k: v for k, v in controlnet_info.items() if v is not None}
                    # End Prepare Image Info

                    # Generate image without ControlNet if needed
                    if context.scene.generation_mode == 'standard' and camera_id == 0 and (context.scene.generation_method == 'sequential' or context.scene.generation_method in ('refine', 'local_edit'))\
                            and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate and not context.scene.use_ipadapter and context.scene.sequential_ipadapter_mode == 'first'\
                            and context.scene.sequential_ipadapter_mode != 'trellis2_input':
                        self._stage = "Generating Reference Image"
                        # Don't use ControlNet for the first image if sequential_ipadapter_regenerate_wo_controlnet is enabled
                        if context.scene.sequential_ipadapter_regenerate_wo_controlnet:
                            original_strengths = [unit.strength for unit in context.scene.controlnet_units]
                            for unit in context.scene.controlnet_units:
                                unit.strength = 0.0
                    else:
                        self._stage = "Uploading to Server"
                    self._progress = 0
                    
                    # Generate the image
                    if context.scene.generation_method in ('refine', 'local_edit'):
                        if context.scene.model_architecture == 'flux1':
                            image = self.workflow_manager.refine_flux(context, controlnet_info=controlnet_info, render_info=render_info, ipadapter_ref_info=ipadapter_ref_info)
                        else:
                            image = self.workflow_manager.refine(context, controlnet_info=controlnet_info, render_info=render_info, ipadapter_ref_info=ipadapter_ref_info)
                    elif context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method in ('refine', 'local_edit'):
                        image = self.workflow_manager.generate_qwen_refine(context, camera_id=camera_id)
                    elif context.scene.generation_method == 'uv_inpaint':
                        if context.scene.model_architecture == 'flux1':
                            image = self.workflow_manager.refine_flux(context, mask_info=mask_info, render_info=render_info)
                        else:
                            image = self.workflow_manager.refine(context, mask_info=mask_info, render_info=render_info)
                    elif context.scene.generation_method == 'sequential':
                        if self._current_image == 0:
                            if context.scene.model_architecture == 'flux1':
                                image = self.workflow_manager.generate_flux(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                            elif context.scene.model_architecture == 'qwen_image_edit':
                                image = self.workflow_manager.generate_qwen_edit(context, camera_id=camera_id)
                            elif context.scene.model_architecture == 'flux2_klein':
                                image = self.workflow_manager.generate_flux2_klein(context, camera_id=camera_id)
                            else:
                                image = self.workflow_manager.generate(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                        else:
                            self._stage = "Preparing Next Camera"
                            self._progress = 0
                            def context_callback():
                                try:
                                    # Export visibility mask and render for the current camera, we need to use a callback to be in the main thread
                                    # Visibility is rendered from the *current* camera's viewpoint
                                    # (export_visibility internally picks the next camera after _vis_cam),
                                    # so use the current camera's resolution for the render.
                                    _vis_cam = self._cameras[self._current_image - 1]
                                    _cur_cam = self._cameras[self._current_image] if self._current_image < len(self._cameras) else _vis_cam
                                    with _SGCameraResolution(context, _cur_cam):
                                        export_visibility(
                                            context,
                                            self._to_texture,
                                            camera_visibility=_vis_cam,
                                            camera_render=_cur_cam,
                                            camera_id=self._current_image,
                                        ) # Export mask for current view
                                    _emit_cam = _cur_cam
                                    context.scene.camera = _emit_cam
                                    with _SGCameraResolution(context, _emit_cam):
                                        if context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein'): # export custom bg and fallback for Qwen/Klein image edit
                                            fallback_color, background_color = self._get_qwen_context_colors(context)
                                            export_emit_image(context, self._to_texture, camera_id=self._current_image, bg_color=background_color, fallback_color=fallback_color) # Export render for next view
                                            self._apply_qwen_binary_context_mask(
                                                context,
                                                self._current_image,
                                                fallback_color,
                                            )
                                        else:
                                            # Use a gray (neutral) background and fallback for other architectures
                                            export_emit_image(context, self._to_texture, camera_id=self._current_image, bg_color=(0.5, 0.5, 0.5), fallback_color=(0.5, 0.5, 0.5))
                                except Exception as e:
                                    self._error = str(e)
                                    traceback.print_exc()
                                self._wait_event.set()
                                return None
                            bpy.app.timers.register(context_callback)
                            self._wait_event.wait()
                            self._wait_event.clear()
                            if self._error:
                                return
                            self._stage = "Uploading to Server"
                            # Get info for the previous render and mask
                            render_info = self._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=self._current_image)
                            mask_info = self._get_uploaded_image_info(context, "inpaint", subtype="visibility", camera_id=self._current_image)

                            if context.scene.model_architecture == 'flux1':
                                image = self.workflow_manager.refine_flux(context, controlnet_info=controlnet_info, render_info=render_info, mask_info=mask_info, ipadapter_ref_info=ipadapter_ref_info)
                            elif context.scene.model_architecture == 'qwen_image_edit':
                                image = self.workflow_manager.generate_qwen_edit(context, camera_id=camera_id)
                            elif context.scene.model_architecture == 'flux2_klein':
                                image = self.workflow_manager.generate_flux2_klein(context, camera_id=camera_id)
                            else:
                                image = self.workflow_manager.refine(context, controlnet_info=controlnet_info, render_info=render_info, mask_info=mask_info, ipadapter_ref_info=ipadapter_ref_info)
                    else: # Grid or Separate
                        if context.scene.model_architecture == 'flux1':
                            image = self.workflow_manager.generate_flux(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                        elif context.scene.model_architecture == 'qwen_image_edit':
                            image = self.workflow_manager.generate_qwen_edit(context, camera_id=camera_id)
                        elif context.scene.model_architecture == 'flux2_klein':
                            image = self.workflow_manager.generate_flux2_klein(context, camera_id=camera_id)
                        else:
                            image = self.workflow_manager.generate(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)

                    if image == {"error": "conn_failed"}:
                        if not self._error:
                            self._error = "Connection to ComfyUI server failed."
                        return # Error message set

                    if (context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein') and
                            context.scene.generation_method == 'sequential' and
                            self._current_image > 0 and
                            context.scene.qwen_context_cleanup and
                            context.scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}):
                        image = self._apply_qwen_context_cleanup(context, image)
                    
                    # Save the generated image using new path structure
                    if context.scene.generation_method == 'uv_inpaint':
                        image_path = get_file_path(context, "generated_baked", object_name=self._to_texture[self._current_image].name, material_id=self._material_id)
                    elif camera_id is not None:
                        image_path = get_file_path(context, "generated", camera_id=camera_id, material_id=self._material_id)
                    else: # Grid mode initial generation
                        image_path = get_file_path(context, "generated", filename="generated_image_grid") # Save grid to a specific name
                    
                    with open(image_path, 'wb') as f:
                        f.write(image)

                    if (context.scene.model_architecture == 'qwen_image_edit'
                            and context.scene.generation_method == 'sequential'
                            and self._current_image > 0):
                        fallback_color, _ = self._get_qwen_context_colors(context)
                        self._reject_qwen_fallback_writes(
                            context,
                            self._current_image,
                            image_path,
                            fallback_color,
                        )

                    
                    # Use hack to re-generate the image using IPAdapter to match IPAdapter style
                    if camera_id == 0 and (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate' or context.scene.generation_method in ('refine', 'local_edit'))\
                            and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate and not context.scene.use_ipadapter and context.scene.sequential_ipadapter_mode == 'first':
                                
                        # Restore original strengths
                        if context.scene.sequential_ipadapter_regenerate_wo_controlnet:
                            for i, unit in enumerate(context.scene.controlnet_units):
                                unit.strength = original_strengths[i]
                        self._stage = "Generating Image"
                        context.scene.use_ipadapter = True
                        context.scene.ipadapter_image = image_path
                        ipadapter_ref_info = self._get_uploaded_image_info(context, "custom", filename=image_path)
                        if context.scene.model_architecture == "sdxl":
                            if context.scene.generation_method in ("refine", "local_edit"):
                                image = self.workflow_manager.refine(context, controlnet_info=controlnet_info, render_info=render_info, mask_info=mask_info, ipadapter_ref_info=ipadapter_ref_info)
                            else:
                                image = self.workflow_manager.generate(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                        elif context.scene.model_architecture == "flux1":
                            if context.scene.generation_method in ("refine", "local_edit"):
                                image = self.workflow_manager.refine_flux(context, controlnet_info=controlnet_info, render_info=render_info, mask_info=mask_info, ipadapter_ref_info=ipadapter_ref_info)
                            else:
                                image = self.workflow_manager.generate_flux(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                        context.scene.use_ipadapter = False
                        image_path = image_path.replace(".png", "_ipadapter.png")
                        with open(image_path, 'wb') as f:
                            f.write(image)

                    # ── PBR decomposition is deferred until after ALL cameras ──

                     # Sequential mode callback
                    if context.scene.generation_method == 'sequential':
                        self._stage = "Projecting Image"
                        def image_project_callback():
                            try:
                                redraw_ui(context)
                                project_image(
                                    context, self._to_texture, self._material_id,
                                    stop_index=self._current_image, cameras=self._cameras,
                                )
                            except Exception as e:
                                self._error = str(e)
                                traceback.print_exc()
                            # Set the event to signal the end of the process
                            self._wait_event.set()
                            return None
                        bpy.app.timers.register(image_project_callback)
                        # Wait for the event to be set
                        self._wait_event.wait()
                        self._wait_event.clear()
                        if self._error:
                            return
                        # Update info for the next iteration (if any)
                        if self._current_image < len(self._cameras) - 1:
                            next_camera_id = self._current_image + 1
                            # ControlNet info will be re-fetched at the start of the next loop iteration
                else: # steps == 0, skip generation
                    pass # No image generation needed

                if context.scene.generation_method in ('separate', 'refine', 'local_edit', 'sequential') or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method in ('refine', 'local_edit')):
                    self._current_image += 1
                    self._threads_left -= 1
                    if self._threads_left > 0:
                        self._progress = 0
                    if camera_id is not None: # Increment camera_id only if it was initially provided
                        camera_id += 1

                elif context.scene.generation_method == 'uv_inpaint':
                    self._current_image += 1
                    self._threads_left -= 1
                    if self._threads_left > 0:
                        self._progress = 0

                elif context.scene.generation_method == 'grid':
                    # Split the generated grid image back into multiple images
                    self.split_generated_grid(context, self._cameras)
                    if context.scene.refine_images:
                        for i, _ in enumerate(self._cameras):
                            self._stage = f"Refining Image {i+1}/{len(self._cameras)}"
                            self._current_image = i + 1
                            # Refine the split images
                            refine_cn_info = {
                                "depth": self._get_uploaded_image_info(context, "controlnet", subtype="depth", camera_id=i),
                                "canny": self._get_uploaded_image_info(context, "controlnet", subtype="canny", camera_id=i),
                                "normal": self._get_uploaded_image_info(context, "controlnet", subtype="normal", camera_id=i)
                            }
                            refine_cn_info = {k: v for k, v in refine_cn_info.items() if v is not None}
                            refine_render_info = self._get_uploaded_image_info(context, "generated", camera_id=i, material_id=self._material_id)

                            if context.scene.model_architecture == 'flux1':
                                image = self.workflow_manager.refine_flux(context, controlnet_info=refine_cn_info, render_info=refine_render_info)
                            else:
                                image = self.workflow_manager.refine(context, controlnet_info=refine_cn_info, render_info=refine_render_info)

                            if image == {"error": "conn_failed"}:
                                self._error = "Failed to connect to ComfyUI server."
                                return
                            # Overwrite the split image with the refined one
                            image_path = get_file_path(context, "generated", camera_id=i, material_id=self._material_id)
                            with open(image_path, 'wb') as f:
                                f.write(image)
                    self._threads_left = 0
                
        except Exception as e:
            self._error = str(e)
            traceback.print_exc()
            return

        # ── PBR Decomposition (runs after ALL cameras are generated) ──
        if getattr(context.scene, 'pbr_decomposition', False):
            self._pbr_maps = {}
            # Collect camera images that need PBR decomposition
            camera_images = {}  # cam_idx → image_path
            per_camera_missing = {}  # cam_idx → set of missing map names (reproject only)
            num_cameras = len(self._cameras)
            for cam_idx in range(num_cameras):
                cam_image_path = get_file_path(
                    context, "generated", camera_id=cam_idx,
                    material_id=self._material_id,
                )
                if os.path.exists(cam_image_path):
                    # In project_only (reproject) mode, reuse existing PBR
                    # maps when all enabled maps are already on disk.  This
                    # avoids re-running the slow ComfyUI decomposition.
                    if context.scene.generation_mode == 'project_only':
                        existing, missing = self._find_existing_pbr_maps(
                            context, cam_idx)
                        if not missing:
                            # All maps present — fully reuse
                            self._ensure_raw_copies(
                                context, cam_idx, existing)
                            self._pbr_maps[cam_idx] = existing
                            print(f"[StableGen] Reusing existing PBR maps "
                                  f"for camera {cam_idx}")
                            continue
                        # Some maps exist, some are missing — seed existing
                        # maps so the batched function only generates what's
                        # missing.
                        if existing:
                            self._ensure_raw_copies(
                                context, cam_idx, existing)
                            self._pbr_maps[cam_idx] = existing
                        per_camera_missing[cam_idx] = missing
                        print(f"[StableGen] PBR maps missing for camera "
                              f"{cam_idx}: {missing}, running selective decomposition…")
                    camera_images[cam_idx] = cam_image_path
            if camera_images:
                self._run_pbr_decomposition_batched(
                    context, camera_images,
                    per_camera_missing=per_camera_missing if per_camera_missing else None)
                if self._error:
                    return

            # ── Batch post-processing (albedo sat/contrast, roughness scale) ──
            # Runs after ALL cameras have raw maps saved, so auto-saturation
            # can average across all cameras uniformly.
            if self._pbr_maps:
                self._apply_albedo_postprocessing_batch(context)

            # ── Persist per-map settings hashes so future reprojects can
            #    detect stale maps and only regenerate what changed. ──
            if camera_images:
                for cam_idx in camera_images:
                    self._save_pbr_settings(context, cam_idx)

        def image_project_callback():
            if context.scene.generation_method == 'sequential':
                # In sequential mode, projection happened per-camera inside the loop.
                # PBR is projected onto the existing material after all cameras.
                if getattr(context.scene, 'pbr_decomposition', False) and hasattr(self, '_pbr_maps') and self._pbr_maps:
                    from .pbr_projection import project_pbr_to_bsdf
                    project_pbr_to_bsdf(
                        context, self._to_texture, self._pbr_maps,
                        material_id=self._material_id
                    )
                    if getattr(context.scene, 'pbr_auto_lighting', False):
                        _pbr_setup_studio_lights(context, self._to_texture)
                return None
            self._stage = "Projecting Image"
            redraw_ui(context)
            if context.scene.generation_method != 'uv_inpaint':
                project_image(
                    context, self._to_texture, self._material_id,
                    cameras=self._cameras,
                )
            else:
                # Apply the UV inpainted textures to each mesh
                from .rendering import apply_uv_inpaint_texture
                for obj in self._to_texture:
                    texture_path = get_file_path(
                        context, "generated_baked", object_name=obj.name, material_id=self._material_id
                    )
                    apply_uv_inpaint_texture(context, obj, texture_path)

            # Project PBR maps onto material after color projection
            if getattr(context.scene, 'pbr_decomposition', False) and hasattr(self, '_pbr_maps') and self._pbr_maps:
                from .pbr_projection import project_pbr_to_bsdf
                project_pbr_to_bsdf(
                    context, self._to_texture, self._pbr_maps,
                    material_id=self._material_id
                )
                if getattr(context.scene, 'pbr_auto_lighting', False):
                    _pbr_setup_studio_lights(context, self._to_texture)

            return None
        
        if context.scene.view_blend_use_color_match and self._to_texture:
            self._stage = "Matching Colors"
            # Use the first object in the target list as the color reference
            ref_np = _get_viewport_ref_np(self._to_texture[0])
            if ref_np is not None:
                # Loop all cameras we generated for
                for cam_idx, cam in enumerate(self._cameras):
                    image_path = get_file_path(
                        context,
                        "generated",
                        camera_id=cam_idx,
                        material_id=self._material_id,
                    )
                    _apply_color_match_to_file(
                        image_path=image_path,
                        ref_rgb=ref_np,
                        scene=context.scene,
                    )

        bpy.app.timers.register(image_project_callback)

        # Update seed based on control parameter
        if context.scene.control_after_generate == 'increment':
            context.scene.seed += 1
        elif context.scene.control_after_generate == 'decrement':
            context.scene.seed -= 1
        elif context.scene.control_after_generate == 'randomize':
            context.scene.seed = np.random.randint(0, 1000000)

    def draw(self, context):
        layout = self.layout
        if context.scene.generation_method == 'uv_inpaint' and self.show_prompt_dialog:
            layout.label(text=f"Enter prompt for object: {self.current_object_name}")
            layout.prop(self, "current_object_prompt", text="")

    def invoke(self, context, event):
        if context.scene.generation_method == 'uv_inpaint':
            # Reset object prompts on every run
            self.show_prompt_dialog = True
            self._object_prompts = {}
            self._to_texture = [obj.name for obj in bpy.context.view_layer.objects if obj.type == 'MESH']
            if context.scene.texture_objects == 'selected':
                self._to_texture = [obj.name for obj in bpy.context.selected_objects if obj.type == 'MESH']
            self.mesh_index = 0
            self.current_object_name = self._to_texture[0] if self._to_texture else ""
            # If "Ask for object prompts" is disabled, don’t prompt per object
            if not context.scene.ask_object_prompts or self._is_running:
                self.show_prompt_dialog = False
                return self.execute(context)
            return context.window_manager.invoke_props_dialog(self, width=400)
        return self.execute(context)
    
    def _dilate_qwen_context_fallback(self, context, camera_id, fallback_color):
        dilation = int(max(0, context.scene.qwen_context_fallback_dilation))
        if dilation <= 0:
            return

        image_path = get_file_path(context, "inpaint", subtype="render", camera_id=camera_id)
        if not image_path or not os.path.exists(image_path):
            return

        try:
            with Image.open(image_path) as img:
                pixel_data = np.array(img.convert("RGBA"))
        except Exception as err:
            print(f"[StableGen] Failed to load context render for dilation at {image_path}: {err}")
            return

        fallback_rgb = np.array([int(round(component * 255.0)) for component in fallback_color], dtype=np.uint8)
        rgb = pixel_data[:, :, :3].astype(np.int16)
        diff = np.abs(rgb - fallback_rgb[np.newaxis, np.newaxis, :])
        mask = np.all(diff <= 3, axis=2)
        if not np.any(mask):
            return

        mask_uint8 = (mask.astype(np.uint8) * 255)
        kernel_size = max(1, dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        dilated_mask = dilated > 0

        pixel_data[dilated_mask, :3] = fallback_rgb
        if pixel_data.shape[2] == 4:
            pixel_data[dilated_mask, 3] = 255

        try:
            Image.fromarray(pixel_data).save(image_path)
        except Exception as err:
            print(f"[StableGen] Failed to save dilated context render at {image_path}: {err}")
            return

        if hasattr(self, '_uploaded_images_cache') and self._uploaded_images_cache is not None:
            self._uploaded_images_cache.pop(os.path.abspath(image_path), None)

    # ── Qwen context cleanup ──────────────────────────────────────────

    def _apply_qwen_context_cleanup(self, context, image_bytes):
        hue_tolerance = max(context.scene.qwen_context_cleanup_hue_tolerance, 0.0)
        value_adjust = context.scene.qwen_context_cleanup_value_adjust
        fallback_color = tuple(context.scene.qwen_guidance_fallback_color)
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                rgba_image = pil_image.convert("RGBA")
                pixel_data = np.array(rgba_image)
        except Exception as err:
            print(f"[StableGen]   Warning: Failed to read Qwen context render for cleanup: {err}")
            traceback.print_exc()
            return image_bytes

        rgb = pixel_data[:, :, :3].astype(np.float32) / 255.0
        alpha = pixel_data[:, :, 3]

        maxc = rgb.max(axis=2)
        minc = rgb.min(axis=2)
        delta = maxc - minc

        hue = np.zeros_like(maxc, dtype=np.float32)
        non_gray = delta > 1e-6
        safe_delta = np.where(non_gray, delta, 1.0)  # avoid divide-by-zero

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        idx = non_gray & (r == maxc)
        hue[idx] = ((g[idx] - b[idx]) / safe_delta[idx]) % 6.0
        idx = non_gray & (g == maxc)
        hue[idx] = ((b[idx] - r[idx]) / safe_delta[idx]) + 2.0
        idx = non_gray & (b == maxc)
        hue[idx] = ((r[idx] - g[idx]) / safe_delta[idx]) + 4.0
        hue = (hue / 6.0) % 1.0

        try:
            fallback_hue = colorsys.rgb_to_hsv(*fallback_color)[0]
        except Exception:
            fallback_hue = 0.0
        hue_tol_normalized = hue_tolerance / 360.0
        if hue_tol_normalized <= 0.0:
            hue_tol_normalized = 0.0

        diff = np.abs(hue - fallback_hue)
        diff = np.minimum(diff, 1.0 - diff)
        target_mask = non_gray & (diff <= hue_tol_normalized)

        if not np.any(target_mask):
            return image_bytes

        value = maxc
        adjusted_value = np.clip(value[target_mask] + value_adjust, 0.0, 1.0)

        updated_rgb = np.array(rgb)
        grayscale_values = np.repeat(adjusted_value[:, None], 3, axis=1)
        updated_rgb[target_mask] = grayscale_values

        updated_pixels = np.empty_like(pixel_data)
        updated_pixels[:, :, :3] = np.clip(np.round(updated_rgb * 255.0), 0, 255).astype(np.uint8)
        updated_pixels[:, :, 3] = alpha

        try:
            buffer = io.BytesIO()
            Image.fromarray(updated_pixels, mode="RGBA").save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as err:
            print(f"[StableGen]   Warning: Failed to write cleaned Qwen context render: {err}")
            traceback.print_exc()
            return image_bytes

    def export_depthmap(self, context, camera_id=None):
        """     
        Exports the depth map of the scene.         
        :param context: Blender context.         
        :param camera_id: ID of the camera.         
        :return: None     
        """
        print("[StableGen] Exporting depth map")
        # Save original settings to restore later.
        original_engine = bpy.context.scene.render.engine
        original_view_transform = bpy.context.scene.view_settings.view_transform
        original_film_transparent = bpy.context.scene.render.film_transparent
        original_use_compositing = bpy.context.scene.render.use_compositing
        original_filepath = bpy.context.scene.render.filepath

        # Set animation frame to 1
        bpy.context.scene.frame_set(1)

        output_dir = get_dir_path(context, "controlnet")["depth"]
        output_file = f"depth_map{camera_id}" if camera_id is not None else "depth_map"

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the active view layer
        view_layer = bpy.context.view_layer

        # Switch to WORKBENCH render engine
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'

        bpy.context.scene.display_settings.display_device = 'sRGB'
        bpy.context.scene.view_settings.view_transform = 'Raw'

        original_pass_z = view_layer.use_pass_z

        # Enable depth pass in the render settings
        view_layer.use_pass_z = True

        # Enable compositor pipeline (may have been disabled by prior GIF export)
        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.use_nodes = True
        node_tree = get_compositor_node_tree(bpy.context.scene)
        nodes = node_tree.nodes
        links = node_tree.links
        
        # Ensure animation format is not selected
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Clear default nodes
        for node in nodes:
            nodes.remove(node)

        # Add render layers node
        render_layers_node = nodes.new(type="CompositorNodeRLayers")
        render_layers_node.location = (0, 0)

        # Add a normalize node (to scale depth values between 0 and 1)
        normalize_node = nodes.new(type="CompositorNodeNormalize")
        normalize_node.location = (200, 0)
        links.new(render_layers_node.outputs["Depth"], normalize_node.inputs[0])

        # Add an invert node to flip the depth map values
        invert_node = nodes.new(type="CompositorNodeInvert")
        invert_node.location = (400, 0)
        # Blender 5.x uses named "Color" input, 4.x uses index 1
        color_input = invert_node.inputs["Color"] if "Color" in invert_node.inputs else invert_node.inputs[1]
        links.new(normalize_node.outputs[0], color_input)

        # Add an output file node
        output_node = nodes.new(type="CompositorNodeOutputFile")
        output_node.location = (600, 0)
        configure_output_node_paths(output_node, output_dir, output_file)
        links.new(invert_node.outputs[0], output_node.inputs[0])

        # Render the scene
        bpy.ops.render.render(write_still=True)

        bpy.context.scene.view_settings.view_transform = 'Standard'

        print(f"[StableGen] Depth map saved to: {os.path.join(output_dir, output_file)}.png")
        
        # Restore original settings
        bpy.context.scene.render.engine = original_engine
        bpy.context.scene.view_settings.view_transform = original_view_transform
        bpy.context.scene.render.film_transparent = original_film_transparent
        bpy.context.scene.render.use_compositing = original_use_compositing
        bpy.context.scene.render.filepath = original_filepath
        view_layer.use_pass_z = original_pass_z

    def export_normal(self, context, camera_id=None):
        """
        Exports the normal map of the scene.
        Areas without geometry will show the neutral color (0.5, 0.5, 1.0).
        :param context: Blender context.
        :param camera_id: ID of the camera.
        :return: None
        """
        print("[StableGen] Exporting normal map")
        bpy.context.scene.frame_set(1)
        output_dir = get_dir_path(context, "controlnet")["normal"]
        output_file = f"normal_map{camera_id}" if camera_id is not None else "normal_map"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        view_layer = bpy.context.view_layer
        original_pass_normal = view_layer.use_pass_normal
        view_layer.use_pass_normal = True

        # Store original settings to restore later.
        original_engine = bpy.context.scene.render.engine
        original_view_transform = bpy.context.scene.view_settings.view_transform
        original_film_transparent = bpy.context.scene.render.film_transparent
        original_use_compositing = bpy.context.scene.render.use_compositing
        original_filepath = bpy.context.scene.render.filepath

        bpy.context.scene.render.engine = get_eevee_engine_id()
        bpy.context.scene.view_settings.view_transform = 'Raw'
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.use_nodes = True

        # Clear existing nodes.
        node_tree = get_compositor_node_tree(bpy.context.scene)
        nodes = node_tree.nodes
        links = node_tree.links
        for node in nodes:
            nodes.remove(node)

        # Create the Render Layers node (provides the baked normal pass).
        render_layers_node = nodes.new(type="CompositorNodeRLayers")
        render_layers_node.location = (0, 0)

        # Create an RGB node set to the neutral normal color (0.5, 0.5, 1.0, 1.0).
        bg_node = nodes.new(type="CompositorNodeRGB")
        bg_node.outputs[0].default_value = (0.5, 0.5, 1.0, 1.0)
        bg_node.location = (0, -200)

        alpha_over_node = nodes.new(type="CompositorNodeAlphaOver")
        alpha_over_node.location = (200, 0)
        # Link the normal pass to the top input.
        links.new(render_layers_node.outputs["Normal"], alpha_over_node.inputs[2])
        # Link the neutral background to the bottom input.
        links.new(bg_node.outputs[0], alpha_over_node.inputs[1])

        # Create the Output File node.
        output_node = nodes.new(type="CompositorNodeOutputFile")
        output_node.location = (400, 0)
        configure_output_node_paths(output_node, output_dir, output_file)
        links.new(alpha_over_node.outputs[0], output_node.inputs[0])
        links.new(render_layers_node.outputs["Alpha"], alpha_over_node.inputs[0])

        bpy.ops.render.render(write_still=True)

        # Restore original settings.
        bpy.context.scene.render.engine = original_engine
        bpy.context.scene.view_settings.view_transform = original_view_transform
        bpy.context.scene.render.film_transparent = original_film_transparent
        bpy.context.scene.render.use_compositing = original_use_compositing
        bpy.context.scene.render.filepath = original_filepath

        view_layer.use_pass_normal = original_pass_normal

        print(f"[StableGen] Normal map saved to: {os.path.join(output_dir, output_file)}.png")

    def combine_maps(self, context, cameras, type):
        """Combines depth maps into a grid."""
        if type == 'depth':
            grid_image_path = get_file_path(context, "controlnet", subtype="depth", camera_id=None, material_id=self._material_id)
        elif type == 'canny':
            grid_image_path = get_file_path(context, "controlnet", subtype="canny", camera_id=None, material_id=self._material_id)
        elif type == 'normal':
            grid_image_path = get_file_path(context, "controlnet", subtype="normal", camera_id=None, material_id=self._material_id)
        elif type == 'workbench':
            grid_image_path = get_file_path(context, "controlnet", subtype="workbench", camera_id=None, material_id=self._material_id)
        elif type == 'viewport':
            grid_image_path = get_file_path(context, "controlnet", subtype="viewport", camera_id=None, material_id=self._material_id)

        # Render depth maps for each camera and combine them into a grid
        depth_maps = []
        for i, camera in enumerate(cameras):
            bpy.context.scene.camera = camera
            if type == 'depth':
                depth_map_path = get_file_path(context, "controlnet", subtype="depth", camera_id=i, material_id=self._material_id)
            elif type == 'canny':
                depth_map_path = get_file_path(context, "controlnet", subtype="canny", camera_id=i, material_id=self._material_id)
            elif type == 'normal':
                depth_map_path = get_file_path(context, "controlnet", subtype="normal", camera_id=i, material_id=self._material_id)
            elif type == 'workbench':
                depth_map_path = get_file_path(context, "controlnet", subtype="workbench", camera_id=i, material_id=self._material_id)
            elif type == 'viewport':
                depth_map_path = get_file_path(context, "controlnet", subtype="viewport", camera_id=i, material_id=self._material_id)
            depth_maps.append(depth_map_path)

        # Combine depth maps into a grid
        grid_image = self.create_grid_image(depth_maps)
        grid_image = self.rescale_to_1mp(grid_image)
        grid_image.save(grid_image_path)
        print(f"[StableGen] Combined depth map grid saved to: {grid_image_path}")

    def create_grid_image(self, image_paths):
        """Creates a grid image from a list of image paths."""
        images = [Image.open(path) for path in image_paths]
        widths, heights = zip(*(i.size for i in images))

        # Calculate grid dimensions to make it as square as possible
        num_images = len(images)
        grid_width = math.ceil(math.sqrt(num_images))
        grid_height = math.ceil(num_images / grid_width)

        max_width = max(widths)
        max_height = max(heights)

        total_width = grid_width * max_width
        total_height = grid_height * max_height

        grid_image = Image.new('RGB', (total_width, total_height))

        x_offset = 0
        y_offset = 0
        for i, img in enumerate(images):
            grid_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_width == 0:
                x_offset = 0
                y_offset += max_height

        return grid_image

    def rescale_to_1mp(self, image):
        """Rescales the image to approximately 1MP."""

        width, height = image.size
        total_pixels = width * height
        scale_factor = (1_000_000 / total_pixels) ** 0.5

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Ensure the new dimensions are divisible by 8 (ComfyUI requirement)
        new_width -= new_width % 8
        new_height -= new_height % 8

        self._grid_height = new_height
        self._grid_width = new_width

        return image.resize((new_width, new_height), Image.LANCZOS)

    def split_generated_grid(self, context, cameras):
        """Splits the generated grid image back into multiple images."""
        grid_image_path = get_file_path(context, "generated", camera_id=None, material_id=self._material_id)

        # Load the generated grid image
        grid_image = Image.open(grid_image_path)

        # Calculate grid dimensions to make it as square as possible
        num_images = len(cameras)
        grid_width = math.ceil(math.sqrt(num_images))
        grid_height = math.ceil(num_images / grid_width)

        max_width = grid_image.width // grid_width
        max_height = grid_image.height // grid_height

        x_offset = 0
        y_offset = 0
        for i in range(num_images):
            bbox = (x_offset, y_offset, x_offset + max_width, y_offset + max_height)
            individual_image = grid_image.crop(bbox)
            individual_image_path = get_file_path(context, "generated", camera_id=i, material_id=self._material_id)
            individual_image.save(individual_image_path)
            print(f"[StableGen] Generated image for camera {i+1} saved to: {individual_image_path}")
            x_offset += max_width
            if (i + 1) % grid_width == 0:
                x_offset = 0
                y_offset += max_height

    def _get_uploaded_image_info(self, context, file_type, subtype=None, filename=None, camera_id=None, object_name=None, material_id=None):
        """
        Gets local path, uploads if needed, caches, and returns ComfyUI upload info.
        Intended to be called within the ComfyUIGenerate operator instance.

        Args:
            self: The instance of the ComfyUIGenerate operator.
            context: Blender context.
            file_type: Type of file (e.g., "controlnet", "generated", "baked").
            subtype: Subtype (e.g., "depth", "render").
            filename: Specific filename if overriding default naming.
            camera_id: Camera index.
            object_name: Object name.
            material_id: Material index.

        Returns:
            dict: Upload info from ComfyUI (containing 'name', etc.) or None if failed/not found.
        """
        effective_material_id = material_id

        # Use the existing get_file_path to determine the canonical local path
        if not file_type == "custom": # Custom files use provided filename directly
            local_path = get_file_path(context, file_type, subtype, filename, camera_id, object_name, effective_material_id)
        else:
            local_path = filename

        # --- Image Modification for 'recent' sequential mode ---
        # Check if we need to modify the image before uploading
        is_recent_mode_ref = (
            file_type == "generated" and
            context.scene.sequential_ipadapter_mode == 'recent' and
            (context.scene.sequential_ipadapter or context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein'))
        )
        
        temp_image_path = None
        upload_path = local_path

        if is_recent_mode_ref:
            desaturate = context.scene.sequential_desaturate_factor
            contrast = context.scene.sequential_contrast_factor

            if desaturate > 0.0 or contrast > 0.0:
                try:
                    with Image.open(local_path) as img:
                        if desaturate > 0.0:
                            enhancer = ImageEnhance.Color(img)
                            img = enhancer.enhance(1.0 - desaturate)
                        
                        if contrast > 0.0:
                            enhancer = ImageEnhance.Contrast(img)
                            img = enhancer.enhance(1.0 - contrast)
                        
                        # Save to a temporary file for upload
                        temp_dir = get_dir_path(context, "temp")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_image_path = os.path.join(temp_dir, f"temp_{os.path.basename(local_path)}")
                        img.save(temp_image_path)
                        upload_path = temp_image_path
                except Exception as e:
                    print(f"[StableGen] Error modifying image {local_path}: {e}. Uploading original.")
                    upload_path = local_path # Fallback to original on error
        # --- End Image Modification ---

        # Use the operator's instance cache variable (self._uploaded_images_cache)
        if not hasattr(self, '_uploaded_images_cache') or self._uploaded_images_cache is None:
            # Initialize cache if it doesn't exist (e.g., first call in execute)
            # Although clearing in execute() is preferred
            self._uploaded_images_cache = {}
            print("[StableGen] Warning: _uploaded_images_cache not found, initializing. Should be cleared in execute().")


        # Check cache first using the absolute local path as the key
        absolute_local_path = os.path.abspath(upload_path)
        cached_info = self._uploaded_images_cache.get(absolute_local_path)
        if cached_info is not None: # Can be None if previous upload failed
            # print(f"Debug: Using cached upload info for: {absolute_local_path}")
            return cached_info # Return cached info (could be None if failed before)

        # File exists locally? If not, we can't upload. Return None. Cache this result.
        if not os.path.exists(absolute_local_path) or not os.path.isfile(absolute_local_path):
            # print(f"Debug: Local file not found or not a file, cannot upload: {absolute_local_path}")
            self._uploaded_images_cache[absolute_local_path] = None # Cache the fact that it's missing/invalid
            return None

        # Not cached and file exists, try to upload it
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        uploaded_info = upload_image_to_comfyui(server_address, absolute_local_path)

        # Store result (the info dict or None if upload failed) in cache
        self._uploaded_images_cache[absolute_local_path] = uploaded_info

        # Clean up the temporary file after upload
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        if uploaded_info:
            return uploaded_info
        else:
            # Upload failed, error message was printed by upload_image_to_comfyui
            # Returning None allows optional inputs to be skipped gracefully.
            # If a *required* image fails to upload, the workflow submission
            # will likely fail later when ComfyUI can't find the input.
            return None


# ── Preview Gallery helpers ───────────────────────────────────────────
