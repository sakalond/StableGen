import os
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import numpy as np
import cv2

import uuid
import json
import urllib.request
import socket
import threading
import requests
import traceback
import io
from datetime import datetime
import math
import colorsys
from PIL import Image, ImageEnhance

import gpu  # pylint: disable=import-error
import blf  # pylint: disable=import-error
from gpu_extras.batch import batch_for_shader  # pylint: disable=import-error

from .util.helpers import prompt_text, prompt_text_img2img, prompt_text_qwen_image_edit # pylint: disable=relative-beyond-top-level
from .render_tools import export_emit_image, export_visibility, export_canny, bake_texture, prepare_baking, unwrap, export_render, export_viewport, render_edge_feather_mask, _SGCameraResolution, _get_camera_resolution, _sg_restore_square_display, _sg_remove_crop_overlay, _sg_ensure_crop_overlay, _sg_hide_label_overlay, _sg_restore_label_overlay # pylint: disable=relative-beyond-top-level
from .utils import get_last_material_index, get_generation_dirs, get_file_path, get_dir_path, remove_empty_dirs, get_compositor_node_tree, configure_output_node_paths, get_eevee_engine_id # pylint: disable=relative-beyond-top-level
from .project import project_image, reinstate_compare_nodes # pylint: disable=relative-beyond-top-level
from .workflows import WorkflowManager
from .util.mirror_color import MirrorReproject, _get_viewport_ref_np, _apply_color_match_to_file
from .timeout_config import get_timeout

# Import wheels
import websocket

def redraw_ui(context):
    """Redraws the UI to reflect changes in the operator's progress and status."""
    for area in context.screen.areas:
        area.tag_redraw()

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
        """     
        Polls whether the operator can be executed.         
        :param context: Blender context.         
        :return: True if the operator can be executed, False otherwise.     
        """
        # Check for other modal operators
        operator = None
        addon_prefs = context.preferences.addons[__package__].preferences
        if not os.path.exists(addon_prefs.output_dir):
            return False
        if not addon_prefs.server_address or not addon_prefs.server_online:
            return False
        if not (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate'):
            return False
        if context.scene.output_timestamp == "":
            return False
        for window in context.window_manager.windows:
                for op in window.modal_operators:
                    if op.bl_idname == 'OBJECT_OT_add_cameras' or op.bl_idname == 'OBJECT_OT_bake_textures' or\
                    op.bl_idname == 'OBJECT_OT_collect_camera_prompts' or op.bl_idname == 'OBJECT_OT_test_stable' or\
                    op.bl_idname == 'OBJECT_OT_stablegen_reproject' or op.bl_idname == 'OBJECT_OT_stablegen_regenerate' \
                          or context.scene.generation_status == 'waiting':
                        operator = op
                        break
                if operator:
                    break
        if operator:
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
                self.report({'ERROR'}, "No mesh objects selected for texturing.")
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
        # Run the generation operator
        bpy.ops.object.test_stable('INVOKE_DEFAULT')

        # Switch to modal and wait for completion
        print("Going modal")
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
        """     
        Polls whether the operator can be executed.         
        :param context: Blender context.         
        :return: True if the operator can be executed, False otherwise.     
        """
        # Check for other modal operators
        operator = None
        if context.scene.output_timestamp == "":
            return False
        for window in context.window_manager.windows:
                for op in window.modal_operators:
                    if op.bl_idname == 'OBJECT_OT_add_cameras' or op.bl_idname == 'OBJECT_OT_bake_textures' or\
                    op.bl_idname == 'OBJECT_OT_collect_camera_prompts' or op.bl_idname == 'OBJECT_OT_test_stable' or\
                    op.bl_idname == 'OBJECT_OT_stablegen_reproject' or op.bl_idname == 'OBJECT_OT_stablegen_regenerate' \
                          or context.scene.generation_status == 'waiting':
                        operator = op
                        break
                if operator:
                    break
        if operator:
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
                self.report({'ERROR'}, f"Camera {i} does not have a corresponding generated image.")
                print(f"{image_path} does not exist")
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
        # Run the generation operator
        bpy.ops.object.test_stable('INVOKE_DEFAULT')

        # Switch to modal and wait for completion
        print("Going modal")
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
    
def upload_image_to_comfyui(server_address, image_path, image_type="input"):
    """
    Uploads an image file to the ComfyUI server's /upload/image endpoint.

    Args:
        server_address (str): The address:port of the ComfyUI server (e.g., "127.0.0.1:8188").
        image_path (str): The local path to the image file to upload.
        image_type (str): The type parameter for the upload (usually "input").

    Returns:
        dict: A dictionary containing the server's response (e.g., {'name': 'filename.png', 'subfolder': '', 'type': 'input'})
              Returns None if the upload fails or file doesn't exist.
    """
    if not os.path.exists(image_path):
        # This is expected for optional files, so don't log as an error
        # print(f"Debug: Image file not found at {image_path}, cannot upload.")
        return None
    if not os.path.isfile(image_path):
        print(f"Error: Path exists but is not a file: {image_path}")
        return None

    upload_url = f"http://{server_address}/upload/image"
    print(f"Uploading {os.path.basename(image_path)} to {upload_url}...")

    try:
        with open(image_path, 'rb') as f:
            # Determine mime type based on extension
            mime_type = 'application/octet-stream' # Default fallback
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_path.lower().endswith('.webp'):
                mime_type = 'image/webp'
            # Add other types if needed (e.g., .bmp, .gif)

            files = {'image': (os.path.basename(image_path), f, mime_type)}
            # 'overwrite': 'true' prevents errors if the same filename is uploaded again
            # useful for re-running generations with the same intermediate files.
            data = {'overwrite': 'true', 'type': image_type}

            # Increased timeout for potentially large images or slow networks
            response = requests.post(upload_url, files=files, data=data, timeout=get_timeout('transfer'))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        print(f"  Upload successful for '{os.path.basename(image_path)}'. Server response: {response_data}")

        # Crucial Validation
        if 'name' not in response_data:
             print(f"  Error: ComfyUI upload response for {os.path.basename(image_path)} missing 'name'. Response: {response_data}")
             return None
        # End Validation

        return response_data # Should contain 'name', often 'subfolder', 'type'

    except requests.exceptions.Timeout:
        print(f"  Error: Timeout uploading image {os.path.basename(image_path)} to {upload_url}.")
    except requests.exceptions.ConnectionError:
        print(f"  Error: Connection failed when uploading image {os.path.basename(image_path)} to {upload_url}. Is ComfyUI running and accessible?")
    except requests.exceptions.HTTPError as e:
         print(f"  Error: HTTP Error {e.response.status_code} uploading image {os.path.basename(image_path)} to {upload_url}.")
         print(f"  Server response content: {e.response.text}") # Show response body on error
    except requests.exceptions.RequestException as e:
        print(f"  Error uploading image {os.path.basename(image_path)} to {upload_url}: {e}")
    except json.JSONDecodeError:
        print(f"  Error decoding ComfyUI response after uploading {os.path.basename(image_path)}. Response text: {response.text}")
    except Exception as e:
        print(f"  An unexpected error occurred during image upload of {os.path.basename(image_path)}: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors

    return None

class ComfyUIGenerate(bpy.types.Operator):
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
        self._wait_event = threading.Event()
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

    @classmethod
    def poll(cls, context):
        """     
        Polls whether the operator can be executed.         
        :param context: Blender context.         
        :return: True if the operator can be executed, False otherwise.     
        """
        # Check for other modal operators
        operator = None
        for window in context.window_manager.windows:
                for op in window.modal_operators:
                    if op.bl_idname == 'OBJECT_OT_add_cameras' or op.bl_idname == 'OBJECT_OT_bake_textures' or op.bl_idname == 'OBJECT_OT_collect_camera_prompts' or context.scene.generation_status == 'waiting':
                        operator = op
                        break
                if operator:
                    break
        if operator:
            return False
        # Check if output directory, model directory, and server address are set
        addon_prefs = context.preferences.addons[__package__].preferences
        if not os.path.exists(addon_prefs.output_dir):
            return False
        if not addon_prefs.server_address or not addon_prefs.server_online:
            return False
        if bpy.app.online_access == False: # Check if online access is disabled
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
        ComfyUIGenerate._is_running = True

        print("Executing ComfyUI Generation")

        if context.scene.model_architecture == 'qwen_image_edit' and not context.scene.generation_mode == 'project_only':
            context.scene.generation_method = 'sequential' # Force sequential for Qwen Image Edit

        render = bpy.context.scene.render
        resolution_x = render.resolution_x
        resolution_y = render.resolution_y
        total_pixels = resolution_x * resolution_y

        # Qwen Image Edit benefits from 112-aligned resolution (LCM of VAE=8,
        # ViT patch=14, spatial merge=2×14=28, ViT window=112) to avoid
        # subtle pixel shifts between the latent, VAE and CLIP grids.
        use_qwen_alignment = (
            context.scene.model_architecture.startswith('qwen')
            and getattr(context.scene, 'qwen_rescale_alignment', False)
        )
        align_step = 112 if use_qwen_alignment else 8

        target_px = int(getattr(context.scene, 'auto_rescale_target_mp', 1.0) * 1_000_000)
        upper_bound = int(target_px * 1.2)
        lower_bound = int(target_px * 0.8)

        if context.scene.auto_rescale and ((total_pixels > upper_bound or total_pixels < lower_bound) or (resolution_x % align_step != 0 or resolution_y % align_step != 0)):
            scale_factor = (target_px / total_pixels) ** 0.5
            render.resolution_x = int(resolution_x * scale_factor)
            render.resolution_y = int(resolution_y * scale_factor)
            # Round down to nearest multiple of align_step
            render.resolution_x -= render.resolution_x % align_step
            render.resolution_y -= render.resolution_y % align_step
            self.report({'INFO'}, f"Resolution automatically rescaled to {render.resolution_x}x{render.resolution_y}.")

        elif total_pixels > upper_bound:
            target_mp = target_px / 1_000_000
            self.report({'WARNING'}, f"High resolution detected. Resolutions above {target_mp:.1f} MP may reduce performance and quality.")
        
        self._cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
        if not self._cameras:
            self.report({'ERROR'}, "No cameras found in the scene.")
            context.scene.generation_status = 'idle'
            ComfyUIGenerate._is_running = False
            return {'CANCELLED'}
        # Sort cameras by name
        self._cameras.sort(key=lambda x: x.name)

        # Apply custom generation order if enabled (non-destructive reorder)
        if context.scene.sg_use_custom_camera_order and len(context.scene.sg_camera_order) > 0:
            order_names = [item.name for item in context.scene.sg_camera_order]
            cam_by_name = {cam.name: cam for cam in self._cameras}
            ordered = []
            for name in order_names:
                if name in cam_by_name:
                    ordered.append(cam_by_name.pop(name))
            # Append any cameras not in the order list (newly added cameras)
            for cam in self._cameras:
                if cam.name in cam_by_name:
                    ordered.append(cam)
            self._cameras = ordered

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
                        sf = (target_px / c_total) ** 0.5
                        crx = int(crx * sf)
                        cry = int(cry * sf)
                        crx -= crx % align_step
                        cry -= cry % align_step
                        cam["sg_res_x"] = crx
                        cam["sg_res_y"] = cry
        self._selected_camera_ids = [i for i, cam in enumerate(self._cameras) if cam in bpy.context.selected_objects] #TEST
        if len(self._selected_camera_ids) == 0:
            self._selected_camera_ids = list(range(len(self._cameras))) # All cameras selected if none are selected
        
        # Check if there is at least one ControlNet unit
        controlnet_units = getattr(context.scene, "controlnet_units", [])
        if not controlnet_units and not (context.scene.use_flux_lora and context.scene.model_architecture == 'flux1'):
            self.report({'ERROR'}, "At least one ControlNet unit is required to run the operator.")
            context.scene.generation_status = 'idle'
            ComfyUIGenerate._is_running = False
            return {'CANCELLED'}
        
        # If there are curves within the scene, warn the user
        if any(obj.type == 'CURVE' for obj in bpy.context.view_layer.objects):
            self.report({'WARNING'}, "Curves detected in the scene. This may cause issues with the generation process. Consider removing them before proceeding.")
        
        if context.scene.generation_mode == 'project_only':
            print(f"Reprojecting images for {len(self._cameras)} cameras")
        elif context.scene.generation_mode == 'standard':
            print(f"Generating images for {len(self._cameras)} cameras")
        else:
            print(f"Regenerating images for {len(self._selected_camera_ids)} selected cameras")

        if context.scene.texture_objects == 'selected':
            self._to_texture = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            # If empty, cancel the operation
            if not self._to_texture:
                self.report({'ERROR'}, "No mesh objects selected for texturing.")
                context.scene.generation_status = 'idle'
                ComfyUIGenerate._is_running = False
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
                self.report({'ERROR'}, "Not enough UV map slots. Please remove at least 1 UV map to free a slot for the projection buffer.")
                context.scene.generation_status = 'idle'
                ComfyUIGenerate._is_running = False
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
                    self.report({'ERROR'}, f"Cannot use UV inpainting with the material of object '{obj.name}': incompatible material. The generated material has to be active.")
                    context.scene.generation_status = 'idle'
                    ComfyUIGenerate._is_running = False
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
        print("Starting thread") 
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
                # Restore original visibility for non-selected objects
                if context.scene.texture_objects == 'selected':
                    for obj in bpy.context.view_layer.objects:
                        if obj.type == 'MESH' and obj.name in self._original_visibility:
                            obj.hide_render = self._original_visibility[obj.name]
                if self._error:
                    if self._error == "'25'" or self._error == "'111'" or self._error == "'5'":
                        # Probably canceled by user, quietly return
                        context.scene.generation_status = 'idle'
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
                    return {'CANCELLED'}
                if not context.scene.generation_mode == 'project_only':
                    self.report({'INFO'}, "Generation complete.")
                
                # Reset discard factor if enabled
                if (context.scene.discard_factor_generation_only and
                        (self._generation_method_on_start == 'sequential' or context.scene.model_architecture == 'qwen_image_edit')):
                    
                    new_discard_angle = context.scene.discard_factor_after_generation
                    print(f"Resetting discard angle in material nodes to {new_discard_angle}...")

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
                    
                    print("Discard angle reset complete.")

                # Reset weight exponent if enabled
                if (context.scene.weight_exponent_generation_only and
                        (self._generation_method_on_start == 'sequential' or context.scene.model_architecture == 'qwen_image_edit')):
                    
                    new_exponent = context.scene.weight_exponent_after_generation
                    print(f"Resetting weight exponent in material nodes to {new_exponent}...")

                    for obj in self._to_texture:
                        if not obj.active_material or not obj.active_material.use_nodes:
                            continue
                        
                        nodes = obj.active_material.node_tree.nodes
                        for node in nodes:
                            # OSL script nodes: update 'Power' input
                            if node.type == 'SCRIPT':
                                if 'Power' in node.inputs:
                                    node.inputs['Power'].default_value = new_exponent
                            # Native MATH POWER nodes (Blender 5.1+ native path)
                            elif node.type == 'MATH' and node.operation == 'POWER' and node.label == 'power_weight':
                                node.inputs[1].default_value = new_exponent
                    
                    print("Weight exponent reset complete.")

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
                # Clear output directories which are not needed anymore
                addon_prefs = context.preferences.addons[__package__].preferences
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
        server_address = context.preferences.addons[__package__].preferences.server_address
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
                or (context.scene.model_architecture == 'qwen_image_edit'
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
                or (context.scene.model_architecture == 'qwen_image_edit'
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
            if (context.scene.model_architecture == 'qwen_image_edit'
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
            elif (context.scene.model_architecture == 'qwen_image_edit'
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
                            project_image(context, self._to_texture, self._material_id, stop_index=self._current_image)
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
                                        export_visibility(context, self._to_texture, camera_visibility=_vis_cam) # Export mask for current view
                                    _emit_cam = _cur_cam
                                    with _SGCameraResolution(context, _emit_cam):
                                        if context.scene.model_architecture == 'qwen_image_edit': # export custom bg and fallback for Qwen image edit
                                            fallback_color, background_color = self._get_qwen_context_colors(context)
                                            export_emit_image(context, self._to_texture, camera_id=self._current_image, bg_color=background_color, fallback_color=fallback_color) # Export render for next view
                                            self._dilate_qwen_context_fallback(context, self._current_image, fallback_color)
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
                            else:
                                image = self.workflow_manager.refine(context, controlnet_info=controlnet_info, render_info=render_info, mask_info=mask_info, ipadapter_ref_info=ipadapter_ref_info)
                    else: # Grid or Separate
                        if context.scene.model_architecture == 'flux1':
                            image = self.workflow_manager.generate_flux(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                        elif context.scene.model_architecture == 'qwen_image_edit':
                            image = self.workflow_manager.generate_qwen_edit(context, camera_id=camera_id)
                        else:
                            image = self.workflow_manager.generate(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)

                    if image == {"error": "conn_failed"}:
                        return # Error message already set

                    if (context.scene.model_architecture == 'qwen_image_edit' and
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
                    
                     # Sequential mode callback
                    if context.scene.generation_method == 'sequential':
                        self._stage = "Projecting Image"
                        def image_project_callback():
                            try:
                                redraw_ui(context)
                                project_image(context, self._to_texture, self._material_id, stop_index=self._current_image)
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

        def image_project_callback():
            if context.scene.generation_method == 'sequential':
                return None
            self._stage = "Projecting Image"
            redraw_ui(context)
            if context.scene.generation_method != 'uv_inpaint':
                project_image(context, self._to_texture, self._material_id)
            else:
                # Apply the UV inpainted textures to each mesh
                from .render_tools import apply_uv_inpaint_texture
                for obj in self._to_texture:
                    texture_path = get_file_path(
                        context, "generated_baked", object_name=obj.name, material_id=self._material_id
                    )
                    apply_uv_inpaint_texture(context, obj, texture_path)
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
            print(f"Failed to load context render for dilation at {image_path}: {err}")
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
            print(f"Failed to save dilated context render at {image_path}: {err}")
            return

        if hasattr(self, '_uploaded_images_cache') and self._uploaded_images_cache is not None:
            self._uploaded_images_cache.pop(os.path.abspath(image_path), None)

    def _apply_qwen_context_cleanup(self, context, image_bytes):
        hue_tolerance = max(context.scene.qwen_context_cleanup_hue_tolerance, 0.0)
        value_adjust = context.scene.qwen_context_cleanup_value_adjust
        fallback_color = tuple(context.scene.qwen_guidance_fallback_color)
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                rgba_image = pil_image.convert("RGBA")
                pixel_data = np.array(rgba_image)
        except Exception as err:
            print(f"  Warning: Failed to read Qwen context render for cleanup: {err}")
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
            print(f"  Warning: Failed to write cleaned Qwen context render: {err}")
            traceback.print_exc()
            return image_bytes

    def export_depthmap(self, context, camera_id=None):
        """     
        Exports the depth map of the scene.         
        :param context: Blender context.         
        :param camera_id: ID of the camera.         
        :return: None     
        """
        print("Exporting depth map")
        # Save original settings to restore later.
        original_engine = bpy.context.scene.render.engine
        original_view_transform = bpy.context.scene.view_settings.view_transform
        original_film_transparent = bpy.context.scene.render.film_transparent

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

        # Use the compositor to save the depth pass
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

        print(f"Depth map saved to: {os.path.join(output_dir, output_file)}.png")
        
        # Restore original settings
        bpy.context.scene.render.engine = original_engine
        bpy.context.scene.view_settings.view_transform = original_view_transform
        bpy.context.scene.render.film_transparent = original_film_transparent
        view_layer.use_pass_z = original_pass_z

    def export_normal(self, context, camera_id=None):
        """
        Exports the normal map of the scene.
        Areas without geometry will show the neutral color (0.5, 0.5, 1.0).
        :param context: Blender context.
        :param camera_id: ID of the camera.
        :return: None
        """
        print("Exporting normal map")
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

        bpy.context.scene.render.engine = get_eevee_engine_id()
        bpy.context.scene.view_settings.view_transform = 'Raw'
        bpy.context.scene.render.film_transparent = True
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

        view_layer.use_pass_normal = original_pass_normal

        print(f"Normal map saved to: {os.path.join(output_dir, output_file)}.png")

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
        print(f"Combined depth map grid saved to: {grid_image_path}")

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
            print(f"Generated image for camera {i+1} saved to: {individual_image_path}")
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
            (context.scene.sequential_ipadapter or context.scene.model_architecture == 'qwen_image_edit')
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
                    print(f"Error modifying image {local_path}: {e}. Uploading original.")
                    upload_path = local_path # Fallback to original on error
        # --- End Image Modification ---

        # Use the operator's instance cache variable (self._uploaded_images_cache)
        if not hasattr(self, '_uploaded_images_cache') or self._uploaded_images_cache is None:
            # Initialize cache if it doesn't exist (e.g., first call in execute)
            # Although clearing in execute() is preferred
            self._uploaded_images_cache = {}
            print("Warning: _uploaded_images_cache not found, initializing. Should be cleared in execute().")


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
        server_address = context.preferences.addons[__package__].preferences.server_address
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

def _draw_rect_2d(x1, y1, x2, y2, color):
    """Draw a filled rectangle in 2D screen-space."""
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": ((x1, y1), (x2, y1), (x2, y2), (x1, y2))},
        indices=((0, 1, 2), (0, 2, 3)),
    )
    shader.bind()
    shader.uniform_float("color", color)
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _draw_texture_2d(texture, x1, y1, x2, y2):
    """Draw a textured rectangle in 2D screen-space."""
    shader = gpu.shader.from_builtin('IMAGE')
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
         "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))},
        indices=((0, 1, 2), (0, 2, 3)),
    )
    shader.bind()
    shader.uniform_sampler("image", texture)
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')


class _PreviewGalleryOverlay:
    """Viewport overlay that shows N generated images and lets the user pick one.

    Lifecycle:
    1. ``__init__`` — creates bpy.data.images + GPU textures, registers draw handler.
    2. ``handle_mouse_move`` / ``handle_click`` — called from Trellis2Generate.modal().
    3. ``update_images`` — called when "Generate More" produces a new batch.
    4. ``cleanup`` — removes draw handler and temp images.
    """

    def __init__(self, pil_images, seeds):
        """*pil_images*: list[PIL.Image]  *seeds*: list[int]"""
        self._pil_images = list(pil_images)
        self._seeds = list(seeds)
        self._n = len(pil_images)
        self._hover_idx = -1
        self._more_hover = False
        self._cancel_hover = False
        self._selected_idx = -1
        self.action = None  # 'select' | 'more' | 'cancel'
        self._cols = max(1, math.ceil(math.sqrt(self._n)))
        self._rows = max(1, math.ceil(self._n / self._cols))
        self._cell_rects: list[tuple] = []
        self._more_rect: tuple | None = None
        self._cancel_rect: tuple | None = None
        self._bpy_images: list = []
        self._textures: list = []
        self._setup_textures()
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (), 'WINDOW', 'POST_PIXEL')

    # ── texture management ──

    def _setup_textures(self):
        self._clear_textures()
        for i, pil_img in enumerate(self._pil_images):
            name = f"_sg_gallery_{i}"
            if name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])
            rgba = pil_img.convert('RGBA')
            w, h = rgba.size
            bpy_img = bpy.data.images.new(name, w, h, alpha=True)
            # Mark as Non-Color so gpu.texture.from_image() stores raw
            # sRGB values — the viewport's own output transform will
            # apply the single correct sRGB curve on display.
            bpy_img.colorspace_settings.name = 'Non-Color'
            # Blender images are stored bottom-to-top; PIL is top-to-bottom
            flipped = rgba.transpose(Image.FLIP_TOP_BOTTOM)
            flat = np.array(flipped, dtype=np.float32).ravel() / 255.0
            bpy_img.pixels.foreach_set(flat)
            bpy_img.pack()
            self._bpy_images.append(bpy_img)
            self._textures.append(gpu.texture.from_image(bpy_img))

    def _clear_textures(self):
        self._textures.clear()
        for img in self._bpy_images:
            if img.name in bpy.data.images:
                bpy.data.images.remove(img)
        self._bpy_images.clear()

    def update_images(self, pil_images, seeds):
        """Replace the gallery with a new batch."""
        self._pil_images = list(pil_images)
        self._seeds = list(seeds)
        self._n = len(pil_images)
        self._cols = max(1, math.ceil(math.sqrt(self._n)))
        self._rows = max(1, math.ceil(self._n / self._cols))
        self._hover_idx = -1
        self._more_hover = False
        self._cancel_hover = False
        self._selected_idx = -1
        self.action = None
        self._setup_textures()

    def cleanup(self):
        if self._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            self._draw_handle = None
        self._clear_textures()

    # ── interaction ──

    def handle_mouse_move(self, mx, my):
        """Update hover state. Returns True if display should redraw."""
        old = (self._hover_idx, self._more_hover, self._cancel_hover)
        self._hover_idx = -1
        self._more_hover = False
        self._cancel_hover = False
        for i, rect in enumerate(self._cell_rects):
            x1, y1, x2, y2 = rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                self._hover_idx = i
                break
        if self._more_rect:
            x1, y1, x2, y2 = self._more_rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                self._more_hover = True
        if self._cancel_rect:
            x1, y1, x2, y2 = self._cancel_rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                self._cancel_hover = True
        return old != (self._hover_idx, self._more_hover, self._cancel_hover)

    def handle_click(self, mx, my):
        """Returns 'select', 'more', 'cancel', or None."""
        if self._more_hover:
            return 'more'
        if self._cancel_hover:
            return 'cancel'
        if self._hover_idx >= 0:
            self._selected_idx = self._hover_idx
            return 'select'
        return None

    @property
    def selected_seed(self):
        if 0 <= self._selected_idx < len(self._seeds):
            return self._seeds[self._selected_idx]
        return None

    @property
    def selected_image_bytes(self):
        """Return PNG bytes of the selected image (for TRELLIS workflow)."""
        if 0 <= self._selected_idx < len(self._pil_images):
            buf = io.BytesIO()
            self._pil_images[self._selected_idx].save(buf, format='PNG')
            return buf.getvalue()
        return None

    # ── drawing ──

    def _draw(self):
        """POST_PIXEL callback — draws the full-screen gallery overlay."""
        context = bpy.context
        region = context.region
        if not region:
            return
        vw, vh = region.width, region.height
        if vw < 100 or vh < 100:
            return

        # Detect sidebar (N-panel) width so the grid avoids being covered.
        sidebar_w = 0
        area = context.area
        if area:
            for r in area.regions:
                if r.type == 'UI' and r.width > 1:
                    sidebar_w = r.width
                    break

        # Layout constants
        pad = 20
        btn_h = 40
        title_h = 30
        bottom_area = btn_h + pad * 2
        usable_w = vw - sidebar_w  # exclude area behind the N-panel

        # --- dark backdrop ---
        _draw_rect_2d(0, 0, vw, vh, (0.08, 0.08, 0.08, 0.90))

        # --- title ---
        blf.size(0, 20)
        title = "Select an image — or generate more"
        tw, th = blf.dimensions(0, title)
        blf.position(0, (usable_w - tw) / 2, vh - pad - th, 0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, title)

        grid_top = vh - pad - title_h - pad
        available_w = usable_w - pad * 2
        available_h = grid_top - bottom_area - pad

        cell_w = max(1, available_w // self._cols)
        cell_h = max(1, available_h // self._rows)

        # Maintain image aspect ratio
        if self._pil_images:
            img_aspect = self._pil_images[0].width / max(1, self._pil_images[0].height)
            desired_h = cell_w / img_aspect
            if desired_h > cell_h:
                cell_w = int(cell_h * img_aspect)
            else:
                cell_h = int(desired_h)

        inner = 6
        total_grid_w = self._cols * cell_w
        offset_x = (usable_w - total_grid_w) / 2.0

        self._cell_rects.clear()

        for i in range(self._n):
            row = i // self._cols
            col = i % self._cols
            x1 = offset_x + col * cell_w + inner
            y2 = grid_top - row * cell_h - inner
            x2 = x1 + cell_w - inner * 2
            y1 = y2 - cell_h + inner * 2
            self._cell_rects.append((x1, y1, x2, y2))

            # Hover highlight ring
            if i == self._hover_idx:
                _draw_rect_2d(x1 - 3, y1 - 3, x2 + 3, y2 + 3, (0.35, 0.65, 1.0, 0.85))

            # Image texture
            if i < len(self._textures):
                _draw_texture_2d(self._textures[i], x1, y1, x2, y2)

            # Number badge
            blf.size(0, 15)
            label = str(i + 1)
            lw, lh = blf.dimensions(0, label)
            _draw_rect_2d(x1, y2 - lh - 8, x1 + lw + 12, y2, (0.0, 0.0, 0.0, 0.70))
            blf.position(0, x1 + 6, y2 - lh - 3, 0)
            blf.color(0, 1.0, 1.0, 1.0, 1.0)
            blf.draw(0, label)

            # Seed label
            blf.size(0, 11)
            seed_txt = f"Seed: {self._seeds[i]}"
            sw, sh = blf.dimensions(0, seed_txt)
            _draw_rect_2d(x1, y1, x1 + sw + 10, y1 + sh + 6, (0, 0, 0, 0.6))
            blf.position(0, x1 + 5, y1 + 3, 0)
            blf.color(0, 0.7, 0.7, 0.7, 1.0)
            blf.draw(0, seed_txt)

        # --- buttons ---
        btn_w = 160
        btn_y1 = pad
        btn_y2 = pad + btn_h

        # "Generate More"
        center_x = usable_w / 2.0
        mx1 = center_x - btn_w - 10
        mx2 = center_x - 10
        c = (0.25, 0.55, 0.85, 0.95) if self._more_hover else (0.20, 0.35, 0.55, 0.85)
        _draw_rect_2d(mx1, btn_y1, mx2, btn_y2, c)
        self._more_rect = (mx1, btn_y1, mx2, btn_y2)
        blf.size(0, 15)
        mt = "Generate More"
        mtw, mth = blf.dimensions(0, mt)
        blf.position(0, (mx1 + mx2 - mtw) / 2, (btn_y1 + btn_y2 - mth) / 2, 0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, mt)

        # "Cancel"
        cx1 = center_x + 10
        cx2 = center_x + btn_w + 10
        c = (0.75, 0.25, 0.25, 0.95) if self._cancel_hover else (0.45, 0.20, 0.20, 0.85)
        _draw_rect_2d(cx1, btn_y1, cx2, btn_y2, c)
        self._cancel_rect = (cx1, btn_y1, cx2, btn_y2)
        blf.size(0, 15)
        ct = "Cancel"
        ctw, cth = blf.dimensions(0, ct)
        blf.position(0, (cx1 + cx2 - ctw) / 2, (btn_y1 + btn_y2 - cth) / 2, 0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, ct)


class Trellis2Generate(bpy.types.Operator):
    """Generate a 3D mesh from a reference image using TRELLIS.2 via ComfyUI.

    Requires the PozzettiAndrea/ComfyUI-TRELLIS2 custom node pack installed on the ComfyUI server.
    Uploads the input image, runs the full TRELLIS.2 pipeline (background removal, conditioning,
    shape generation, texture generation, GLB export), downloads the resulting GLB file, and
    imports it into the Blender scene."""
    bl_idname = "object.trellis2_generate"
    bl_label = "Generate 3D Mesh (TRELLIS.2)"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _error = None
    _glb_data = None
    _is_running = False
    _cancelled = False
    _active_ws = None  # WebSocket reference for cancel-time close
    _progress = 0.0
    _stage = "Initializing"
    workflow_manager: object = None

    # ── Preview gallery state ─────────────────────────────────────────
    _gallery_overlay: _PreviewGalleryOverlay | None = None
    _gallery_event: threading.Event | None = None
    _gallery_ready: bool = False
    _gallery_action: str | None = None  # 'select' | 'more' | 'cancel'
    _gallery_selected_bytes: bytes | None = None
    _gallery_selected_seed: int | None = None
    _progress_remap: tuple | None = None  # (base, span) for gallery sub-range scaling

    # ── 3-tier progress ──────────────────────────────────────────────
    _overall_progress: float = 0.0
    _overall_stage: str = "Initializing"
    _phase_progress: float = 0.0
    _phase_stage: str = ""
    _detail_progress: float = 0.0
    _detail_stage: str = ""
    _current_phase: int = 0
    _total_phases: int = 3  # 2 when gen_from == 'image'

    def _update_overall(self):
        """Recompute *_overall_progress* from current phase + phase progress."""
        layout = getattr(self, '_phase_layout', '')
        if layout == 'txt2img+trellis+texturing':  # 3 phases
            starts  = {1: 0,  2: 15, 3: 65}
            weights = {1: 15, 2: 50, 3: 35}
        elif layout == 'trellis+texturing':  # 2 phases: big mesh, then texturing
            starts  = {1: 0,  2: 65}
            weights = {1: 65, 2: 35}
        elif layout == 'txt2img+trellis':  # 2 phases: quick txt2img, then big mesh+native tex
            starts  = {1: 0,  2: 15}
            weights = {1: 15, 2: 85}
        else:  # Single phase — scale to full 0-100
            starts  = {1: 0}
            weights = {1: 100}
        s = starts.get(self._current_phase, 0)
        w = weights.get(self._current_phase, 0)
        self._overall_progress = s + (self._phase_progress / 100.0) * w
        self._overall_progress = max(0.0, min(self._overall_progress, 100.0))
        # Keep legacy _progress in sync for any code that reads it
        self._progress = self._overall_progress

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return True  # Allow cancellation
        addon_prefs = context.preferences.addons[__package__].preferences
        if not addon_prefs.server_address or not addon_prefs.server_online:
            return False
        if not os.path.exists(addon_prefs.output_dir):
            return False
        if bpy.app.online_access == False:
            return False
        # Check that an input image is specified (only required when generate_from = image)
        gen_from = getattr(context.scene, 'trellis2_generate_from', 'image')
        if gen_from == 'image' and not context.scene.trellis2_input_image:
            return False
        # Check no other generation is running
        for window in context.window_manager.windows:
            for op in window.modal_operators:
                if op.bl_idname in ('OBJECT_OT_test_stable', 'OBJECT_OT_trellis2_generate',
                                    'OBJECT_OT_bake_textures', 'OBJECT_OT_add_cameras'):
                    return False
        return True

    def execute(self, context):
        if Trellis2Generate._is_running:
            # Cancel — tell the server to stop and close the WebSocket
            # so the background thread unblocks from ws.recv().
            Trellis2Generate._cancelled = True
            Trellis2Generate._is_running = False

            # Send /interrupt to ComfyUI (same as standard texturing cancel)
            try:
                server_address = context.preferences.addons[__package__].preferences.server_address
                data = json.dumps({"client_id": str(uuid.uuid4())}).encode('utf-8')
                req = urllib.request.Request("http://{}/interrupt".format(server_address), data=data)
                urllib.request.urlopen(req)
            except Exception:
                pass  # Best effort — server may already be gone

            # Close the active WebSocket so the thread's ws.recv() raises
            ws = Trellis2Generate._active_ws
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass
                Trellis2Generate._active_ws = None

            # Wake up the gallery event in case the thread is blocked there
            if self._gallery_event:
                self._gallery_action = 'cancel'
                self._gallery_event.set()

            self.report({'WARNING'}, "TRELLIS.2 generation cancelled")
            return {'FINISHED'}

        scene = context.scene
        gen_from = getattr(scene, 'trellis2_generate_from', 'image')
        tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')

        # Validate input image (only required in image mode)
        image_path = None
        if gen_from == 'image':
            image_path = bpy.path.abspath(scene.trellis2_input_image)
            if not os.path.exists(image_path):
                self.report({'ERROR'}, f"Input image not found: {image_path}")
                return {'CANCELLED'}

        Trellis2Generate._is_running = True
        self._error = None
        self._glb_data = None
        self._progress = 0.0
        self._stage = "Initializing"
        self._texture_mode = tex_mode
        self.workflow_manager = WorkflowManager(self)

        # Gallery state reset
        self._gallery_overlay = None
        self._gallery_event = threading.Event()
        self._gallery_ready = False
        self._gallery_action = None
        self._gallery_selected_bytes = None
        self._gallery_selected_seed = None

        # 3-tier progress init
        has_txt2img = (gen_from == 'prompt')
        has_texturing = (tex_mode in ('sdxl', 'flux1', 'qwen_image_edit'))
        if has_txt2img and has_texturing:
            self._total_phases = 3
            self._phase_layout = 'txt2img+trellis+texturing'
        elif has_txt2img:
            self._total_phases = 2
            self._phase_layout = 'txt2img+trellis'
        elif has_texturing:
            self._total_phases = 2
            self._phase_layout = 'trellis+texturing'
        else:
            self._total_phases = 1
            self._phase_layout = 'trellis_only'
        self._current_phase = 0
        self._overall_progress = 0.0
        self._overall_stage = "Initializing"
        self._phase_progress = 0.0
        self._phase_stage = ""
        self._detail_progress = 0.0
        self._detail_stage = ""

        # Compute revision directory on the main thread (may write output_timestamp)
        from .utils import get_generation_dirs
        gen_dirs = get_generation_dirs(context)
        revision_dir = gen_dirs.get("revision", "")

        # Start generation in background thread
        self._thread = threading.Thread(
            target=self._run_trellis2,
            args=(context, image_path, gen_from, revision_dir),
            daemon=True
        )
        self._thread.start()

        # Register modal timer
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)

        return {'RUNNING_MODAL'}

    def _cleanup_gallery(self):
        """Remove the gallery overlay and free GPU resources."""
        if self._gallery_overlay:
            self._gallery_overlay.cleanup()
            self._gallery_overlay = None

    def modal(self, context, event):
        # ── Gallery mode: intercept mouse + keyboard ──────────────
        if self._gallery_overlay is not None:
            if event.type == 'MOUSEMOVE':
                if self._gallery_overlay.handle_mouse_move(
                        event.mouse_region_x, event.mouse_region_y):
                    for area in context.screen.areas:
                        area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                action = self._gallery_overlay.handle_click(
                    event.mouse_region_x, event.mouse_region_y)
                if action == 'select':
                    self._gallery_selected_bytes = self._gallery_overlay.selected_image_bytes
                    self._gallery_selected_seed = self._gallery_overlay.selected_seed
                    self._gallery_action = 'select'
                    self._cleanup_gallery()
                    self._gallery_ready = False
                    self._gallery_event.set()
                    return {'RUNNING_MODAL'}
                elif action == 'more':
                    self._gallery_action = 'more'
                    self._cleanup_gallery()
                    self._gallery_ready = False
                    self._gallery_event.set()
                    return {'RUNNING_MODAL'}
                elif action == 'cancel':
                    self._gallery_action = 'cancel'
                    self._cleanup_gallery()
                    self._gallery_ready = False
                    self._gallery_event.set()
                    return {'RUNNING_MODAL'}
                return {'RUNNING_MODAL'}

            if event.type == 'ESC' and event.value == 'PRESS':
                self._gallery_action = 'cancel'
                self._cleanup_gallery()
                self._gallery_ready = False
                self._gallery_event.set()
                return {'RUNNING_MODAL'}

            if event.type == 'TIMER':
                for area in context.screen.areas:
                    area.tag_redraw()
            return {'RUNNING_MODAL'}

        # ── Normal mode ───────────────────────────────────────────
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # Redraw UI for progress updates
        for area in context.screen.areas:
            area.tag_redraw()

        # Check if gallery is ready (thread waiting for user input)
        if self._gallery_ready and self._gallery_overlay is None:
            gallery_data = getattr(self, '_gallery_data', None)
            if gallery_data:
                pil_imgs, seeds = gallery_data
                self._gallery_overlay = _PreviewGalleryOverlay(pil_imgs, seeds)
                for area in context.screen.areas:
                    area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Check if thread is still running
        if self._thread and self._thread.is_alive():
            return {'RUNNING_MODAL'}

        # Thread finished - clean up timer
        context.window_manager.event_timer_remove(self._timer)
        self._timer = None
        was_cancelled = Trellis2Generate._cancelled
        Trellis2Generate._is_running = False
        Trellis2Generate._cancelled = False
        Trellis2Generate._active_ws = None
        self._cleanup_gallery()

        # User cancelled — exit silently (no error toast)
        if was_cancelled:
            context.scene.generation_status = 'idle'
            return {'FINISHED'}

        if self._error:
            self.report({'ERROR'}, f"TRELLIS.2 error: {self._error}")
            return {'CANCELLED'}

        if self._glb_data is None or (isinstance(self._glb_data, dict) and "error" in self._glb_data):
            error_msg = self._glb_data.get("error", "Unknown error") if isinstance(self._glb_data, dict) else "No data received"
            self.report({'ERROR'}, f"TRELLIS.2 failed: {error_msg}")
            return {'CANCELLED'}

        # Surface mesh-corruption warning to the user (set by workflows.py
        # when the GLB validator detects artifacts but recovery failed).
        _mesh_warning = getattr(self, '_warning', None)
        if _mesh_warning:
            self.report({'WARNING'}, _mesh_warning)
            self._warning = None  # consumed

        # Save GLB to revision directory and import into Blender
        try:
            from .utils import get_generation_dirs
            gen_dirs = get_generation_dirs(context)
            save_dir = gen_dirs.get("revision", "")
            if not save_dir:
                save_dir = context.preferences.addons[__package__].preferences.output_dir
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            glb_filename = f"trellis2_{timestamp}.glb"
            glb_path = os.path.join(save_dir, glb_filename)

            with open(glb_path, 'wb') as f:
                f.write(self._glb_data)

            # Store the TRELLIS.2 input image path for downstream use (IPAdapter/Qwen style)
            input_img = getattr(self, '_input_image_path', None)
            if input_img:
                context.scene.trellis2_last_input_image = input_img

            print(f"[TRELLIS2] Saved GLB to: {glb_path} ({len(self._glb_data)} bytes)")

            # Import GLB into Blender
            bpy.ops.import_scene.gltf(filepath=glb_path)

            # --- Normalise imported mesh to a reasonable Blender-unit size ---
            target_bu = getattr(context.scene, 'trellis2_import_scale', 2.0)
            if target_bu > 0:
                imported_objects = [obj for obj in context.selected_objects]
                if imported_objects:
                    # Compute combined world-space bounding box across all
                    # imported objects (meshes, empties, armatures …).
                    all_corners = []
                    for obj in imported_objects:
                        for corner in obj.bound_box:
                            all_corners.append(obj.matrix_world @ mathutils.Vector(corner))
                    if all_corners:
                        xs = [c.x for c in all_corners]
                        ys = [c.y for c in all_corners]
                        zs = [c.z for c in all_corners]
                        extent = max(
                            max(xs) - min(xs),
                            max(ys) - min(ys),
                            max(zs) - min(zs),
                        )
                        if extent > 1e-6:
                            scale_factor = target_bu / extent
                            # Find the root objects (those without an imported parent)
                            roots = [o for o in imported_objects if o.parent not in imported_objects]
                            for root in roots:
                                root.scale *= scale_factor
                            # Apply scale so downstream code sees unit scale
                            bpy.ops.object.select_all(action='DESELECT')
                            for obj in imported_objects:
                                obj.select_set(True)
                            bpy.context.view_layer.objects.active = imported_objects[0]
                            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                            print(f"[TRELLIS2] Scaled mesh to {target_bu} BU (factor {scale_factor:.4f})")

            # --- Optional studio lighting for native PBR textures ---
            tex_mode = getattr(self, '_texture_mode', 'native')
            if tex_mode == 'native' and getattr(context.scene, 'trellis2_auto_lighting', False):
                self._setup_studio_lighting(context, target_bu)

            # --- Phase 3: If diffusion texturing, auto-place cameras + start generation ---
            if tex_mode in ('sdxl', 'flux1', 'qwen_image_edit'):
                # Place cameras NOW (while operator context is still valid)
                camera_count = getattr(context.scene, 'trellis2_camera_count', 8)
                imported_objects = [obj for obj in context.selected_objects]

                if imported_objects:
                    bpy.context.view_layer.objects.active = imported_objects[0]
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in imported_objects:
                        obj.select_set(True)

                # Force viewport to standard front view so AddCameras uses a
                # consistent reference direction for sorting and auto-prompts.
                # TRELLIS.2 always imports meshes in standard orientation so the
                # viewport should match.
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                rv3d = space.region_3d
                                if rv3d:
                                    # Blender front view (Numpad 1): -Y looking at +Y
                                    rv3d.view_rotation = mathutils.Quaternion(
                                        (0.7071068, 0.7071068, 0.0, 0.0)
                                    )
                                    rv3d.view_perspective = 'PERSP'
                        break

                try:
                    _pm = getattr(context.scene, 'trellis2_placement_mode', 'normal_weighted')
                    _cam_kwargs = {
                        'placement_mode': _pm,
                        'num_cameras': camera_count,
                        'auto_prompts': getattr(context.scene, 'trellis2_auto_prompts', True),
                        'review_placement': False,
                        'purge_others': True,
                        'exclude_bottom': getattr(context.scene, 'trellis2_exclude_bottom', True),
                        'exclude_bottom_angle': getattr(context.scene, 'trellis2_exclude_bottom_angle', 1.5533),
                        'auto_aspect': getattr(context.scene, 'trellis2_auto_aspect', 'per_camera'),
                        'occlusion_mode': getattr(context.scene, 'trellis2_occlusion_mode', 'none'),
                        'consider_existing': getattr(context.scene, 'trellis2_consider_existing', True),
                        'clamp_elevation': getattr(context.scene, 'trellis2_clamp_elevation', False),
                        'max_elevation_angle': getattr(context.scene, 'trellis2_max_elevation', 1.2217),
                        'min_elevation_angle': getattr(context.scene, 'trellis2_min_elevation', -0.1745),
                    }
                    if _pm == 'greedy_coverage':
                        _cam_kwargs['coverage_target'] = getattr(context.scene, 'trellis2_coverage_target', 0.95)
                        _cam_kwargs['max_auto_cameras'] = getattr(context.scene, 'trellis2_max_auto_cameras', 12)
                    if _pm == 'fan_from_camera':
                        _cam_kwargs['fan_angle'] = getattr(context.scene, 'trellis2_fan_angle', 90.0)
                    bpy.ops.object.add_cameras(**_cam_kwargs)
                    print(f"[TRELLIS2] Placed {camera_count} cameras ({_pm})")
                except Exception as cam_err:
                    print(f"[TRELLIS2] Warning: Camera placement failed: {cam_err}")
                    traceback.print_exc()

                # Defer texture generation so Blender digests the new cameras
                self._schedule_texture_generation(context)
                self.report({'INFO'}, f"TRELLIS.2: Mesh imported. Camera placement done, texture generation starting...")
            else:
                self.report({'INFO'}, f"TRELLIS.2: Imported 3D mesh from {glb_filename}")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to import GLB: {e}")
            traceback.print_exc()
            return {'CANCELLED'}

    # -----------------------------------------------------------------
    # Studio lighting (three-point rig for PBR showcase)
    # -----------------------------------------------------------------
    def _setup_studio_lighting(self, context, import_scale):
        """Create a three-point studio lighting setup around the imported mesh."""
        S = max(import_scale, 0.5)   # Avoid degenerate positions
        dist = S * 2.5               # Distance from origin

        # (name, watts, size, color_rgb, azimuth_deg, elevation_deg)
        light_defs = [
            ("SG_Key",  200, 1.5 * S, (1.0, 0.96, 0.90),   45, 40),
            ("SG_Fill",  80, 2.5 * S, (0.90, 0.94, 1.0),   -60, 15),
            ("SG_Rim",  120, 0.8 * S, (1.0, 1.0, 1.0),     170, 55),
        ]

        collection = context.collection
        created = []

        for name, power, size, color, az_deg, el_deg in light_defs:
            # Remove any stale light with the same name
            old = bpy.data.objects.get(name)
            if old:
                bpy.data.objects.remove(old, do_unlink=True)

            az = math.radians(az_deg)
            el = math.radians(el_deg)
            x = dist * math.cos(el) * math.sin(az)
            y = -dist * math.cos(el) * math.cos(az)   # -Y = Blender front
            z = dist * math.sin(el)

            light_data = bpy.data.lights.new(name=name, type='AREA')
            light_data.energy = power
            light_data.size = size
            light_data.color = color

            light_obj = bpy.data.objects.new(name=name, object_data=light_data)
            collection.objects.link(light_obj)

            light_obj.location = (x, y, z)
            # Aim at origin
            direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector((x, y, z))
            rot = direction.to_track_quat('-Z', 'Y')
            light_obj.rotation_euler = rot.to_euler()

            created.append(light_obj)

        print(f"[TRELLIS2] Studio lighting created: {[o.name for o in created]}")
        return created

    def _schedule_texture_generation(self, context):
        """Defer texture generation via a timer so Blender can digest the new cameras.

        Camera placement has already happened in ``modal()``.  This only
        selects all cameras and starts ``object.test_stable``.
        Sets scene-level pipeline flags so the UI can show the overall
        progress bar on top of the ComfyUIGenerate bars.
        """
        # Compute the overall-% at which texturing begins
        if self._total_phases == 3:
            phase_start = 65.0
        elif self._total_phases == 2:
            phase_start = 65.0
        else:
            phase_start = 0.0

        scene = context.scene
        scene.trellis2_pipeline_active = True
        scene.trellis2_pipeline_phase_start_pct = phase_start
        scene.trellis2_pipeline_total_phases = self._total_phases

        def _deferred_generate():
            try:
                # Defensive VRAM flush before loading the diffusion checkpoint.
                # The TRELLIS post-generation flush should have freed VRAM,
                # but if it silently failed the models are still resident (Gap C).
                # Also clear history to release cached node outputs.
                try:
                    srv = bpy.context.preferences.addons[__package__].preferences.server_address
                    # 1. Set unload flags
                    flush_data = json.dumps({"unload_models": True, "free_memory": True}).encode('utf-8')
                    flush_req = urllib.request.Request(
                        f"http://{srv}/free", data=flush_data,
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(flush_req, timeout=get_timeout('api'))
                    # 2. Clear history/cache
                    hist_data = json.dumps({"clear": True}).encode('utf-8')
                    hist_req = urllib.request.Request(
                        f"http://{srv}/history", data=hist_data,
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(hist_req, timeout=get_timeout('api'))
                    # 3. Wait for VRAM release
                    import time
                    time.sleep(3)
                    print("[TRELLIS2] Pre-texturing VRAM flush sent (unload+history clear)")
                except Exception as flush_err:
                    print(f"[TRELLIS2] Pre-texturing flush warning: {flush_err}")

                bpy.ops.object.select_all(action='DESELECT')
                for obj in bpy.context.scene.objects:
                    if obj.type == 'CAMERA':
                        obj.select_set(True)

                bpy.ops.object.test_stable('INVOKE_DEFAULT')
                print("[TRELLIS2] Texture generation started")
            except Exception as e:
                print(f"[TRELLIS2] Warning: Texture generation failed to start: {e}")
                traceback.print_exc()
            return None  # Run once

        # Remember which cameras exist before texturing so we can delete
        # the ones we placed if the user opted in.
        _pre_tex_cameras = {obj.name for obj in bpy.context.scene.objects if obj.type == 'CAMERA'}

        def _pipeline_watcher():
            """Clear the pipeline flag when texturing finishes or is cancelled."""
            if bpy.context.scene.generation_status in ('idle', 'waiting'):
                bpy.context.scene.trellis2_pipeline_active = False
                print("[TRELLIS2] Pipeline complete — overall bar removed")

                # Delete auto-placed cameras if the user requested it
                if getattr(bpy.context.scene, 'trellis2_delete_cameras', False):
                    to_remove = [obj for obj in bpy.context.scene.objects
                                 if obj.type == 'CAMERA' and obj.name in _pre_tex_cameras]
                    if to_remove:
                        bpy.ops.object.select_all(action='DESELECT')
                        for obj in to_remove:
                            obj.select_set(True)
                        bpy.ops.object.delete()
                        print(f"[TRELLIS2] Deleted {len(to_remove)} auto-placed cameras")

                return None  # Stop timer
            return 1.0  # Check again in 1s

        bpy.app.timers.register(_deferred_generate, first_interval=0.5)
        bpy.app.timers.register(_pipeline_watcher, first_interval=2.0)

    def _run_trellis2(self, context, image_path, gen_from, revision_dir):
        """Background thread: runs the TRELLIS.2 pipeline.

        If *gen_from* is ``'prompt'`` the method first generates an input
        image via a lightweight txt2img ComfyUI workflow, saves it to the
        revision directory and passes that to the TRELLIS.2 mesh workflow.

        When the preview gallery is enabled (``trellis2_preview_gallery_enabled``),
        the prompt path generates N images with different seeds and pauses to
        let the user pick one via the viewport overlay before continuing.
        """
        import random as _rng
        try:
            # --- Phase 1: Image acquisition ---
            if gen_from == 'prompt':
                self._current_phase = 1
                self._phase_stage = "Generating Input Image"
                self._phase_progress = 0
                self._detail_progress = 0
                self._detail_stage = "Flushing stale models"
                self._overall_stage = f"Phase 1/{self._total_phases}: Input Image"
                self._update_overall()

                # Flush any stale models from prior runs before loading a
                # diffusion checkpoint for txt2img (Gap A).
                try:
                    server_addr = context.preferences.addons[__package__].preferences.server_address
                    self.workflow_manager._flush_comfyui_vram(server_addr, label="Pre-txt2img")
                except Exception:
                    pass

                self._detail_stage = "Starting txt2img"

                gallery_enabled = getattr(context.scene, 'trellis2_preview_gallery_enabled', False)
                gallery_count = max(1, int(getattr(context.scene, 'trellis2_preview_gallery_count', 4)))

                if gallery_enabled and gallery_count >= 1:
                    # ── Preview gallery loop ──────────────────────────
                    img_result = None  # will hold the chosen image bytes

                    # Seed a local RNG for deterministic gallery sequences.
                    # Same scene seed ➜ same gallery images every run.
                    base_seed = int(getattr(context.scene, 'seed', 0))
                    if base_seed == 0:
                        gallery_rng = _rng.Random()       # truly random
                    else:
                        gallery_rng = _rng.Random(base_seed)  # deterministic

                    while True:
                        pil_images = []
                        seeds = []
                        # Reset progress for each batch
                        self._phase_progress = 0
                        self._update_overall()
                        for i in range(gallery_count):
                            self._detail_stage = f"Generating preview {i + 1}/{gallery_count}"
                            # Set up remapping so WebSocket progress (0-100 per image)
                            # maps to the correct slice of the overall phase bar.
                            base = (i / gallery_count) * 90
                            span = (1 / gallery_count) * 90
                            self._progress_remap = (base, span)
                            self._phase_progress = base
                            self._update_overall()

                            rand_seed = gallery_rng.randint(1, 2**31 - 1)
                            result = self.workflow_manager.generate_txt2img(
                                context, seed_override=rand_seed)
                            if isinstance(result, dict) and "error" in result:
                                self._error = f"txt2img failed (seed {rand_seed}): {result['error']}"
                                return

                            pil_img = Image.open(io.BytesIO(result))
                            pil_images.append(pil_img)
                            seeds.append(rand_seed)

                        # Clear remapping before waiting
                        self._progress_remap = None

                        # Hand off to the main thread for user selection
                        self._gallery_data = (pil_images, seeds)
                        self._gallery_ready = True
                        self._detail_stage = "Waiting for selection"
                        self._phase_progress = 95
                        self._update_overall()

                        # Block until the modal sets the event
                        self._gallery_event.wait()
                        self._gallery_event.clear()

                        if self._gallery_action == 'select':
                            img_result = self._gallery_selected_bytes
                            chosen_seed = self._gallery_selected_seed
                            if chosen_seed is not None:
                                context.scene.seed = chosen_seed
                            break
                        elif self._gallery_action == 'more':
                            # Loop around and generate another batch
                            continue
                        else:  # cancel
                            self._error = "Preview gallery cancelled"
                            return

                    if img_result is None:
                        self._error = "No image selected from gallery"
                        return
                else:
                    # ── Single image (legacy path) ───────────────────
                    img_result = self.workflow_manager.generate_txt2img(context)
                    if isinstance(img_result, dict) and "error" in img_result:
                        self._error = f"txt2img failed: {img_result['error']}"
                        return

                # Phase 1 complete
                self._phase_progress = 100
                self._update_overall()

                # Early exit if cancelled during txt2img
                if self._cancelled:
                    return

                # Save the generated image bytes to the revision directory
                save_dir = revision_dir if revision_dir else (
                    context.preferences.addons[__package__].preferences.output_dir
                )
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(save_dir, f"trellis2_input_{timestamp}.png")
                with open(image_path, 'wb') as f:
                    f.write(img_result)
                print(f"[TRELLIS2] Saved txt2img result to: {image_path}")

                # Flush VRAM so the txt2img model (SDXL/Flux) is evicted
                # before TRELLIS loads its own models via raw PyTorch.
                # Without this, both models coexist and OOM on <=16 GB GPUs.
                # (Gap B – between txt2img and TRELLIS Phase 1)
                self._detail_stage = "Flushing txt2img models"
                try:
                    server_addr = context.preferences.addons[__package__].preferences.server_address
                    self.workflow_manager._flush_comfyui_vram(server_addr, label="Post-txt2img")
                except Exception:
                    pass

                # Early exit if cancelled during txt2img
                if self._cancelled:
                    return

            # --- Phase 2 (or 1 if no txt2img): TRELLIS.2 mesh generation ---
            trellis_phase = 2 if gen_from == 'prompt' else 1
            self._current_phase = trellis_phase
            self._phase_stage = "TRELLIS.2 Mesh Generation"
            self._phase_progress = 0
            self._detail_progress = 0
            self._detail_stage = "Uploading image"
            self._overall_stage = f"Phase {trellis_phase}/{self._total_phases}: 3D Mesh"
            self._update_overall()

            # Store the final input image path for later use (IPAdapter/Qwen style)
            self._input_image_path = image_path

            result = self.workflow_manager.generate_trellis2(context, image_path)

            # Suppress error reporting when the user cancelled
            if self._cancelled:
                return

            if isinstance(result, dict) and "error" in result:
                self._error = result["error"]
            else:
                self._glb_data = result

        except Exception as e:
            if self._cancelled:
                return  # Swallow exceptions caused by cancel-time WS close
            self._error = str(e)
            traceback.print_exc()
