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


def setup_studio_lighting(context, scale=1.0):
    """Create a three-point studio lighting rig (key, fill, rim).

    Re-usable from any generation mode (TRELLIS.2, PBR decomposition, etc.).

    Args:
        context: Blender context.
        scale: Scene scale factor — lights are placed at ``scale * 2.5`` distance.
    """
    S = max(scale, 0.5)
    dist = S * 2.5

    light_defs = [
        ("SG_Key",  200, 1.5 * S, (1.0, 0.96, 0.90),   45, 40),
        ("SG_Fill",  80, 2.5 * S, (0.90, 0.94, 1.0),   -60, 15),
        ("SG_Rim",  120, 0.8 * S, (1.0, 1.0, 1.0),     170, 55),
    ]

    collection = context.collection
    created = []

    for name, power, size, color, az_deg, el_deg in light_defs:
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        az = math.radians(az_deg)
        el = math.radians(el_deg)
        x = dist * math.cos(el) * math.sin(az)
        y = -dist * math.cos(el) * math.cos(az)
        z = dist * math.sin(el)

        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_data.energy = power
        light_data.size = size
        light_data.color = color

        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        collection.objects.link(light_obj)

        light_obj.location = (x, y, z)
        direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector((x, y, z))
        rot = direction.to_track_quat('-Z', 'Y')
        light_obj.rotation_euler = rot.to_euler()

        created.append(light_obj)

    print(f"[StableGen] Studio lighting created: {[o.name for o in created]}")
    return created


def _pbr_setup_studio_lights(context, to_texture):
    """Calculate scene scale from target objects and set up studio lights."""
    max_dim = 1.0
    for obj in to_texture:
        if hasattr(obj, 'dimensions'):
            max_dim = max(max_dim, *obj.dimensions)
    setup_studio_lighting(context, scale=max_dim)


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
        context.scene.sg_last_gen_error = False
        ComfyUIGenerate._is_running = True

        print("Executing ComfyUI Generation")

        if context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein') and not context.scene.generation_mode == 'project_only':
            context.scene.generation_method = 'sequential' # Force sequential for edit models

        render = bpy.context.scene.render
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
            context.scene.sg_last_gen_error = True
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
        if not controlnet_units and not (context.scene.use_flux_lora and context.scene.model_architecture == 'flux1') and context.scene.model_architecture != 'flux2_klein':
            self.report({'ERROR'}, "At least one ControlNet unit is required to run the operator.")
            context.scene.generation_status = 'idle'
            context.scene.sg_last_gen_error = True
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
                context.scene.sg_last_gen_error = True
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
                context.scene.sg_last_gen_error = True
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
                        (self._generation_method_on_start == 'sequential' or context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein'))):
                    
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
                context.scene.sg_last_gen_error = False
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
                                        export_visibility(context, self._to_texture, camera_visibility=_vis_cam) # Export mask for current view
                                    _emit_cam = _cur_cam
                                    with _SGCameraResolution(context, _emit_cam):
                                        if context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein'): # export custom bg and fallback for Qwen/Klein image edit
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

        # ── PBR Decomposition (runs after ALL cameras are generated) ──
        if getattr(context.scene, 'pbr_decomposition', False):
            self._pbr_maps = {}
            # Collect camera images that need PBR decomposition
            camera_images = {}  # cam_idx → image_path
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
                        existing = self._find_existing_pbr_maps(
                            context, cam_idx)
                        if existing:
                            self._pbr_maps[cam_idx] = existing
                            print(f"[StableGen] Reusing existing PBR maps "
                                  f"for camera {cam_idx}")
                            continue
                        print(f"[StableGen] PBR maps missing for camera "
                              f"{cam_idx}, running decomposition…")
                    camera_images[cam_idx] = cam_image_path
            if camera_images:
                self._run_pbr_decomposition_batched(context, camera_images)

        def image_project_callback():
            if context.scene.generation_method == 'sequential':
                # In sequential mode, projection happened per-camera inside the loop.
                # PBR is projected onto the existing material after all cameras.
                if getattr(context.scene, 'pbr_decomposition', False) and hasattr(self, '_pbr_maps') and self._pbr_maps:
                    from .project import project_pbr_to_bsdf
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
                project_image(context, self._to_texture, self._material_id)
            else:
                # Apply the UV inpainted textures to each mesh
                from .render_tools import apply_uv_inpaint_texture
                for obj in self._to_texture:
                    texture_path = get_file_path(
                        context, "generated_baked", object_name=obj.name, material_id=self._material_id
                    )
                    apply_uv_inpaint_texture(context, obj, texture_path)

            # Project PBR maps onto material after color projection
            if getattr(context.scene, 'pbr_decomposition', False) and hasattr(self, '_pbr_maps') and self._pbr_maps:
                from .project import project_pbr_to_bsdf
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

    # ── PBR Decomposition ──────────────────────────────────────────────
    def _find_existing_pbr_maps(self, context, camera_id):
        """Check if all enabled PBR maps already exist on disk for a camera.

        Returns a dict ``{map_name: file_path}`` when every required map
        is present, or ``None`` if any enabled map is missing (the caller
        should run decomposition in that case).
        """
        scene = context.scene
        existing = {}

        map_checks = []
        if getattr(scene, 'pbr_map_albedo', True):
            map_checks.append('albedo')
        if getattr(scene, 'pbr_map_roughness', True):
            map_checks.append('roughness')
        if getattr(scene, 'pbr_map_metallic', True):
            map_checks.append('metallic')
        if getattr(scene, 'pbr_map_normal', True):
            map_checks.append('normal')
        if getattr(scene, 'pbr_map_height', False):
            map_checks.append('height')
        if getattr(scene, 'pbr_map_emission', False):
            map_checks.append('emission')

        if not map_checks:
            return None

        for map_name in map_checks:
            path = get_file_path(
                context, "pbr", subtype=map_name,
                camera_id=camera_id, material_id=self._material_id
            )
            if os.path.exists(path):
                existing[map_name] = path
            else:
                return None  # At least one required map is missing

        return existing

    # ── Tiled PBR processing ──────────────────────────────────────────

    @staticmethod
    def _create_tile_blend_mask(h, w, overlap_x, overlap_y,
                                fade_left, fade_right,
                                fade_top, fade_bottom):
        """Return a float32 [H, W, 1] weight mask with cosine fade on inner edges.

        Only the edges that border a neighbouring tile (indicated by the
        ``fade_*`` booleans) get a cosine-weighted ramp.
        ``overlap_x`` controls left/right ramp length,
        ``overlap_y`` controls top/bottom ramp length.
        Image-boundary edges stay at full weight (1.0).
        """
        mask = np.ones((h, w, 1), dtype=np.float32)

        if overlap_x > 0:
            ramp_x = np.linspace(0.0, np.pi, overlap_x, dtype=np.float32)
            ramp_x = (1.0 - np.cos(ramp_x)) * 0.5      # 0 → 1 cosine ease
            if fade_left and overlap_x <= w:
                mask[:, :overlap_x, 0] *= ramp_x[np.newaxis, :]
            if fade_right and overlap_x <= w:
                mask[:, -overlap_x:, 0] *= ramp_x[np.newaxis, ::-1]

        if overlap_y > 0:
            ramp_y = np.linspace(0.0, np.pi, overlap_y, dtype=np.float32)
            ramp_y = (1.0 - np.cos(ramp_y)) * 0.5
            if fade_top and overlap_y <= h:
                mask[:overlap_y, :, 0] *= ramp_y[:, np.newaxis]
            if fade_bottom and overlap_y <= h:
                mask[-overlap_y:, :, 0] *= ramp_y[::-1, np.newaxis]

        return mask

    def _process_model_tiled(self, context, image_path, model_name=None,
                              process_fn=None):
        """Run a model on an N×N grid of overlapping tiles.

        Each tile is **upscaled to the full image's longest edge** before
        processing, so the model spends its full resolution budget on
        only 1/N² of the spatial area — effectively N²× the detail.
        The stitched output is at the **upscaled** resolution (~N× the
        original), producing super-resolution PBR maps.

        Args:
            model_name: Passed to ``generate_pbr_maps()``.
            process_fn: Optional callable ``(context, tile_path) → result``.
                        If given, called instead of ``generate_pbr_maps``.
                        May return ``bytes`` (single image) or ``list[bytes]``.

        Returns the same ``list[bytes]`` as ``generate_pbr_maps()``.
        """
        import tempfile

        OVERLAP = 64  # pixels each tile extends beyond its boundary
        MIN_DIM = 256 # skip tiling if either dimension is too small

        scene = context.scene
        N = getattr(scene, 'pbr_tile_grid', 2)

        src = Image.open(image_path)
        W, H = src.size
        longest = max(W, H)

        if W < MIN_DIM or H < MIN_DIM:
            src.close()
            print(f"[StableGen]     Image {W}×{H} too small for tiling, "
                  f"processing normally")
            if process_fn is not None:
                fb = process_fn(context, image_path)
                return [fb] if isinstance(fb, bytes) else fb
            return self.workflow_manager.generate_pbr_maps(
                context, image_path, model_name=model_name)

        # Compute tile boundaries in original-image coordinates
        x_bounds = [W * c // N for c in range(N + 1)]   # [0, W/N, 2W/N, …, W]
        y_bounds = [H * r // N for r in range(N + 1)]

        tile_info = []  # list of (l, t, r, b, fade_left, fade_right, fade_top, fade_bottom)
        for row in range(N):
            for col in range(N):
                l = x_bounds[col]   - (OVERLAP if col > 0     else 0)
                r = x_bounds[col+1] + (OVERLAP if col < N - 1 else 0)
                t = y_bounds[row]   - (OVERLAP if row > 0     else 0)
                b = y_bounds[row+1] + (OVERLAP if row < N - 1 else 0)
                # Clamp to image bounds
                l = max(l, 0);  r = min(r, W)
                t = max(t, 0);  b = min(b, H)
                tile_info.append((
                    l, t, r, b,
                    col > 0,        # fade_left
                    col < N - 1,    # fade_right
                    row > 0,        # fade_top
                    row < N - 1,    # fade_bottom
                ))

        total_tiles = len(tile_info)

        # Uniform scale factor — upscale tiles so longest edge ≈ full image
        sample_tw = tile_info[0][2] - tile_info[0][0]
        sample_th = tile_info[0][3] - tile_info[0][1]
        sample_longest = max(sample_tw, sample_th)
        scale = longest / sample_longest if sample_longest < longest else 1.0

        # ── Process each tile ─────────────────────────────────
        all_tile_results = []   # list of list[bytes]
        tile_up_sizes = []      # (up_w, up_h) per tile
        tmp_dir = tempfile.gettempdir()

        for i, (l, t, r, b, fl, fr, ft, fb) in enumerate(tile_info):
            tw, th = r - l, b - t
            tile_img = src.crop((l, t, r, b))

            tile_longest = max(tw, th)
            if tile_longest < longest:
                up_w = int(round(tw * scale))
                up_h = int(round(th * scale))
                tile_img = tile_img.resize(
                    (up_w, up_h), Image.LANCZOS)
                print(f"[StableGen]     Tile {i+1}/{total_tiles}:  "
                      f"{tw}×{th} → {up_w}×{up_h}  "
                      f"region ({l},{t})→({r},{b})")
            else:
                up_w, up_h = tw, th
                print(f"[StableGen]     Tile {i+1}/{total_tiles}:  "
                      f"{tw}×{th}  region ({l},{t})→({r},{b})")
            tile_up_sizes.append((up_w, up_h))

            tile_path = os.path.join(tmp_dir, f"sg_tile_{i}.png")
            tile_img.save(tile_path)

            if process_fn is not None:
                result = process_fn(context, tile_path)
                # Normalise to list[bytes] (StableDelight returns bytes)
                if isinstance(result, bytes):
                    result = [result]
            else:
                result = self.workflow_manager.generate_pbr_maps(
                    context, tile_path, model_name=model_name,
                    force_native_resolution=True)

            try:
                os.remove(tile_path)
            except OSError:
                pass

            if isinstance(result, dict):
                print(f"[StableGen]     Tile {i+1} failed, falling back "
                      f"to full-image processing")
                src.close()
                if process_fn is not None:
                    fb = process_fn(context, image_path)
                    return [fb] if isinstance(fb, bytes) else fb
                return self.workflow_manager.generate_pbr_maps(
                    context, image_path, model_name=model_name)

            # Resize model output to expected upscaled tile dims
            resized = []
            for map_bytes in result:
                map_img = Image.open(io.BytesIO(map_bytes))
                if map_img.size != (up_w, up_h):
                    map_img = map_img.resize((up_w, up_h), Image.LANCZOS)
                buf = io.BytesIO()
                map_img.save(buf, format='PNG')
                resized.append(buf.getvalue())
            all_tile_results.append(resized)

        src.close()

        # ── Stitch tiles ──────────────────────────────────────
        superres = getattr(scene, 'pbr_tile_superres', False)

        if superres:
            # Super-resolution: stitch at the upscaled tile size (~N× original)
            out_W = int(round(W * scale))
            out_H = int(round(H * scale))
            scaled_overlap = int(round(OVERLAP * scale))
        else:
            # Original resolution: downscale tiles back before stitching
            out_W, out_H = W, H
            scaled_overlap = OVERLAP  # overlap in original coords

        num_maps = len(all_tile_results[0])
        stitched = []

        for map_idx in range(num_maps):
            canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)
            weight = np.zeros((out_H, out_W, 1), dtype=np.float32)

            for i, (l, t, r, b, fl, fr, ft, fb) in enumerate(tile_info):
                tile_bytes = all_tile_results[i][map_idx]
                tile_img = Image.open(io.BytesIO(tile_bytes)).convert('RGB')

                if superres:
                    up_w, up_h = tile_up_sizes[i]
                    sl = int(round(l * scale))
                    st = int(round(t * scale))
                    sr = min(sl + up_w, out_W)
                    sb = min(st + up_h, out_H)
                else:
                    # Downscale tile back to original crop dimensions
                    orig_w, orig_h = r - l, b - t
                    if tile_img.size != (orig_w, orig_h):
                        tile_img = tile_img.resize(
                            (orig_w, orig_h), Image.LANCZOS)
                    sl, st, sr, sb = l, t, r, b

                tile_arr = np.asarray(tile_img, dtype=np.float32)
                tile_arr = tile_arr[:sb - st, :sr - sl]

                overlap_px = 2 * scaled_overlap
                mask = self._create_tile_blend_mask(
                    sb - st, sr - sl,
                    overlap_x=overlap_px,
                    overlap_y=overlap_px,
                    fade_left=fl, fade_right=fr,
                    fade_top=ft, fade_bottom=fb)

                canvas[st:sb, sl:sr] += tile_arr * mask
                weight[st:sb, sl:sr] += mask

            canvas /= np.maximum(weight, 1e-6)
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            out_img = Image.fromarray(canvas, mode='RGB')
            buf = io.BytesIO()
            out_img.save(buf, format='PNG')
            stitched.append(buf.getvalue())

        sr_label = " (super-res)" if superres else ""
        print(f"[StableGen]     Stitched {N}×{N} output: {out_W}×{out_H}"
              f"{sr_label}")

        return stitched

    def _convert_normals_cam_to_world(self, map_bytes, camera_id):
        """Convert a Marigold camera-space normal map to world space.

        Marigold normals are in camera space (X = right, Y = up,
        Z = toward the camera).  The PNG encodes them as
        ``(N + 1) / 2``, mapping [-1, 1] to [0, 1].

        This method:
          1. Decodes the PNG into an RGB float array.
          2. Converts [0, 1] → [-1, 1].
          3. Rotates each normal by the camera's world rotation
             (Marigold OpenGL convention and Blender camera-local
             convention agree: X-right, Y-up, Z-toward-viewer).
          4. Re-normalises and re-encodes to [0, 1] PNG bytes.

        Returns PNG bytes, or ``None`` on failure.
        """
        try:
            img = Image.open(io.BytesIO(map_bytes)).convert('RGB')
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0,1]

            # Decode to [-1, 1]
            normals = arr * 2.0 - 1.0  # [H, W, 3]  (X, Y, Z) camera-space

            # Marigold (OpenGL) convention:
            #   X = right, Y = up, Z = toward camera (out of screen)
            # Blender camera-local convention:
            #   X = right, Y = up, Z = behind camera (camera looks -Z)
            #
            # These AGREE: a surface facing the camera has Z = +1 in
            # both systems (+Z_local points toward the viewer).
            # No axis flip is needed.

            # Get camera rotation (camera-local → world)
            cam = self._cameras[camera_id]
            cam_rot = np.array(cam.matrix_world.to_3x3(), dtype=np.float32)  # 3×3

            # Rotate all normals: N_world = cam_rot @ N_local
            # Reshape to [H*W, 3] for batch matrix multiply
            H, W, _ = normals.shape
            flat = normals.reshape(-1, 3)           # [H*W, 3]
            flat_world = (cam_rot @ flat.T).T       # [H*W, 3]

            # Re-normalise (avoid div-by-zero)
            norms = np.linalg.norm(flat_world, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            flat_world /= norms

            # Encode back to [0, 1] PNG
            world_normals = flat_world.reshape(H, W, 3)
            encoded = np.clip((world_normals + 1.0) / 2.0, 0.0, 1.0)
            encoded_u8 = (encoded * 255.0).astype(np.uint8)

            out_img = Image.fromarray(encoded_u8, mode='RGB')
            buf = io.BytesIO()
            out_img.save(buf, format='PNG')
            print(f"  Converted normal map to world space for camera {camera_id}")
            return buf.getvalue()
        except Exception as err:
            print(f"  Warning: camera→world normal conversion failed: {err}")
            return None

    def _run_pbr_decomposition_batched(self, context, camera_images):
        """Run PBR decomposition **model-first** across all cameras.

        Instead of iterating cameras → models (which forces repeated model
        loading/unloading), this iterates models → cameras so each model's
        weights are loaded once and reused for every camera image.

        Args:
            context: Blender context.
            camera_images: ``{cam_idx: image_path}`` for cameras that need
                processing.
        """
        scene = context.scene
        self._stage = "PBR Decomposition"
        self._progress = 0
        num_cams = len(camera_images)
        cam_ids = sorted(camera_images.keys())
        print(f"[StableGen] Running batched PBR decomposition on "
              f"{num_cams} camera(s)…")

        # Free VRAM before loading PBR models — the previous generation
        # model may still be cached, and Marigold/StableDelight need
        # GPU memory.
        try:
            server_address = context.preferences.addons[
                __package__].preferences.server_address
            self.workflow_manager._flush_comfyui_vram(
                server_address, retries=1, label="PBR pre-load")
        except Exception as e:
            print(f"[StableGen] VRAM flush before PBR failed (non-fatal): {e}")

        want_albedo = getattr(scene, 'pbr_map_albedo', True)
        want_roughness = getattr(scene, 'pbr_map_roughness', True)
        want_metallic = getattr(scene, 'pbr_map_metallic', True)
        want_normal = getattr(scene, 'pbr_map_normal', True)
        want_height = getattr(scene, 'pbr_map_height', False)
        want_emission = getattr(scene, 'pbr_map_emission', False)
        emission_method = getattr(scene, 'pbr_emission_method', 'residual')
        albedo_source = getattr(scene, 'pbr_albedo_source', 'delight')

        use_delight = (want_albedo and albedo_source == 'delight')
        use_lighting_albedo = (want_albedo and albedo_source == 'lighting')

        # ── Determine which Marigold models to run ────────────────────
        need_appearance = (
            (want_albedo and albedo_source == 'marigold')
            or want_roughness
            or want_metallic
        )
        need_normals = want_normal
        need_height = want_height
        # IID-Lighting is needed for 'residual' emission OR 'lighting' albedo
        need_iid_lighting = (
            (want_emission and emission_method == 'residual')
            or use_lighting_albedo
        )

        models_to_run = []
        if need_appearance:
            models_to_run.append((
                'prs-eth/marigold-iid-appearance-v1-1',
                ['albedo', 'material'],
            ))
        if need_normals:
            models_to_run.append((
                'prs-eth/marigold-normals-lcm-v0-1',
                ['normal'],
            ))
        if need_height:
            models_to_run.append((
                'prs-eth/marigold-depth-lcm-v1-0',
                ['height'],
            ))
        if need_iid_lighting:
            models_to_run.append((
                'prs-eth/marigold-iid-lighting-v1-1',
                ['_lighting_albedo', '_lighting_shading', '_lighting_residual'],
            ))

        # Count extra post-processing steps (HSV / VLM emission)
        extra_emission_steps = 0
        if want_emission and emission_method in ('hsv', 'vlm_seg'):
            extra_emission_steps = 1

        total_model_steps = (len(models_to_run)
                             + (1 if use_delight else 0)
                             + extra_emission_steps)
        if total_model_steps == 0:
            print("[StableGen] No PBR maps enabled, skipping decomposition")
            return

        # Activate PBR progress tracking for the UI
        self._pbr_active = True
        self._pbr_step = 0
        self._pbr_total_steps = total_model_steps
        self._pbr_cam_total = num_cams

        # Ensure every camera has an entry in _pbr_maps
        for cam_idx in cam_ids:
            if cam_idx not in self._pbr_maps:
                self._pbr_maps[cam_idx] = {}

        # ── Tiling settings ───────────────────────────────────────────
        tiling_mode = getattr(scene, 'pbr_tiling', 'off')
        tile_grid = getattr(scene, 'pbr_tile_grid', 2)

        tile_model_keys = set()

        # Custom mode per-map toggles (read once, used later)
        custom_tile_albedo = False
        custom_tile_material = False
        custom_tile_normal = False
        custom_tile_height = False
        custom_tile_emission = False
        # Whether IID-Appearance is the albedo provider
        appearance_provides_albedo = (not use_delight
                                      and not use_lighting_albedo)

        if tiling_mode == 'selective':
            # When using lighting albedo, don't tile appearance
            # (its albedo will be discarded — only material channels needed).
            if not use_lighting_albedo:
                tile_model_keys.add('appearance')
            if use_lighting_albedo:
                tile_model_keys.add('lighting')
        elif tiling_mode == 'all':
            tile_model_keys = {'appearance', 'normals', 'depth', 'lighting'}
        elif tiling_mode == 'custom':
            custom_tile_albedo = getattr(scene, 'pbr_tile_albedo', True)
            custom_tile_material = getattr(scene, 'pbr_tile_material', False)
            custom_tile_normal = getattr(scene, 'pbr_tile_normal', False)
            custom_tile_height = getattr(scene, 'pbr_tile_height', False)
            custom_tile_emission = (
                getattr(scene, 'pbr_tile_emission', False)
                and want_emission
                and emission_method == 'residual'
            )

            # IID-Appearance: tile if ANY of its outputs are tiled
            need_tile_appearance = (
                custom_tile_material
                or (custom_tile_albedo and appearance_provides_albedo)
            )
            if need_tile_appearance:
                tile_model_keys.add('appearance')

            # IID-Lighting: tile if albedo or residual/emission output
            # needs tiling
            need_tile_lighting = (
                (use_lighting_albedo and custom_tile_albedo)
                or custom_tile_emission
            )
            if need_tile_lighting:
                tile_model_keys.add('lighting')

            # Single-output models
            if custom_tile_normal:
                tile_model_keys.add('normals')
            if custom_tile_height:
                tile_model_keys.add('depth')

        model_step = 0

        # ── StableDelight pass (all cameras) ──────────────────────────
        if use_delight:
            model_step += 1
            self._pbr_step = model_step
            tile_delight = (tiling_mode in ('selective', 'all')
                            or (tiling_mode == 'custom' and custom_tile_albedo))
            tile_label = f" (tiled {tile_grid}×{tile_grid})" if tile_delight else ""
            print(f"[StableGen]   Model {model_step}/{total_model_steps}: "
                  f"StableDelight{tile_label}")

            for ci, cam_idx in enumerate(cam_ids):
                image_path = camera_images[cam_idx]
                self._pbr_cam = ci
                self._stage = (f"PBR: StableDelight "
                               f"(cam {ci+1}/{num_cams})")
                self._progress = 0
                print(f"    Camera {cam_idx} ({ci+1}/{num_cams})…")

                if tile_delight:
                    result = self._process_model_tiled(
                        context, image_path,
                        process_fn=self.workflow_manager.generate_delight_map)
                    if isinstance(result, list) and result:
                        result = result[0]
                else:
                    result = self.workflow_manager.generate_delight_map(
                        context, image_path
                    )

                if isinstance(result, dict) and "error" in result:
                    print(f"    StableDelight error: {result['error']}")
                    self._warning = f"StableDelight failed: {result['error']}"
                elif isinstance(result, bytes):
                    pbr_path = get_file_path(
                        context, "pbr", subtype="albedo",
                        camera_id=cam_idx,
                        material_id=self._material_id)
                    try:
                        with open(pbr_path, 'wb') as f:
                            f.write(result)
                        self._pbr_maps[cam_idx]['albedo'] = pbr_path
                        print(f"    Saved albedo (delight) → "
                              f"{os.path.basename(pbr_path)}")
                    except Exception as err:
                        print(f"    Failed to save delight albedo: {err}")

        # ── Marigold model passes (all cameras per model) ─────────────
        for model_name, map_names in models_to_run:
            model_step += 1
            self._pbr_step = model_step
            model_short = model_name.split('/')[-1]
            should_tile = any(k in model_short for k in tile_model_keys)
            is_iid = 'appearance' in model_short
            is_iid_lighting = 'lighting' in model_short

            tile_label = f" (tiled {tile_grid}×{tile_grid})" if should_tile else ""
            print(f"[StableGen]   Model {model_step}/{total_model_steps}: "
                  f"{model_name}{tile_label}")

            for ci, cam_idx in enumerate(cam_ids):
                image_path = camera_images[cam_idx]
                self._pbr_cam = ci
                self._stage = (f"PBR: {model_short} "
                               f"(cam {ci+1}/{num_cams})")
                self._progress = 0

                # ── Dual-run logic ─────────────────────────────────
                # Dual-run = run tiled + untiled, then stitch outputs.
                # Needed when some outputs of a multi-output model
                # should be tiled and others should not.
                dual_run_iid = False
                dual_run_iid_reverse = False   # tile material, untile albedo
                dual_run_lighting = False

                if tiling_mode == 'selective':
                    dual_run_iid = (
                        should_tile and is_iid and not use_delight
                    )
                    dual_run_lighting = (
                        should_tile and is_iid_lighting
                        and use_lighting_albedo
                    )
                elif tiling_mode == 'custom' and should_tile:
                    if is_iid:
                        # IID-Appearance: [albedo, material]
                        want_tiled_albedo = (custom_tile_albedo
                                             and appearance_provides_albedo)
                        want_tiled_material = custom_tile_material
                        # Dual-run only when there's a mismatch
                        if want_tiled_albedo and not want_tiled_material:
                            dual_run_iid = True
                        elif want_tiled_material and not want_tiled_albedo:
                            dual_run_iid = True
                            dual_run_iid_reverse = True
                        # If both tiled → full tile (no dual-run)
                        # If neither → shouldn't reach here (should_tile=False)
                    elif is_iid_lighting:
                        # IID-Lighting: [albedo, shading, residual]
                        # Determine per-output tiling needs
                        want_tiled_l_albedo = (use_lighting_albedo
                                               and custom_tile_albedo)
                        want_tiled_l_residual = custom_tile_emission
                        # Shading is always untiled (no benefit).
                        # Dual-run when at least one output is tiled
                        # but not all (shading is never tiled, so dual-run
                        # is needed whenever ANY lighting output is tiled).
                        any_tiled = want_tiled_l_albedo or want_tiled_l_residual
                        if any_tiled:
                            dual_run_lighting = True

                extra = ""
                if dual_run_iid and not dual_run_iid_reverse:
                    extra = " + untiled material"
                elif dual_run_iid and dual_run_iid_reverse:
                    extra = " + untiled albedo"
                elif dual_run_lighting:
                    extra = " + untiled shading"
                print(f"    Camera {cam_idx} ({ci+1}/{num_cams}){extra}…")

                if dual_run_iid:
                    result_tiled = self._process_model_tiled(
                        context, image_path, model_name)
                    result_untiled = self.workflow_manager.generate_pbr_maps(
                        context, image_path, model_name=model_name)
                    if (isinstance(result_tiled, list)
                            and len(result_tiled) >= 2
                            and isinstance(result_untiled, list)
                            and len(result_untiled) >= 2):
                        if dual_run_iid_reverse:
                            # Tile material, keep albedo untiled
                            result = [result_untiled[0], result_tiled[1]]
                        else:
                            # Tile albedo, keep material untiled
                            result = [result_tiled[0], result_untiled[1]]
                    elif isinstance(result_tiled, dict):
                        result = result_tiled
                    elif isinstance(result_untiled, dict):
                        result = result_untiled
                    else:
                        result = (result_tiled
                                  if isinstance(result_tiled, list)
                                  else result_untiled)
                elif dual_run_lighting:
                    # IID-Lighting: stitch per-output from tiled/untiled
                    result_tiled = self._process_model_tiled(
                        context, image_path, model_name)
                    result_untiled = self.workflow_manager.generate_pbr_maps(
                        context, image_path, model_name=model_name)
                    if (isinstance(result_tiled, list)
                            and len(result_tiled) >= 3
                            and isinstance(result_untiled, list)
                            and len(result_untiled) >= 3):
                        # Per-output: pick tiled or untiled based on toggles
                        if tiling_mode == 'custom':
                            tile_l_albedo = (use_lighting_albedo
                                             and custom_tile_albedo)
                            tile_l_residual = custom_tile_emission
                        else:
                            # selective: tile albedo only
                            tile_l_albedo = True
                            tile_l_residual = False
                        r_albedo = (result_tiled[0] if tile_l_albedo
                                    else result_untiled[0])
                        r_shading = result_untiled[1]   # always untiled
                        r_residual = (result_tiled[2] if tile_l_residual
                                      else result_untiled[2])
                        result = [r_albedo, r_shading, r_residual]
                    elif isinstance(result_tiled, dict):
                        result = result_tiled
                    elif isinstance(result_untiled, dict):
                        result = result_untiled
                    else:
                        result = (result_tiled
                                  if isinstance(result_tiled, list)
                                  else result_untiled)
                elif should_tile:
                    # Tile the model even when StableDelight handles albedo —
                    # the IID-Appearance albedo output is discarded by
                    # _save_pbr_map_outputs, but roughness/metallic still
                    # benefit from tiling.
                    result = self._process_model_tiled(
                        context, image_path, model_name)
                else:
                    result = self.workflow_manager.generate_pbr_maps(
                        context, image_path, model_name=model_name)

                if isinstance(result, dict) and "error" in result:
                    print(f"    PBR model error: {result['error']}")
                    self._warning = (f"PBR decomposition failed: "
                                     f"{result['error']}")
                    continue

                # ── Save each output component ────────────────────
                self._save_pbr_map_outputs(
                    context, result, map_names, cam_idx,
                    want_roughness=want_roughness,
                    want_metallic=want_metallic,
                    use_delight=use_delight,
                    use_lighting_albedo=use_lighting_albedo)

        # ── Post-Marigold emission passes ─────────────────────────────
        if want_emission:
            if emission_method == 'residual':
                # IID-Lighting residual was already saved by the model
                # loop above – now gate it with roughness/metallic.
                model_step += 1
                self._pbr_step = model_step
                self._progress = 0
                print(f"[StableGen]   Post-processing emission (residual "
                      f"gating) – step {model_step}/{total_model_steps}")
                for ci, cam_idx in enumerate(cam_ids):
                    self._pbr_cam = ci
                    self._stage = (f"PBR: Emission gating "
                                   f"(cam {ci+1}/{num_cams})")
                    self._progress = (ci / num_cams) * 100
                    self._gate_emission_residual(context, cam_idx)
                self._progress = 100

            elif emission_method == 'hsv':
                model_step += 1
                self._pbr_step = model_step
                self._progress = 0
                print(f"[StableGen]   Emission via HSV threshold "
                      f"– step {model_step}/{total_model_steps}")
                for ci, cam_idx in enumerate(cam_ids):
                    self._pbr_cam = ci
                    self._stage = (f"PBR: HSV emission "
                                   f"(cam {ci+1}/{num_cams})")
                    self._progress = (ci / num_cams) * 100
                    image_path = camera_images[cam_idx]
                    self._generate_emission_hsv(context, cam_idx, image_path)
                self._progress = 100

            elif emission_method == 'vlm_seg':
                model_step += 1
                self._pbr_step = model_step
                self._progress = 0
                print(f"[StableGen]   Emission via VLM segmentation "
                      f"– step {model_step}/{total_model_steps}")
                for ci, cam_idx in enumerate(cam_ids):
                    self._pbr_cam = ci
                    self._stage = (f"PBR: VLM emission "
                                   f"(cam {ci+1}/{num_cams})")
                    image_path = camera_images[cam_idx]
                    self._generate_emission_vlm(context, cam_idx, image_path)

        # Deactivate PBR progress tracking
        self._pbr_active = False
        self._pbr_step = 0
        self._pbr_total_steps = 0

        print(f"[StableGen] PBR decomposition complete for "
              f"{num_cams} camera(s)")

    def _save_pbr_map_outputs(self, context, result, map_names, camera_id,
                               want_roughness=True, want_metallic=True,
                               use_delight=False, use_lighting_albedo=False):
        """Save the output images from a single model run to disk.

        Handles the IID material channel split (R=roughness, G=metallic)
        and the camera→world normal conversion.

        Args:
            result: list of bytes (one per output map).
            map_names: list of names corresponding to each output.
            camera_id: Camera index.
            want_roughness / want_metallic: Whether to save those channels.
            use_delight: Whether StableDelight handles albedo (skip IID albedo).
        """
        scene = context.scene
        for i, map_bytes in enumerate(result):
            map_name = map_names[i] if i < len(map_names) else f"component_{i}"

            # ── IID-Appearance "material" channel split ───────────
            if map_name == 'material':
                try:
                    mat_img = Image.open(io.BytesIO(map_bytes)).convert('RGB')
                except Exception as err:
                    print(f"    Failed to decode material image: {err}")
                    continue

                if want_roughness:
                    rough_img = mat_img.getchannel('R').convert('L')
                    rough_path = get_file_path(
                        context, "pbr", subtype="roughness",
                        camera_id=camera_id,
                        material_id=self._material_id)
                    try:
                        rough_img.save(rough_path)
                        self._pbr_maps[camera_id]['roughness'] = rough_path
                        print(f"    Saved roughness (material R) → "
                              f"{os.path.basename(rough_path)}")
                    except Exception as err:
                        print(f"    Failed to save roughness: {err}")

                if want_metallic:
                    metal_img = mat_img.getchannel('G').convert('L')
                    metal_path = get_file_path(
                        context, "pbr", subtype="metallic",
                        camera_id=camera_id,
                        material_id=self._material_id)
                    try:
                        metal_img.save(metal_path)
                        self._pbr_maps[camera_id]['metallic'] = metal_path
                        print(f"    Saved metallic (material G) → "
                              f"{os.path.basename(metal_path)}")
                    except Exception as err:
                        print(f"    Failed to save metallic: {err}")
                continue

            # ── IID-Lighting intermediate outputs ─────────────────
            # These are prefixed with '_lighting_' to avoid toggle
            # checks.  The residual is saved for later gating; albedo
            # and shading are discarded unless using lighting albedo.
            if map_name.startswith('_lighting_'):
                lighting_key = map_name  # e.g. '_lighting_residual'
                lighting_path = get_file_path(
                    context, "pbr", subtype=lighting_key.lstrip('_'),
                    camera_id=camera_id,
                    material_id=self._material_id)
                try:
                    with open(lighting_path, 'wb') as f:
                        f.write(map_bytes)
                    self._pbr_maps[camera_id][lighting_key] = lighting_path
                    print(f"    Saved {lighting_key} → "
                          f"{os.path.basename(lighting_path)}")

                    # When using IID-Lighting as albedo source, copy the
                    # lighting albedo to the main albedo slot as well.
                    if map_name == '_lighting_albedo' and use_lighting_albedo:
                        albedo_path = get_file_path(
                            context, "pbr", subtype="albedo",
                            camera_id=camera_id,
                            material_id=self._material_id)
                        with open(albedo_path, 'wb') as f:
                            f.write(map_bytes)
                        self._pbr_maps[camera_id]['albedo'] = albedo_path
                        print(f"    Saved albedo (IID-Lighting) → "
                              f"{os.path.basename(albedo_path)}")
                except Exception as err:
                    print(f"    Failed to save {lighting_key}: {err}")
                continue

            # Skip maps the user didn't enable
            toggle_attr = f"pbr_map_{map_name}"
            if hasattr(scene, toggle_attr) and not getattr(scene, toggle_attr):
                continue

            # Skip Marigold IID-Appearance albedo when another source handles it
            if map_name == 'albedo' and (use_delight or use_lighting_albedo):
                continue

            pbr_path = get_file_path(
                context, "pbr", subtype=map_name,
                camera_id=camera_id,
                material_id=self._material_id)
            try:
                # ── Camera→world-space conversion for normals ─────
                if map_name == 'normal':
                    converted = self._convert_normals_cam_to_world(
                        map_bytes, camera_id)
                    if converted is not None:
                        map_bytes = converted

                with open(pbr_path, 'wb') as f:
                    f.write(map_bytes)
                self._pbr_maps[camera_id][map_name] = pbr_path
                print(f"    Saved {map_name} → "
                      f"{os.path.basename(pbr_path)}")
            except Exception as err:
                print(f"    Failed to save PBR map {map_name}: {err}")

    # ── Emission extraction methods ───────────────────────────────────

    def _gate_emission_residual(self, context, camera_id):
        """Gate the IID-Lighting residual with roughness/metallic to
        produce a cleaner emission map.

        emission = residual × (1 − metallic) × roughness_mask, then
        threshold.  High-metallic + low-roughness regions are likely
        specular reflections, not true emission.
        """
        scene = context.scene
        cam_maps = self._pbr_maps.get(camera_id, {})
        residual_path = cam_maps.get('_lighting_residual')
        if not residual_path or not os.path.exists(residual_path):
            print(f"    Emission residual not found for camera {camera_id}")
            return

        threshold = getattr(scene, 'pbr_emission_threshold', 0.2)

        try:
            residual = np.array(Image.open(residual_path).convert('RGB')).astype(np.float32) / 255.0

            # Optionally gate with metallic if available — high metallic
            # surfaces produce strong specular that isn't true emission.
            metal_path = cam_maps.get('metallic')
            if metal_path and os.path.exists(metal_path):
                metal_img = Image.open(metal_path).convert('L')
                # Resize metallic to match residual dims (they may differ
                # when tiling/super-res settings aren't identical).
                res_h, res_w = residual.shape[:2]
                if metal_img.size != (res_w, res_h):
                    metal_img = metal_img.resize(
                        (res_w, res_h), Image.Resampling.BILINEAR)
                metallic = np.array(metal_img).astype(np.float32) / 255.0
                gate = (1.0 - metallic)[:, :, np.newaxis]
                residual = residual * gate

            # Apply soft threshold — pixels below the threshold are
            # smoothly faded out rather than hard-clipped.
            luminance = np.mean(residual, axis=2)
            # Smooth fade: rescale [0, threshold] → 0, [threshold, 1] → preserved
            fade = np.clip((luminance - threshold * 0.5) / max(threshold * 0.5, 1e-6),
                           0.0, 1.0)[:, :, np.newaxis]
            emission = residual * fade

            emission_img = Image.fromarray(
                np.clip(emission * 255, 0, 255).astype(np.uint8), mode='RGB')
            emission_path = get_file_path(
                context, "pbr", subtype="emission",
                camera_id=camera_id,
                material_id=self._material_id)
            emission_img.save(emission_path)
            self._pbr_maps[camera_id]['emission'] = emission_path
            print(f"    Saved emission (residual gated) → "
                  f"{os.path.basename(emission_path)}")
        except Exception as err:
            print(f"    Failed to generate emission (residual): {err}")
            import traceback; traceback.print_exc()

    def _generate_emission_hsv(self, context, camera_id, image_path):
        """Extract emission via high-saturation + high-value thresholding
        in HSV space.  Glowing objects retain colour intensity while
        normally-lit surfaces desaturate under bright light.
        """
        scene = context.scene
        sat_min = getattr(scene, 'pbr_emission_saturation_min', 0.5)
        val_min = getattr(scene, 'pbr_emission_value_min', 0.85)
        bloom_radius = getattr(scene, 'pbr_emission_bloom', 5.0)
        threshold = getattr(scene, 'pbr_emission_threshold', 0.3)

        try:
            original = np.array(Image.open(image_path).convert('RGB'))
            rgb_f = original.astype(np.float32) / 255.0

            # RGB → HSV
            maxc = rgb_f.max(axis=2)
            minc = rgb_f.min(axis=2)
            delta = maxc - minc
            saturation = np.where(maxc > 1e-6, delta / maxc, 0.0)
            value = maxc

            # Build binary mask: high saturation AND high value
            mask = ((saturation >= sat_min) & (value >= val_min)).astype(np.float32)

            # Apply bloom (Gaussian blur) to soften edges
            if bloom_radius > 0:
                try:
                    import cv2
                    ksize = int(bloom_radius * 4) | 1  # must be odd
                    mask = cv2.GaussianBlur(mask, (ksize, ksize), bloom_radius)
                except ImportError:
                    # Fallback: simple box blur via PIL
                    from PIL import ImageFilter
                    mask_pil = Image.fromarray(
                        (mask * 255).astype(np.uint8), mode='L')
                    mask_pil = mask_pil.filter(
                        ImageFilter.GaussianBlur(radius=bloom_radius))
                    mask = np.array(mask_pil).astype(np.float32) / 255.0

            # Apply threshold on the blurred mask
            mask = np.where(mask > threshold, mask, 0.0)

            # Multiply mask by original RGB to get coloured emission
            emission = rgb_f * mask[:, :, np.newaxis]

            emission_img = Image.fromarray(
                np.clip(emission * 255, 0, 255).astype(np.uint8), mode='RGB')
            emission_path = get_file_path(
                context, "pbr", subtype="emission",
                camera_id=camera_id,
                material_id=self._material_id)
            emission_img.save(emission_path)
            self._pbr_maps[camera_id]['emission'] = emission_path
            print(f"    Saved emission (HSV threshold) → "
                  f"{os.path.basename(emission_path)}")
        except Exception as err:
            print(f"    Failed to generate emission (HSV): {err}")
            import traceback; traceback.print_exc()

    def _generate_emission_vlm(self, context, camera_id, image_path):
        """Extract emission via Grounding DINO + SAM 2 segmentation.

        Uses ComfyUI nodes to run Grounding DINO with emissive keywords,
        then SAM 2 to get pixel-perfect masks, and multiplies with the
        original image for a coloured emission map.
        """
        scene = context.scene
        keywords = getattr(scene, 'pbr_emission_keywords',
                           'neon sign, LED, fire, candle flame, '
                           'screen display, laser, hologram, glowing crystal')
        threshold = getattr(scene, 'pbr_emission_threshold', 0.3)

        server_address = context.preferences.addons[
            __package__].preferences.server_address

        # Check that required nodes exist
        required_nodes = [
            'GroundingDinoModelLoader (segment anything)',
            'GroundingDinoSAMSegment (segment anything)',
            'SAMModelLoader (segment anything)',
        ]
        for node_class in required_nodes:
            try:
                resp = urllib.request.urlopen(
                    f"http://{server_address}/object_info/{urllib.parse.quote(node_class)}",
                    timeout=get_timeout('api')
                )
                data = json.loads(resp.read())
            except Exception:
                print(f"    VLM emission: ComfyUI node '{node_class}' not found. "
                      f"Skipping VLM emission. Install comfyui_segment_anything.")
                self._warning = (f"VLM Emission requires comfyui_segment_anything. "
                                 f"Node '{node_class}' not found.")
                return

        # Upload image
        from .generator import upload_image_to_comfyui
        image_info = upload_image_to_comfyui(server_address, image_path)
        if image_info is None:
            print(f"    Failed to upload image for VLM emission")
            return

        uploaded_name = image_info.get("name", os.path.basename(image_path))

        # Build the ComfyUI workflow:
        # LoadImage → GroundingDINO detect → SAM segment → mask output
        prompt = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_name}
            },
            "2": {
                "class_type": "GroundingDinoModelLoader (segment anything)",
                "inputs": {
                    "model_name": "GroundingDINO_SwinB (938MB)"
                }
            },
            "3": {
                "class_type": "SAMModelLoader (segment anything)",
                "inputs": {
                    "model_name": "sam_vit_h (2.56GB)"
                }
            },
            "4": {
                "class_type": "GroundingDinoSAMSegment (segment anything)",
                "inputs": {
                    "grounding_dino_model": ["2", 0],
                    "sam_model": ["3", 0],
                    "image": ["1", 0],
                    "prompt": keywords,
                    "threshold": threshold,
                }
            },
            # SAM output 1 is MASK; convert to IMAGE for SaveImageWebsocket
            "4b": {
                "class_type": "MaskToImage",
                "inputs": {"mask": ["4", 1]}
            },
            "5": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["4b", 0]}
            }
        }

        NODES = {"save_image": "5"}

        client_id = str(uuid.uuid4())
        ws = self.workflow_manager._connect_to_websocket(
            server_address, client_id)
        if ws is None:
            print(f"    VLM emission: WebSocket connection failed")
            return

        try:
            images = self.workflow_manager._execute_prompt_and_get_images(
                ws, prompt, client_id, server_address, NODES
            )
        finally:
            try:
                ws.close()
            except Exception:
                pass

        if (images is None or not isinstance(images, dict)
                or "5" not in images or not images["5"]):
            print(f"    VLM emission: No mask output received. "
                  f"The scene may not contain recognisable emissive objects.")
            # Fall back to a black emission map
            try:
                orig = Image.open(image_path).convert('RGB')
                black = Image.new('RGB', orig.size, (0, 0, 0))
                emission_path = get_file_path(
                    context, "pbr", subtype="emission",
                    camera_id=camera_id,
                    material_id=self._material_id)
                black.save(emission_path)
                self._pbr_maps[camera_id]['emission'] = emission_path
                print(f"    Saved emission (VLM – no emissive objects) → "
                      f"{os.path.basename(emission_path)}")
            except Exception:
                pass
            return

        mask_bytes = images["5"][0]
        try:
            mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert('L')).astype(np.float32) / 255.0
            original = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0

            # Resize mask if dimensions differ
            if mask.shape[:2] != original.shape[:2]:
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_pil = mask_pil.resize(
                    (original.shape[1], original.shape[0]),
                    Image.Resampling.BILINEAR)
                mask = np.array(mask_pil).astype(np.float32) / 255.0

            emission = original * mask[:, :, np.newaxis]

            emission_img = Image.fromarray(
                np.clip(emission * 255, 0, 255).astype(np.uint8), mode='RGB')
            emission_path = get_file_path(
                context, "pbr", subtype="emission",
                camera_id=camera_id,
                material_id=self._material_id)
            emission_img.save(emission_path)
            self._pbr_maps[camera_id]['emission'] = emission_path
            print(f"    Saved emission (VLM segmentation) → "
                  f"{os.path.basename(emission_path)}")
        except Exception as err:
            print(f"    Failed to generate emission (VLM): {err}")
            import traceback; traceback.print_exc()

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

        print(f"Depth map saved to: {os.path.join(output_dir, output_file)}.png")
        
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
        context.scene.sg_last_gen_error = False
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
        has_texturing = (tex_mode in ('sdxl', 'flux1', 'qwen_image_edit', 'flux2_klein'))
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
            context.scene.sg_last_gen_error = True
            return {'FINISHED'}

        if self._error:
            self.report({'ERROR'}, f"TRELLIS.2 error: {self._error}")
            context.scene.sg_last_gen_error = True
            return {'CANCELLED'}

        if self._glb_data is None or (isinstance(self._glb_data, dict) and "error" in self._glb_data):
            error_msg = self._glb_data.get("error", "Unknown error") if isinstance(self._glb_data, dict) else "No data received"
            self.report({'ERROR'}, f"TRELLIS.2 failed: {error_msg}")
            context.scene.sg_last_gen_error = True
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
            if tex_mode in ('sdxl', 'flux1', 'qwen_image_edit', 'flux2_klein'):
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
                # Find the 3D viewport area + WINDOW region so add_cameras
                # gets a full context (region_data etc.) even when invoked
                # from a timer-driven modal callback.
                _v3d_area = None
                _v3d_region = None
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
                        _v3d_area = area
                        for reg in area.regions:
                            if reg.type == 'WINDOW':
                                _v3d_region = reg
                                break
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

                    # Use temp_override so add_cameras gets proper region_data
                    if _v3d_area and _v3d_region:
                        with bpy.context.temp_override(area=_v3d_area, region=_v3d_region):
                            bpy.ops.object.add_cameras(**_cam_kwargs)
                    else:
                        bpy.ops.object.add_cameras(**_cam_kwargs)

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
        return setup_studio_lighting(context, scale=import_scale)

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
                scene_cams = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
                for obj in scene_cams:
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
