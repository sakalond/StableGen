import os
import bpy  # pylint: disable=import-error
import numpy as np
import cv2

import uuid
import json
import urllib.request
import socket
import threading
import requests
import traceback
import json
import io
from datetime import datetime

import math
import colorsys
from PIL import Image, ImageEnhance
import mathutils

from .util.helpers import prompt_text, prompt_text_img2img, prompt_text_qwen_image_edit # pylint: disable=relative-beyond-top-level
from .render_tools import export_emit_image, export_visibility, export_canny, bake_texture, prepare_baking, unwrap # pylint: disable=relative-beyond-top-level
from .utils import get_last_material_index, get_generation_dirs, get_file_path, get_dir_path, remove_empty_dirs # pylint: disable=relative-beyond-top-level
from .project import project_image, reinstate_compare_nodes # pylint: disable=relative-beyond-top-level
from .workflows import WorkflowManager

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
                # Identify the OSL script nodes used for raycasting
                if node.type == 'SCRIPT' and node.mode == 'EXTERNAL' and 'raycast.osl' in node.filepath:
                    if 'AngleThreshold' in node.inputs:
                        node.inputs['AngleThreshold'].default_value = new_discard_angle
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
            response = requests.post(upload_url, files=files, data=data, timeout=120)
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

def mirror_camera_matrix_x(cam: bpy.types.Object) -> mathutils.Matrix:
    """
    Mirror a camera across the global X=0 plane, but rebuild the rotation
    so we don't end up with negative scale / flipped handedness.
    """
    m = cam.matrix_world.copy()
    loc = m.to_translation()
    basis = m.to_3x3()

    # Camera conventions: looks along local -Z, local +Y is "up"
    forward = -(basis @ mathutils.Vector((0.0, 0.0, 1.0)))  # world-space view direction
    up      =  (basis @ mathutils.Vector((0.0, 1.0, 0.0)))  # world-space up

    # Mirror position & directions across X=0 (YZ plane)
    loc_m = mathutils.Vector((-loc.x,  loc.y,  loc.z))
    f_m   = mathutils.Vector((-forward.x, forward.y, forward.z))
    u_m   = mathutils.Vector((-up.x,      up.y,      up.z))

    f = f_m.normalized()
    u0 = u_m.normalized()

    # Build an orthonormal basis: right, up, -forward
    r = f.cross(u0)
    if r.length < 1e-6:
        # Degenerate case: fall back to world Z as "up"
        r = f.cross(mathutils.Vector((0.0, 0.0, 1.0)))
    r.normalize()
    u = r.cross(f).normalized()

    # Columns: X = right, Y = up, Z = -forward (camera looks along -Z)
    ori3 = mathutils.Matrix((r, u, -f)).transposed()
    ori4 = ori3.to_4x4()
    ori4.translation = loc_m
    return ori4

class MirrorReproject(bpy.types.Operator):
    """Duplicate & mirror the last projection camera and image.

    - In standard / non-refine workflows: mirrors the image and camera, then calls Reproject.
    - In refine+preserve workflows: mirrors as a *new refine layer* using project_image directly.
    """
    bl_idname = "object.stablegen_mirror_reproject"
    bl_label = "Mirror Last Projection"
    bl_options = {'REGISTER', 'UNDO'}

    _original_method = None
    _original_overwrite_material = None
    _timer = None

    @classmethod
    def poll(cls, context):
        # Same checks as Reproject
        if context.scene.output_timestamp == "":
            return False

        operator = None
        for window in context.window_manager.windows:
            for op in window.modal_operators:
                if (
                    op.bl_idname == 'OBJECT_OT_add_cameras'
                    or op.bl_idname == 'OBJECT_OT_bake_textures'
                    or op.bl_idname == 'OBJECT_OT_collect_camera_prompts'
                    or op.bl_idname == 'OBJECT_OT_test_stable'
                    or op.bl_idname == 'OBJECT_OT_stablegen_reproject'
                    or op.bl_idname == 'OBJECT_OT_stablegen_regenerate'
                    or op.bl_idname == 'OBJECT_OT_stablegen_mirror_reproject'
                    or context.scene.generation_status == 'waiting'
                ):
                    operator = op
                    break
            if operator:
                break
        if operator:
            return False
        return True

    def _mirror_image_file(self, src_path: str, dst_path: str):
        """Load an image from src_path, mirror it horizontally, save to dst_path."""
        import bpy

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        img = bpy.data.images.load(src_path)
        try:
            w, h = img.size
            pixels = list(img.pixels[:])  # flat RGBA array

            mirrored = [0.0] * len(pixels)
            row_stride = w * 4  # RGBA per pixel

            for y in range(h):
                row_start = y * row_stride
                for x in range(w):
                    src_idx = row_start + x * 4
                    mx = w - 1 - x
                    dst_idx = row_start + mx * 4
                    mirrored[dst_idx:dst_idx + 4] = pixels[src_idx:src_idx + 4]

            img.pixels[:] = mirrored
            img.filepath_raw = dst_path
            img.file_format = 'PNG'
            img.save()
        finally:
            bpy.data.images.remove(img)

    def _mirror_camera_matrix_x(self, cam: bpy.types.Object) -> mathutils.Matrix:
        """Mirror a camera across X=0 without negative scale / flipped handedness."""
        m = cam.matrix_world.copy()
        loc = m.to_translation()
        basis = m.to_3x3()

        # Camera convention: looks along local -Z, +Y is up
        forward = -(basis @ mathutils.Vector((0.0, 0.0, 1.0)))
        up      =  (basis @ mathutils.Vector((0.0, 1.0, 0.0)))

        # Mirror across X=0 (YZ plane)
        loc_m = mathutils.Vector((-loc.x,  loc.y,  loc.z))
        f_m   = mathutils.Vector((-forward.x, forward.y, forward.z))
        u_m   = mathutils.Vector((-up.x,      up.y,      up.z))

        f = f_m.normalized()
        u0 = u_m.normalized()

        r = f.cross(u0)
        if r.length < 1e-6:
            r = f.cross(mathutils.Vector((0.0, 0.0, 1.0)))
        r.normalize()
        u = r.cross(f).normalized()

        ori3 = mathutils.Matrix((r, u, -f)).transposed()
        ori4 = ori3.to_4x4()
        ori4.translation = loc_m
        return ori4

    def _get_to_texture(self, context):
        if context.scene.texture_objects == 'all':
            return [
                obj for obj in bpy.context.view_layer.objects
                if obj.type == 'MESH' and not obj.hide_get()
            ]
        else:
            return [
                obj for obj in bpy.context.selected_objects
                if obj.type == 'MESH'
            ]

    def _get_sorted_cameras(self, context):
        cams = [obj for obj in context.scene.objects if obj.type == 'CAMERA']
        cams.sort(key=lambda c: c.name)
        return cams

    def _compute_latest_material_id(self, to_texture):
        max_id = -1
        for obj in to_texture:
            mat_id = get_last_material_index(obj)
            if mat_id > max_id:
                max_id = mat_id
        return max_id

    def _do_refine_mirror(self, context, to_texture):
        """
        Special path for generation_method == 'refine' and refine_preserve == True.
        We treat the mirror as a new refine layer: new material_id, and call project_image directly.
        """
        import shutil

        scene = context.scene

        # 1) Base material id (from existing stack)
        base_id = self._compute_latest_material_id(to_texture)
        if base_id < 0:
            self.report({'ERROR'}, "No StableGen materials found to mirror.")
            return {'CANCELLED'}

        # Next refine layer id
        new_material_id = base_id + 1

        # 2) Cameras & active camera index
        cameras = self._get_sorted_cameras(context)
        if not cameras:
            self.report({'ERROR'}, "No cameras found in the scene.")
            return {'CANCELLED'}

        orig_cam = scene.camera if scene.camera in cameras else cameras[0]
        src_cam_idx = cameras.index(orig_cam)

        # 3) Duplicate & mirror camera
        mir_cam = orig_cam.copy()
        mir_cam.data = orig_cam.data.copy()
        scene.collection.objects.link(mir_cam)
        mir_cam.matrix_world = self._mirror_camera_matrix_x(orig_cam)
        mir_cam.name = orig_cam.name + "_MIRROR"

        # Rebuild camera list so we can get the index of the mirrored camera
        cameras = self._get_sorted_cameras(context)
        dst_cam_idx = cameras.index(mir_cam)

        # 4) Prepare images for ALL cameras in the new refine layer
        #    For each camera:
        #      - copy base_id image -> new_material_id image
        #    Then we'll overwrite the mirrored camera's copy with a horizontally mirrored version.
        for cam_idx, cam_obj in enumerate(cameras):
            src_path = get_file_path(
                context,
                "generated",
                camera_id=cam_idx,
                material_id=base_id,
            )
            dst_path = get_file_path(
                context,
                "generated",
                camera_id=cam_idx,
                material_id=new_material_id,
            )

            if not os.path.exists(src_path):
                # Nothing to copy for this camera; just skip it.
                # project_image will either skip or log an error, but we won't
                # clobber existing textures.
                print(f"[MirrorReproject] No base image for cam {cam_idx} at {src_path}, skipping copy.")
                continue

            # For the mirrored camera index, we'll write a flipped image later,
            # so don't bother copying the original now.
            if cam_idx == dst_cam_idx:
                continue

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            print(f"[MirrorReproject] Copied base image for cam {cam_idx}\n  {src_path}\n-> {dst_path}")

        # 5) Now create the mirrored image just for the mirrored camera in the new layer
        src_path = get_file_path(
            context,
            "generated",
            camera_id=src_cam_idx,
            material_id=base_id,
        )
        if not os.path.exists(src_path):
            self.report(
                {'ERROR'},
                f"No generated image found for active camera index {src_cam_idx} at:\n{src_path}"
            )
            return {'CANCELLED'}

        dst_path = get_file_path(
            context,
            "generated",
            camera_id=dst_cam_idx,
            material_id=new_material_id,
        )

        print(f"[MirrorReproject] Refine mirror image\n  {src_path}\n-> {dst_path}")
        self._mirror_image_file(src_path, dst_path)

        # 6) Call project_image directly as a new refine layer
        project_image(context, to_texture, new_material_id)

        # 7) Delete the temporary mirrored camera
        try:
            bpy.data.objects.remove(mir_cam, do_unlink=True)
        except Exception:
            pass

        self.report(
            {'INFO'},
            f"Refine mirror applied as new layer (material id {new_material_id}). Temporary camera removed."
        )
        return {'FINISHED'}

    def _do_standard_mirror(self, context, to_texture):
        """
        Original behavior: mirror within the current material id.
        For standard / non-refine workflows, we mirror the image and camera,
        then call project_image directly (no Reproject operator).
        """
        scene = context.scene

        # 1) Find the latest StableGen material id
        base_id = self._compute_latest_material_id(to_texture)
        if base_id < 0:
            self.report({'ERROR'}, "No StableGen materials found to mirror.")
            return {'CANCELLED'}

        # 2) Get cameras in the same order StableGen uses
        cameras = self._get_sorted_cameras(context)
        if not cameras:
            self.report({'ERROR'}, "No cameras found in the scene.")
            return {'CANCELLED'}

        # Use active scene camera if possible, otherwise first camera
        orig_cam = scene.camera if scene.camera in cameras else cameras[0]
        src_cam_idx = cameras.index(orig_cam)

        # 3) Duplicate & mirror camera
        mir_cam = orig_cam.copy()
        mir_cam.data = orig_cam.data.copy()
        scene.collection.objects.link(mir_cam)
        mir_cam.matrix_world = self._mirror_camera_matrix_x(orig_cam)
        mir_cam.name = orig_cam.name + "_MIRROR"

        # Rebuild sorted list so we know the mirrored camera's index
        cameras = self._get_sorted_cameras(context)
        dst_cam_idx = cameras.index(mir_cam)

        # 4) Mirror the image on disk for this material id
        src_path = get_file_path(
            context,
            "generated",
            camera_id=src_cam_idx,
            material_id=base_id,
        )
        if not os.path.exists(src_path):
            self.report(
                {'ERROR'},
                f"No generated image found for active camera index {src_cam_idx} at:\n{src_path}"
            )
            # Clean up the temp camera if we bail early
            try:
                bpy.data.objects.remove(mir_cam, do_unlink=True)
            except Exception:
                pass
            return {'CANCELLED'}

        dst_path = get_file_path(
            context,
            "generated",
            camera_id=dst_cam_idx,
            material_id=base_id,
        )
        print(f"[MirrorReproject] Standard mirror image\n  {src_path}\n-> {dst_path}")
        self._mirror_image_file(src_path, dst_path)

        # 5) Project back onto the meshes directly for this material id
        #    (this is the same core projection function used by the generate operator)
        project_image(context, to_texture, base_id)

        # 6) Delete the temporary mirrored camera now that projection is done
        try:
            bpy.data.objects.remove(mir_cam, do_unlink=True)
        except Exception:
            pass

        self.report(
            {'INFO'},
            "Standard mirror applied and temporary camera removed."
        )
        return {'FINISHED'}


    def execute(self, context):
        scene = context.scene
        to_texture = self._get_to_texture(context)

        if not to_texture:
            self.report({'ERROR'}, "No mesh objects found for texturing.")
            return {'CANCELLED'}

        # Branch: refine+preserve vs everything else
        if scene.generation_method == 'refine' and getattr(scene, "refine_preserve", False):
            return self._do_refine_mirror(context, to_texture)
        else:
            return self._do_standard_mirror(context, to_texture)



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

        if context.scene.auto_rescale and ((total_pixels > 1_200_000 or total_pixels < 800_000) or (resolution_x % 8 != 0 or resolution_y % 8 != 0)):
            scale_factor = (1_000_000 / total_pixels) ** 0.5
            render.resolution_x = int(resolution_x * scale_factor)
            render.resolution_y = int(resolution_y * scale_factor)
            # ComfyUI requires resolution to be divisible by 8
            render.resolution_x -= render.resolution_x % 8
            render.resolution_y -= render.resolution_y % 8
            self.report({'INFO'}, f"Resolution automatically rescaled to {render.resolution_x}x{render.resolution_y}.")

        elif total_pixels > 1_200_000:  # 1MP + 20%
            self.report({'WARNING'}, "High resolution detected. Resolutions above 1MP may reduce performance and quality.")
        
        self._cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
        if not self._cameras:
            self.report({'ERROR'}, "No cameras found in the scene.")
            context.scene.generation_status = 'idle'
            ComfyUIGenerate._is_running = False
            return {'CANCELLED'}
        # Sort cameras by name
        self._cameras.sort(key=lambda x: x.name)
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

        uv_slots_needed = len(self._cameras)

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
            # Check if there are enough UV map slots
            if not context.scene.bake_texture and context.scene.generation_method != 'uv_inpaint':
                if not context.scene.overwrite_material or (context.scene.generation_method == 'refine' and context.scene.refine_preserve):
                    if 8 - len(obj.data.uv_layers) < uv_slots_needed:
                        self.report({'ERROR'}, "Not enough UV map slots for all cameras.")
                        context.scene.generation_status = 'idle'
                        ComfyUIGenerate._is_running = False
                        return {'CANCELLED'}
                else: # Overwrite material is enabled
                    uv_maps = set()
                    mesh = obj.data
                    uv_maps = [uv_layer.name for uv_layer in mesh.uv_layers]
                    usable_maps = 0
                    if self._material_id == -1:
                        self._material_id = 0
                    # Count only stablegen UV maps
                    for uv_map in uv_maps:
                        for i in range(uv_slots_needed):
                            if uv_map == f"ProjectionUV_{i}_{self._material_id}":
                                usable_maps += 1
                    if 8 - len(obj.data.uv_layers) + usable_maps < uv_slots_needed:
                            print(f"8 - {len(obj.data.uv_layers)} + {usable_maps} < {uv_slots_needed}")
                            self.report({'ERROR'}, "Not enough UV map slots for all cameras.")
                            context.scene.generation_status = 'idle'
                            ComfyUIGenerate._is_running = False
                            return {'CANCELLED'}
                        
            else: # Baking
                if 8 - len(obj.data.uv_layers) < 1:
                    self.report({'ERROR'}, "Not enough UV map slots for baking. At least 1 slot is required.")

        if not context.scene.overwrite_material or self._material_id == -1 or (context.scene.generation_method == 'refine' and context.scene.refine_preserve):
            self._material_id += 1

        if context.scene.generation_method == 'sequential' and context.scene.sequential_custom_camera_order != "":
            # The format is: index1,index2,index3,...,indexN
            camera_order = context.scene.sequential_custom_camera_order.split(',')
            # Check if there is index for each camera
            if len(camera_order) != len(self._cameras):
                self.report({'ERROR'}, "The number of indices in the custom camera order must match the number of cameras.")
                context.scene.generation_status = 'idle'
                ComfyUIGenerate._is_running = False
                return {'CANCELLED'}
            # Make a backup of all cameras, remove and then add them in the custom order
            cameras = self._cameras.copy()
            cameras_backup = [camera.copy() for camera in cameras]
            for camera in cameras:
                bpy.data.objects.remove(camera)
            self._cameras = []
            # Re-add the cameras in the custom order
            for i, index in enumerate(camera_order):
                camera = cameras_backup[int(index)]
                # Rename the camera to match the index
                camera.name = f"Camera_{i}"
                self._cameras.append(camera)
                bpy.context.scene.collection.objects.link(camera)

        if context.scene.generation_mode == 'standard':
            # If there is depth controlnet unit
            if any(unit["unit_type"] == "depth" for unit in controlnet_units) or (context.scene.use_flux_lora and context.scene.model_architecture == 'flux1') or (context.scene.model_architecture == 'qwen_image_edit' and context.scene.qwen_guidance_map_type == 'depth'):
                if context.scene.generation_method != 'uv_inpaint':
                    # Export depth maps for each camera
                    for i, camera in enumerate(self._cameras):
                        bpy.context.scene.camera = camera
                        self.export_depthmap(context, camera_id=i)
                    if context.scene.generation_method == 'grid':
                        self.combine_maps(context, self._cameras, type="depth")
            # If there is canny controlnet unit
            if any(unit["unit_type"] == "canny" for unit in controlnet_units):
                if context.scene.generation_method != 'uv_inpaint':
                    # Export canny maps for each camera
                    for i, camera in enumerate(self._cameras):
                        bpy.context.scene.camera = camera
                        export_canny(context, camera_id=i, low_threshold=context.scene.canny_threshold_low, high_threshold=context.scene.canny_threshold_high)
                    if context.scene.generation_method == 'grid':
                        self.combine_maps(context, self._cameras, type="canny")
            # If there is normal controlnet unit
            if any(unit["unit_type"] == "normal" for unit in controlnet_units) or (context.scene.model_architecture == 'qwen_image_edit' and context.scene.qwen_guidance_map_type == 'normal'):
                if context.scene.generation_method != 'uv_inpaint':
                    # Export normal maps for each camera
                    for i, camera in enumerate(self._cameras):
                        bpy.context.scene.camera = camera
                        self.export_normal(context, camera_id=i)
                    if context.scene.generation_method == 'grid':
                        self.combine_maps(context, self._cameras, type="normal")

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

        # Refine mode preparation
        if context.scene.generation_method == 'refine':
            for i, camera in enumerate(self._cameras):
                bpy.context.scene.camera = camera
                export_emit_image(context, self._to_texture, camera_id=i)

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
                export_visibility(context, None, obj)
        
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
            # Reset weights for selected viewpoints
            # Prepare list of camera_ids and material_ids to reset weights for
            ids = []
            for camera_id in self._selected_camera_ids:
                ids.append((camera_id, self._material_id))
            reinstate_compare_nodes(context, self._to_texture, ids)

        # Add modal timer
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)       
        print("Starting thread") 
        if context.scene.generation_method == 'grid':
            self._thread = threading.Thread(target=self.async_generate, args=(context,))
        else:
            self._thread = threading.Thread(target=self.async_generate, args=(context, 0))
        
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
                        remove_empty_dirs(context)
                        return {'CANCELLED'}
                    self.report({'ERROR'}, self._error)
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
                            # Identify the OSL script nodes used for raycasting
                            if node.type == 'SCRIPT' and node.mode == 'EXTERNAL' and 'raycast.osl' in node.filepath:
                                if 'AngleThreshold' in node.inputs:
                                    node.inputs['AngleThreshold'].default_value = new_discard_angle
                    
                    print("Discard angle reset complete.")

                # If viewport rendering mode is 'Rendered' and mode is 'regenerate_selected', switch to 'Solid' and then back to 'Rendered' to refresh the viewport
                if context.scene.generation_mode == 'regenerate_selected' and context.area.spaces.active.shading.type == 'RENDERED':
                    context.area.spaces.active.shading.type = 'SOLID'
                    context.area.spaces.active.shading.type = 'RENDERED'
                context.scene.display_settings.display_device = 'sRGB'
                context.scene.view_settings.view_transform = 'Standard'
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
        remove_empty_dirs(context)

    def async_generate(self, context, camera_id = None):
        """     
        Asynchronously generates the image using ComfyUI.         
        :param context: Blender context.         
        :return: None     
        """
        self._error = None
        try:
            while self._threads_left > 0 and not context.scene.generation_mode == 'project_only':
                if context.scene.steps != 0 and not (context.scene.generation_mode == 'regenerate_selected' and camera_id not in self._selected_camera_ids):
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
                    if context.scene.generation_method == 'refine':
                        render_info = self._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=camera_id)
                    elif context.scene.generation_method == 'sequential' and self._current_image > 0:
                        render_info = self._get_uploaded_image_info(context, "inpaint", subtype="render", camera_id=self._current_image)
                        mask_info = self._get_uploaded_image_info(context, "inpaint", subtype="visibility", camera_id=self._current_image)

                    # Get info for IPAdapter reference image
                    if context.scene.use_ipadapter:
                        ipadapter_ref_info = self._get_uploaded_image_info(context, "custom", filename=bpy.path.abspath(context.scene.ipadapter_image))
                    elif context.scene.sequential_ipadapter and self._current_image > 0:
                        cam_id = 0 if context.scene.sequential_ipadapter_mode == 'first' else self._current_image - 1
                        ipadapter_ref_info = self._get_uploaded_image_info(context, "generated", camera_id=cam_id, material_id=self._material_id)

                    # Filter out None values from controlnet_info
                    controlnet_info = {k: v for k, v in controlnet_info.items() if v is not None}
                    # End Prepare Image Info

                    # Generate image without ControlNet if needed
                    if context.scene.generation_mode == 'standard' and camera_id == 0 and (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'refine')\
                            and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate and not context.scene.use_ipadapter and context.scene.sequential_ipadapter_mode == 'first':
                        self._stage = "Generating Reference Image"
                        # Don't use ControlNet for the first image if sequential_ipadapter_regenerate_wo_controlnet is enabled
                        if context.scene.sequential_ipadapter_regenerate_wo_controlnet:
                            original_strengths = [unit.strength for unit in context.scene.controlnet_units]
                            for unit in context.scene.controlnet_units:
                                unit.strength = 0.0
                    else:
                        self._stage = "Generating Image"
                    self._progress = 0
                    
                    # Generate the image
                    if context.scene.generation_method == 'refine':
                        if context.scene.model_architecture == 'flux1':
                            image = self.workflow_manager.refine_flux(context, controlnet_info=controlnet_info, render_info=render_info, ipadapter_ref_info=ipadapter_ref_info)
                        else:
                            image = self.workflow_manager.refine(context, controlnet_info=controlnet_info, render_info=render_info, ipadapter_ref_info=ipadapter_ref_info)
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
                            def context_callback():
                                # Export visibility mask and render for the current camera, we need to use a callback to be in the main thread
                                export_visibility(context, self._to_texture, camera_visibility=self._cameras[self._current_image - 1]) # Export mask for current view
                                if context.scene.model_architecture == 'qwen_image_edit': # export custom bg and fallback for Qwen image edit
                                    fallback_color, background_color = self._get_qwen_context_colors(context)
                                    export_emit_image(context, self._to_texture, camera_id=self._current_image, bg_color=background_color, fallback_color=fallback_color) # Export render for next view
                                    self._dilate_qwen_context_fallback(context, self._current_image, fallback_color)
                                else:
                                    export_emit_image(context, self._to_texture, camera_id=self._current_image, bg_color=context.scene.fallback_color, fallback_color=context.scene.fallback_color) # Export render for next view
                                self._wait_event.set()
                                return None
                            bpy.app.timers.register(context_callback)
                            self._wait_event.wait()
                            self._wait_event.clear()
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
                    if camera_id == 0 and (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate' or context.scene.generation_method == 'refine')\
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
                            if context.scene.generation_method == "refine":
                                image = self.workflow_manager.refine(context, controlnet_info=controlnet_info, render_info=render_info, mask_info=mask_info, ipadapter_ref_info=ipadapter_ref_info)
                            else:
                                image = self.workflow_manager.generate(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                        elif context.scene.model_architecture == "flux1":
                            if context.scene.generation_method == "refine":
                                image = self.workflow_manager.refine_flux(context, controlnet_info=controlnet_info, render_info=render_info, mask_info=mask_info, ipadapter_ref_info=ipadapter_ref_info)
                            else:
                                image = self.workflow_manager.generate_flux(context, controlnet_info=controlnet_info, ipadapter_ref_info=ipadapter_ref_info)
                        context.scene.use_ipadapter = False
                        image_path = image_path.replace(".png", "_ipadapter.png")
                        with open(image_path, 'wb') as f:
                            f.write(image)
                    
                     # Sequential mode callback
                    if context.scene.generation_method == 'sequential':
                        def image_project_callback():
                            redraw_ui(context)
                            project_image(context, self._to_texture, self._material_id, stop_index=self._current_image)
                            # Set the event to signal the end of the process
                            self._wait_event.set()
                            return None
                        bpy.app.timers.register(image_project_callback)
                        # Wait for the event to be set
                        self._wait_event.wait()
                        self._wait_event.clear()
                        # Update info for the next iteration (if any)
                        if self._current_image < len(self._cameras) - 1:
                            next_camera_id = self._current_image + 1
                            # ControlNet info will be re-fetched at the start of the next loop iteration
                else: # steps == 0, skip generation
                    pass # No image generation needed

                if context.scene.generation_method == 'separate' or context.scene.generation_method == 'refine' or context.scene.generation_method == 'sequential':
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
            return

        def image_project_callback():
            if context.scene.generation_method == 'sequential':
                return None
            self._stage = "Projecting Image"
            if context.scene.bake_texture:
                self._stage = "Baking Textures & Projecting"
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
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links
        
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
        links.new(normalize_node.outputs[0], invert_node.inputs[1])

        # Add an output file node
        output_node = nodes.new(type="CompositorNodeOutputFile")
        output_node.location = (600, 0)
        output_node.base_path = output_dir
        output_node.file_slots[0].path = output_file
        output_node.format.file_format = "PNG"  # Save as PNG
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

        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
        bpy.context.scene.view_settings.view_transform = 'Raw'
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.use_nodes = True

        # Clear existing nodes.
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links
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
        output_node.base_path = output_dir
        output_node.file_slots[0].path = output_file
        output_node.format.file_format = "PNG"
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
        from PIL import Image

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
        from PIL import Image
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
