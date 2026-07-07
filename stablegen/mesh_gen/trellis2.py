"""TRELLIS.2 mesh generation operator."""

import os
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import json
import uuid
import urllib.request
import urllib.parse
import threading
import traceback
from datetime import datetime
import math
import io
import websocket
from PIL import Image

from ..utils import get_generation_dirs, sg_modal_active
from ..timeout_config import get_timeout
from .._generator_utils import setup_studio_lighting, redraw_ui, upload_image_to_comfyui
from ..texturing.gallery import _PreviewGalleryOverlay

_ADDON_PKG = __package__.rsplit('.', 1)[0]

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

    @staticmethod
    def _clear_imported_custom_normals(mesh_objects):
        """Clear TRELLIS/GLB custom normals without changing object hierarchy."""
        if not mesh_objects:
            return

        previous_active = bpy.context.view_layer.objects.active
        previous_selection = list(bpy.context.selected_objects)
        cleared = 0

        try:
            if bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        try:
            for obj in mesh_objects:
                if obj.type != 'MESH':
                    continue
                if hasattr(obj.data, "has_custom_normals") and not obj.data.has_custom_normals:
                    continue

                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                try:
                    bpy.ops.mesh.customdata_custom_splitnormals_clear()
                    cleared += 1
                except Exception as err:
                    print(f"[TRELLIS2] Warning: Could not clear custom normals on {obj.name}: {err}")
        finally:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in previous_selection:
                if bpy.context.scene.objects.get(obj.name):
                    obj.select_set(True)
            if previous_active and bpy.context.scene.objects.get(previous_active.name):
                bpy.context.view_layer.objects.active = previous_active

        if cleared:
            print(f"[TRELLIS2] Cleared custom split normals on {cleared} imported mesh object(s)")

    @staticmethod
    def _disable_not_exported_render_helpers():
        """Disable glTF control/helper geometry for render and bake."""
        disabled_objects = 0
        disabled_collections = 0

        def iter_collection_objects(collection):
            seen = set()

            def visit(col):
                for obj in col.objects:
                    if obj.name in seen:
                        continue
                    seen.add(obj.name)
                    yield obj
                for child in col.children:
                    yield from visit(child)

            yield from visit(collection)

        for collection in bpy.data.collections:
            if "not_exported" not in collection.name.lower():
                continue

            if not collection.hide_render:
                collection.hide_render = True
                disabled_collections += 1

            for obj in iter_collection_objects(collection):
                if not obj.hide_render:
                    obj.hide_render = True
                    disabled_objects += 1

        if disabled_objects or disabled_collections:
            print(
                "[TRELLIS2] Disabled render visibility for "
                f"{disabled_objects} glTF not_exported helper object(s) "
                f"across {disabled_collections} collection(s)."
            )

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return True  # Allow cancellation
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        if not addon_prefs.server_address or not addon_prefs.server_online:
            cls.poll_message_set("ComfyUI server is not connected")
            return False
        if not os.path.exists(addon_prefs.output_dir):
            cls.poll_message_set("Output directory not set or does not exist (check addon preferences)")
            return False
        if bpy.app.online_access == False:
            cls.poll_message_set("Blender's online access is disabled (File → Preferences → System)")
            return False
        gen_from = getattr(context.scene, 'trellis2_generate_from', 'image')
        if gen_from == 'image' and not context.scene.trellis2_input_image:
            cls.poll_message_set("No input image selected for TRELLIS.2 generation")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
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
                server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
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
        self._autorig_enabled = False
        from ..workflows import WorkflowManager
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
        from ..utils import get_generation_dirs
        gen_dirs = get_generation_dirs(context)
        revision_dir = gen_dirs.get("revision", "")

        # Read parameters for background processing
        decimate_method = getattr(scene, 'trellis2_decimate_method', 'server')
        target_faces = getattr(scene, 'trellis2_decimation', 1000000)

        # Start generation in background thread
        self._thread = threading.Thread(
            target=self._run_trellis2,
            args=(context, image_path, gen_from, revision_dir, decimate_method, target_faces),
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
            from ..utils import get_generation_dirs
            gen_dirs = get_generation_dirs(context)
            save_dir = gen_dirs.get("revision", "")
            if not save_dir:
                save_dir = context.preferences.addons[_ADDON_PKG].preferences.output_dir
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
            autorig_enabled = getattr(self, '_autorig_enabled', False)

            # --- Clean hierarchy, unparent meshes, and remove GLTF/GLB structural empties ---
            imported_objects = [obj for obj in context.selected_objects]
            for obj in imported_objects:
                obj["sg_generated"] = True
            self._disable_not_exported_render_helpers()
            mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
            empty_objects = [obj for obj in imported_objects if obj.type == 'EMPTY']

            if mesh_objects:
                # TRELLIS.2/GLB exports can carry broken custom split normals.
                # Clear them for both plain and rigged imports; this only
                # touches mesh data and leaves armature/object hierarchy intact.
                self._clear_imported_custom_normals(mesh_objects)

                if autorig_enabled:
                    # Rigged GLBs need their armature/object hierarchy intact.
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in imported_objects:
                        obj.select_set(True)
                    bpy.context.view_layer.objects.active = mesh_objects[0]
                else:
                    # 1. Unparent meshes while preserving their world space transforms
                    for obj in mesh_objects:
                        # Save world matrix
                        world_mat = obj.matrix_world.copy()
                        # Unparent
                        obj.parent = None
                        # Restore world matrix
                        obj.matrix_world = world_mat

                    # 2. Delete the structural empties (e.g. "world", "geometry_0")
                    if empty_objects:
                        bpy.ops.object.select_all(action='DESELECT')
                        for obj in empty_objects:
                            obj.select_set(True)
                        bpy.ops.object.delete()
                        print(f"[TRELLIS2] Cleaned up {len(empty_objects)} GLB parent empty objects")

                    # 3. Reselect only the imported mesh objects for downstream processing
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in mesh_objects:
                        obj.select_set(True)
                    bpy.context.view_layer.objects.active = mesh_objects[0]

            # --- Normalise imported mesh to a reasonable Blender-unit size ---
            target_bu = getattr(context.scene, 'trellis2_import_scale', 2.0)
            if target_bu > 0:
                imported_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
                if imported_meshes:
                    all_corners = []
                    for obj in imported_meshes:
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
                            if autorig_enabled:
                                root_objects = [obj for obj in imported_objects if obj.parent is None]
                                if not root_objects:
                                    root_objects = imported_objects
                                for obj in root_objects:
                                    obj.scale *= scale_factor
                                print(
                                    f"[TRELLIS2] Scaled rigged import to {target_bu} BU "
                                    f"(factor {scale_factor:.4f}, unapplied)"
                                )
                            else:
                                for obj in imported_meshes:
                                    obj.scale *= scale_factor
                                # Apply scale so downstream code sees unit scale
                                bpy.ops.object.select_all(action='DESELECT')
                                for obj in imported_meshes:
                                    obj.select_set(True)
                                bpy.context.view_layer.objects.active = imported_meshes[0]
                                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                                print(f"[TRELLIS2] Scaled mesh to {target_bu} BU (factor {scale_factor:.4f})")

            # --- Solidification ---
            solidify_mesh = getattr(context.scene, 'trellis2_solidify', False)
            if autorig_enabled and solidify_mesh:
                print("[TRELLIS2] Skipping solidify because SkinTokens auto-rigging is enabled")
                solidify_mesh = False
            if solidify_mesh:
                from ..texturing.print_export import _make_solid_mesh_object
                imported_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
                for obj in imported_meshes:
                    try:
                        print(f"[TRELLIS2] Solidifying mesh: {obj.name} before decimation/retopology...")
                        solid_mesh = _make_solid_mesh_object(obj)
                        old_mesh = obj.data
                        obj.data = solid_mesh
                        bpy.data.meshes.remove(old_mesh)
                        print(f"[TRELLIS2] Solidified mesh: {obj.name} successfully")
                    except Exception as e:
                        print(f"[TRELLIS2] Error solidifying mesh {obj.name}: {e}")

            # --- Local Decimation & Retopology ---
            decimate_method = getattr(context.scene, 'trellis2_decimate_method', 'server')

            remesh_method = getattr(context.scene, 'trellis2_remesh_method', 'qdc')
            
            imported_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if imported_meshes and not autorig_enabled:
                # 1. Fallback decimation on main thread if background decimation was requested but didn't run
                if decimate_method == 'collapse' and not getattr(self, '_trimesh_decimated', False):
                    import trimesh
                    import numpy as np
                    import bmesh
                    
                    target_faces = context.scene.trellis2_decimation
                    for obj in imported_meshes:
                        mesh = obj.data
                        vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                        mesh.vertices.foreach_get("co", vertices)
                        vertices = vertices.reshape((-1, 3))
                        
                        bm = bmesh.new()
                        bm.from_mesh(mesh)
                        bmesh.ops.triangulate(bm, faces=bm.faces)
                        
                        faces = np.empty(len(bm.faces) * 3, dtype=np.int32)
                        for i, face in enumerate(bm.faces):
                            faces[i*3 : (i+1)*3] = [v.index for v in face.verts]
                        bm.free()
                        
                        t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces.reshape((-1, 3)))
                        if len(t_mesh.faces) > target_faces:
                            print(f"[TRELLIS2] Main thread fallback trimesh decimation: {len(t_mesh.faces)} -> {target_faces}")
                            decimated = t_mesh.simplify_quadric_decimation(face_count=target_faces)
                            mesh.clear_geometry()
                            mesh.from_pydata(decimated.vertices, [], decimated.faces.tolist())
                            mesh.update()
                
                # 2. Local remeshing retopology
                if remesh_method == 'quadriflow':
                    self._perform_retopology(context, imported_meshes, remesh_method)
            elif imported_meshes and autorig_enabled and (
                    decimate_method == 'collapse' or remesh_method == 'quadriflow'):
                print("[TRELLIS2] Skipping local decimation/retopology because SkinTokens auto-rigging is enabled")

            # --- Apply shading mode to imported meshes ---
            _shade_mode = getattr(context.scene, 'trellis2_shade_mode', 'flat')
            if _shade_mode == 'smooth':
                bpy.ops.object.shade_smooth()
            elif _shade_mode == 'flat':
                bpy.ops.object.shade_flat()
            elif _shade_mode == 'auto_smooth':
                if hasattr(bpy.ops.object, "shade_auto_smooth"):
                    bpy.ops.object.shade_auto_smooth()
                else:
                    bpy.ops.object.shade_smooth()

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

                    # Use temp_override so add_cameras gets proper region_data.
                    # Temporarily bypass the sg_modal_active() poll guard so
                    # add_cameras can run while this TRELLIS.2 modal is active.
                    from .. import utils as _sg_utils
                    _sg_utils._sg_bypass_modal_check = True
                    try:
                        if _v3d_area and _v3d_region:
                            with bpy.context.temp_override(area=_v3d_area, region=_v3d_region):
                                bpy.ops.object.add_cameras(**_cam_kwargs)
                        else:
                            bpy.ops.object.add_cameras(**_cam_kwargs)
                    finally:
                        _sg_utils._sg_bypass_modal_check = False

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

    def _perform_retopology(self, context, meshes, remesh_method):
        """Perform high-quality local retopology on imported meshes.

        Applies shrinkwrap projection to recover fine details from the high-poly source mesh.
        """
        def progress_cb(pct, status_text):
            self._overall_stage = "Local Processing"
            self._phase_stage = status_text
            self._phase_progress = pct
            self._current_phase = self._total_phases
            self._update_overall()
            
        perform_local_retopology(context, meshes, remesh_method, progress_cb=progress_cb)

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
                    srv = bpy.context.preferences.addons[_ADDON_PKG].preferences.server_address
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

                from .. import utils as _sg_utils
                _sg_utils._sg_bypass_modal_check = True
                try:
                    bpy.ops.object.test_stable('INVOKE_DEFAULT')
                finally:
                    _sg_utils._sg_bypass_modal_check = False
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

    def _run_trimesh_decimation(self, glb_bytes, target_faces):
        """Runs the trimesh decimation directly inside Blender's python (on the background thread)."""
        import io
        import trimesh
        
        try:
            print(f"[TRELLIS2] Starting trimesh decimation in Blender's python (target: {target_faces} faces)...")
            # Load glb from bytes using an in-memory file-like object
            glb_file = io.BytesIO(glb_bytes)
            scene = trimesh.load(glb_file, file_type='glb')
            
            if scene.is_empty:
                print("[TRELLIS2] Scene is empty")
                return None
                
            simplified_any = False
            for name, geom in list(scene.geometry.items()):
                if isinstance(geom, trimesh.Trimesh):
                    orig_faces = len(geom.faces)
                    if orig_faces > target_faces:
                        print(f"[TRELLIS2] Simplifying {name} directly in Blender Python: {orig_faces} -> {target_faces} faces")
                        scene.geometry[name] = geom.simplify_quadric_decimation(face_count=target_faces)
                        simplified_any = True
                    else:
                        print(f"[TRELLIS2] Skipping {name}: already has {orig_faces} faces (target {target_faces})")
            
            if simplified_any:
                # Export to bytes
                out_bytes = scene.export(file_type='glb')
                return out_bytes
            else:
                return glb_bytes
                
        except Exception as e:
            print(f"[TRELLIS2] Error during trimesh decimation: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _run_trellis2(self, context, image_path, gen_from, revision_dir, decimate_method, target_faces):
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
                    server_addr = context.preferences.addons[_ADDON_PKG].preferences.server_address
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
                    context.preferences.addons[_ADDON_PKG].preferences.output_dir
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
                    server_addr = context.preferences.addons[_ADDON_PKG].preferences.server_address
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
            self._autorig_enabled = bool(getattr(context.scene, 'trellis2_autorig_enabled', False))

            result = self.workflow_manager.generate_trellis2(context, image_path)

            # Suppress error reporting when the user cancelled
            if self._cancelled:
                return

            if isinstance(result, dict) and "error" in result:
                self._error = result["error"]
            else:
                self._trimesh_decimated = False
                if decimate_method == 'collapse' and not self._autorig_enabled:
                    self._detail_stage = "Decimating mesh via trimesh"
                    self._phase_stage = "Trimesh decimation"
                    self._phase_progress = 90
                    self._update_overall()
                    
                    try:
                        decimated_bytes = self._run_trimesh_decimation(result, target_faces)
                        if decimated_bytes:
                            result = decimated_bytes
                            self._trimesh_decimated = True
                            print("[TRELLIS2] Background trimesh decimation completed successfully")
                    except Exception as dec_err:
                        print(f"[TRELLIS2] Background trimesh decimation failed: {dec_err}")
                        # Fallback: result is unmodified raw GLB
                elif decimate_method == 'collapse' and self._autorig_enabled:
                    print("[TRELLIS2] Skipping background trimesh decimation because SkinTokens auto-rigging is enabled")
                self._glb_data = result

        except Exception as e:
            if self._cancelled:
                return  # Swallow exceptions caused by cancel-time WS close
            self._error = str(e)
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Standalone Local Post-Processing & Retopology Utilities
# ---------------------------------------------------------------------------

def perform_local_retopology(context, meshes, remesh_method, progress_cb=None):
    """Perform high-quality local retopology on meshes.
    
    Applies shrinkwrap projection to recover fine details from the high-poly source mesh.
    """
    import tempfile
    import subprocess
    import math
    import os
    import traceback
    
    print(f"[StableGen] Starting local retopology solver: remesh={remesh_method}")
    wm = context.window_manager
    wm.progress_begin(0, 100)
    
    try:
        for obj_idx, obj in enumerate(meshes):
            if obj.type != 'MESH':
                continue
                
            def update_progress(pct, status_text):
                base = (obj_idx / len(meshes)) * 100
                span = (1 / len(meshes)) * 100
                overall_pct = base + (pct / 100) * span
                
                wm.progress_update(overall_pct)
                context.workspace.status_text_set(f"StableGen [Local Retopo]: {status_text} ({int(overall_pct)}%)")
                
                if progress_cb:
                    progress_cb(pct, status_text)
                    
                # Force Blender to redraw all panels and viewports immediately
                try:
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                except Exception:
                    pass

            # 1. Duplicate original high-detail mesh as reference for shrinkwrap projection
            update_progress(10, "Preserving high-detail reference...")
            high_poly = obj.copy()
            high_poly.data = obj.data.copy()
            high_poly.name = f"{obj.name}_HighPoly_Reference"
            context.collection.objects.link(high_poly)
            high_poly.hide_set(True) # Hide reference mesh

            # Set active and select the low-poly target mesh
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj

            # 2. Solver Setup
            target_faces = context.scene.trellis2_decimation
            

            def run_quadriflow(faces_count):
                if faces_count > 30000:
                    print(f"[StableGen] Capping Quadriflow target faces from {faces_count} to 30000 for stability")
                    faces_count = 30000
                print(f"[StableGen] Running native C++ Quadriflow solver (target faces: {faces_count})")
                res = {'CANCELLED'}
                try:
                    res = bpy.ops.object.quadriflow_remesh(
                        target_faces=faces_count,
                        use_mesh_symmetry=False,
                        use_preserve_sharp=False,
                        use_preserve_boundary=False
                    )
                except Exception as q_err:
                    print(f"[StableGen] Native Quadriflow custom call failed, retrying with simple signature: {q_err}")
                    res = bpy.ops.object.quadriflow_remesh(target_faces=faces_count)
                
                if 'FINISHED' not in res:
                    raise RuntimeError(f"Quadriflow operator returned non-finished status: {res}")

            def run_solver(faces_count):
                if remesh_method == 'quadriflow':
                    run_quadriflow(faces_count)

            # --- Local Remeshing Solver Stage ---
            update_progress(50, f"Executing {remesh_method} retopology solver...")
            try:
                run_solver(target_faces)
            except Exception as solve_err:
                print(f"[StableGen] Solver failed directly on mesh: {solve_err}. Attempting high-resolution OpenVDB Voxel Repair healing...")
                
                target_bu = getattr(context.scene, 'trellis2_import_scale', 2.0)
                voxel_size = max(0.002, target_bu / 512.0)
                print(f"[StableGen] Healing mesh with OpenVDB Voxel Remesh (size: {voxel_size:.4f})")
                
                obj.data.remesh_voxel_size = voxel_size
                res_vr = {'CANCELLED'}
                try:
                    res_vr = bpy.ops.object.voxel_remesh()
                except Exception as remesh_err:
                    print(f"[StableGen] OpenVDB Voxel Remesh healing failed: {remesh_err}")
                
                # Retry solver on the healed mesh
                print("[StableGen] Retrying solver execution on healed manifold mesh...")
                try:
                    run_solver(target_faces)
                except Exception as retry_err:
                    print(f"[StableGen] Solver failed even after voxel healing: {retry_err}")
                    if target_faces > 30000:
                        stable_target = 20000
                        print(f"[StableGen] Retrying with a mathematically stable target count of {stable_target} quads...")
                        try:
                            run_solver(stable_target)
                        except Exception as final_err:
                            print(f"[StableGen] Fatal: Solver failed even at stable target: {final_err}")
                    else:
                        print(f"[StableGen] Fatal: Solver execution failed: {retry_err}")

            # 4. Detail Projection back from HighPoly_Reference via temporary SHRINKWRAP modifier
            update_progress(80, "Projecting high-detail reference back...")
            print("[StableGen] Snapping quad grid and projecting high-detail reference details back")
            sw_mod = obj.modifiers.new(name="SG_Shrinkwrap", type='SHRINKWRAP')
            sw_mod.target = high_poly
            sw_mod.wrap_method = 'NEAREST_SURFACEPOINT'
            sw_mod.wrap_mode = 'ON_SURFACE'
            
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_apply(modifier=sw_mod.name)
            except Exception as sw_err:
                print(f"[StableGen] Warning: Detail projection shrinkwrap application failed: {sw_err}")

            # 5. Clean up HighPoly_Reference duplicate mesh
            update_progress(95, "Cleaning up reference assets...")
            bpy.ops.object.select_all(action='DESELECT')
            high_poly.select_set(True)
            try:
                bpy.ops.object.delete()
                print(f"[StableGen] Retopology completed successfully for object: {obj.name}")
            except Exception as del_err:
                print(f"[StableGen] Warning: Could not delete high-poly reference: {del_err}")

            # Reselect the final low-poly mesh
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            update_progress(100, "Local retopology completed!")
            
    finally:
        wm.progress_end()
        context.workspace.status_text_set(None)


class StableGenLocalPostProcess(bpy.types.Operator):
    """Run local post-processing (decimation and/or retopology) on the active mesh"""
    bl_idname = "object.stablegen_local_postprocess"
    bl_label = "Run Local Post-Processing"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        scene = context.scene
        decimate_method = getattr(scene, 'trellis2_decimate_method', 'server')
        remesh_method = getattr(scene, 'trellis2_remesh_method', 'qdc')
        solidify = getattr(scene, 'trellis2_solidify', False)
        
        # 1. Check if all post-processing options are disabled/none
        if decimate_method == 'none' and remesh_method == 'none' and not solidify:
            self.report({'WARNING'}, "No local post-processing is enabled (Decimation, Retopology, and Solidification are all disabled/set to None).")
            return {'CANCELLED'}
            
        # 2. Error if set to server-side post-processing
        if decimate_method == 'server' or remesh_method == 'qdc':
            self.report({'ERROR'}, "Server-side processing cannot be run locally. Please select a local decimation/remesh method.")
            return {'CANCELLED'}
            
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object.")
            return {'CANCELLED'}
            
        target_faces = scene.trellis2_decimation
        do_decimate = (decimate_method == 'collapse')
        
        print(f"[StableGen] Running manual local post-processing on {obj.name}: decimate={decimate_method}, remesh={remesh_method}, solidify={solidify}, target={target_faces}")
        
        # Switch to object mode
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        # 3. Run solidification locally on the main thread first if enabled
        if solidify:
            from ..texturing.print_export import _make_solid_mesh_object
            try:
                print(f"[StableGen] Solidifying active mesh: {obj.name}...")
                solid_mesh = _make_solid_mesh_object(obj)
                old_mesh = obj.data
                obj.data = solid_mesh
                bpy.data.meshes.remove(old_mesh)
                print(f"[StableGen] Solidification completed for {obj.name}")
            except Exception as e:
                self.report({'ERROR'}, f"Failed to solidify mesh: {e}")
                return {'CANCELLED'}

            
        # If we need local decimation, run it asynchronously
        if do_decimate:
            import numpy as np
            import bmesh
            
            print(f"[StableGen] Starting async trimesh decimation for {obj.name}...")
            
            # Extract mesh data on main thread
            mesh = obj.data
            vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", vertices)
            vertices = vertices.reshape((-1, 3))
            
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            
            faces = np.empty(len(bm.faces) * 3, dtype=np.int32)
            for i, face in enumerate(bm.faces):
                faces[i*3 : (i+1)*3] = [v.index for v in face.verts]
            bm.free()
            
            faces = faces.reshape((-1, 3))
            
            # Define worker and callback
            def bg_decimate_work():
                import trimesh
                t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                orig_faces = len(t_mesh.faces)
                if orig_faces > target_faces:
                    decimated = t_mesh.simplify_quadric_decimation(face_count=target_faces)
                    return {
                        "vertices": decimated.vertices.tolist(),
                        "faces": decimated.faces.tolist(),
                        "success": True
                    }
                return {"success": False}
                
            def on_decimate_done(result):
                if not result:
                    print("[StableGen] Async decimation failed or was aborted.")
                    return
                
                # Check if object still exists
                if not obj or obj.name not in bpy.context.scene.objects:
                    print("[StableGen] Active mesh object no longer exists.")
                    return
                    
                if result.get("success"):
                    mesh = obj.data
                    mesh.clear_geometry()
                    mesh.from_pydata(result["vertices"], [], result["faces"])
                    mesh.update()
                    print(f"[StableGen] Async decimation written back successfully.")
                else:
                    print(f"[StableGen] Mesh already has fewer faces than target. Bypassed decimation.")
                
                # Post-decimation stage (retopology)
                try:
                    if remesh_method == 'quadriflow':
                        perform_local_retopology(bpy.context, [obj], remesh_method)
                    bpy.context.workspace.status_text_set(None)
                    # Trigger viewport redraw
                    for area in bpy.context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                except Exception as post_err:
                    print(f"[StableGen] Error in post-decimation stage: {post_err}")
            
            # Run async decimation
            from ..core.state import _run_async
            bpy.context.workspace.status_text_set("StableGen: Decimating mesh in background...")
            _run_async(bg_decimate_work, on_decimate_done)
            self.report({'INFO'}, "Trimesh decimation started in the background...")
            return {'FINISHED'}
            
        else:
            # No background decimation needed, run retopology synchronously on main thread
            try:
                if remesh_method == 'quadriflow':
                    perform_local_retopology(context, [obj], remesh_method)
                    
                self.report({'INFO'}, "Local post-processing completed successfully!")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f"Failed to run local post-processing: {e}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}
