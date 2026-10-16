"""AddCameras operator – multi-strategy camera placement."""

import math
import time
import bpy, bmesh  # pylint: disable=import-error
import blf  # pylint: disable=import-error
import numpy as np
import mathutils
from mathutils import Matrix
from ..utils import get_file_path, get_dir_path, sg_modal_active
from .geometry import (
    _existing_camera_directions, _filter_near_existing,
    _fibonacci_sphere_points, _gather_target_meshes,
    _get_mesh_face_data, _filter_bottom_faces, _get_mesh_verts_world,
    _camera_basis, _rotation_from_basis, _compute_silhouette_distance,
    _kmeans_on_sphere, _compute_pca_axes, _greedy_coverage_directions,
    _build_bvh_trees, _ray_occluded, _greedy_select_from_visibility,
    _occ_filter_faces_generator, _occ_vis_count_generator,
    _occ_fom_generator, _occ_tpr_generator,
    _sort_directions_spatially, _classify_camera_direction,
    _compute_per_camera_aspect, _perspective_aspect, _resolution_from_aspect,
    _get_resolution_align, _apply_auto_aspect,
    _store_per_camera_resolution, _get_fov,
)
from .overlays import (
    _sg_hide_label_overlay, _sg_remove_label_overlay,
    _sg_restore_label_overlay, _setup_square_camera_display,
)

# Share of the gap between a camera and its subject that the camera's viewport
# pyramid (and the generated image drawn at its base) may fill.
_DISPLAY_GAP_FRACTION = 0.45


def fit_display_size(cam_obj, verts_world, scene):
    """Shrink a camera's viewport display so it stops short of the subject.

    Fitted cameras can sit closer than Blender's 1 m default display, as top and
    bottom views of a tall character do; their pyramid then reaches into the mesh.
    Never enlarges the display.
    """
    data = cam_obj.data
    basis = cam_obj.matrix_basis
    forward = np.array(basis.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0)))
    gap = float(np.min((verts_world - np.array(basis.translation)) @ forward))
    depth = abs(data.view_frame(scene=scene)[0].z) / max(data.display_size, 1e-6)
    if gap <= 0 or depth <= 0:
        return
    data.display_size = max(0.05, min(data.display_size, _DISPLAY_GAP_FRACTION * gap / depth))

class AddCameras(bpy.types.Operator):
    """Add cameras using various placement strategies and adjust their positions
    
    Uses the active camera as a reference for settings. If there is no active camera, a new one is created based on the viewport.
    
    Placement modes:
    - Orbit Ring: cameras in a circle (inherits elevation from reference camera)
    - Sphere Coverage: Fibonacci spiral for even sphere distribution
    - Auto (Normal-Weighted): K-means clustering of mesh face normals
    - Auto (PCA Axes): cameras along the mesh's principal component axes
    - Auto (Greedy Coverage): iteratively adds cameras that maximise new visible surface area and auto-determines the count
    - Fan from Camera: arc of cameras near the active camera
    
    Tips: 
    - Try to frame the object / scene with minimal margin around it.
    - Aim to achieve a uniform coverage of the object / scene.
    - Areas not visible from any camera won't get textured. (Can still be UV-inpainted)
    - Aspect ratio is set by Blender's output settings (or auto-computed)."""
    bl_category = "ControlNet"
    bl_idname = "object.add_cameras"
    bl_label = "Add Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    placement_mode: bpy.props.EnumProperty(
        name="Placement Mode",
        description="Strategy for placing cameras around the subject",
        items=[
            ('orbit_ring', "Orbit Ring", "Place cameras in a circle around the center (original behaviour). Inherits elevation from the reference camera or viewport"),
            ('hemisphere', "Sphere Coverage", "Distribute cameras evenly across a sphere using a Fibonacci spiral"),
            ('normal_weighted', "Auto (Normal-Weighted)", "Automatically place cameras to cover the most surface area, using K-means on face normals weighted by area"),
            ('pca_axes', "Auto (PCA Axes)", "Place cameras along the mesh's principal axes of variation"),
            ('greedy_coverage', "Auto (Greedy Coverage)", "Iteratively add cameras that maximise new visible surface. Automatically determines the number of cameras needed"),
            ('fan_from_camera', "Fan from Camera", "Spread cameras in an arc around the active camera's orbit position"),
        ],
        default='normal_weighted'
    ) # type: ignore

    num_cameras: bpy.props.IntProperty(
        name="Number of Cameras",
        description="Number of cameras to add (not used by Greedy Coverage which auto-determines count)",
        default=4,
        min=1,
        max=100
    ) # type: ignore

    center_type: bpy.props.EnumProperty(
        name="Center Type",
        description="Type of center for the cameras",
        items=[
            ('object', "Object", "Use the active object as the center"),
            ('view center', "View Center", "Use the view center as the center"),
        ],
        default='object'
    ) # type: ignore

    purge_others: bpy.props.BoolProperty(
        name="Remove Existing Cameras",
        description="Delete ALL existing cameras (including active) before adding new ones",
        default=True
    ) # type: ignore

    consider_existing: bpy.props.BoolProperty(
        name="Consider Existing Cameras",
        description="Treat existing cameras as already-placed directions so auto modes avoid duplicating their coverage",
        default=True
    ) # type: ignore

    fan_angle: bpy.props.FloatProperty(
        name="Fan Angle",
        description="Total angular spread of the fan in degrees",
        default=90.0,
        min=10.0,
        max=350.0
    ) # type: ignore

    coverage_target: bpy.props.FloatProperty(
        name="Coverage Target",
        description="Stop adding cameras when this fraction of surface area is visible (Greedy mode)",
        default=0.95,
        min=0.5,
        max=1.0,
        subtype='FACTOR'
    ) # type: ignore

    max_auto_cameras: bpy.props.IntProperty(
        name="Max Cameras",
        description="Upper limit on cameras for Greedy Coverage mode",
        default=12,
        min=2,
        max=50
    ) # type: ignore

    auto_aspect: bpy.props.EnumProperty(
        name="Auto Aspect Ratio",
        description="Automatically adjust render aspect ratio to match the mesh silhouette",
        items=[
            ('off', "Off", "Use current scene resolution for all cameras"),
            ('shared', "Shared", "Average silhouette aspect across all cameras and set a single scene resolution"),
            ('per_camera', "Per Camera", "Each camera gets its own optimal aspect ratio (resolution is swapped during generation)"),
        ],
        default='per_camera'
    ) # type: ignore

    exclude_bottom: bpy.props.BoolProperty(
        name="Exclude Bottom Faces",
        description="Ignore downward-facing geometry (e.g. flat building undersides) when placing cameras",
        default=True
    ) # type: ignore

    exclude_bottom_angle: bpy.props.FloatProperty(
        name="Bottom Angle Threshold",
        description="Faces whose normal points more than this many degrees below the horizon are excluded",
        default=1.5533,  # 89 degrees in radians
        min=0.1745,       # 10 degrees
        max=1.5708,       # 90 degrees
        subtype='ANGLE',
        unit='ROTATION'
    ) # type: ignore

    auto_prompts: bpy.props.BoolProperty(
        name="Auto View Prompts",
        description="Automatically generate view-direction prompts (e.g. 'front view', 'rear view, from above') for each camera based on the viewport reference orientation",
        default=False
    ) # type: ignore

    review_placement: bpy.props.BoolProperty(
        name="Review Camera Placement",
        description="After placing cameras, fly through each one for review. "
                    "When disabled the cameras are created immediately without the interactive fly-through",
        default=True
    ) # type: ignore

    occlusion_mode: bpy.props.EnumProperty(
        name="Occlusion Handling",
        description="How to account for self-occlusion when choosing camera directions",
        items=[
            ('none', "None (Fast)",
             "Back-face culling only – ignores self-occlusion. Fastest option"),
            ('full_matrix', "Full Occlusion Matrix",
             "Build a complete BVH-validated visibility matrix before greedy selection. Most accurate but slower"),
            ('two_pass', "Two-Pass Refinement",
             "Fast back-face pass, then targeted BVH refinement only for faces with zero true coverage"),
            ('vis_weighted', "Visibility-Weighted",
             "Weight faces by their visibility fraction from 200 directions. "
             "Mostly-occluded faces have reduced influence on camera placement (linear). "
             "Only affects Normal-Weighted mode; other modes fall back to Full Occlusion Matrix"),
            ('vis_interactive', "Interactive Visibility",
             "Like Visibility-Weighted but with a real-time preview: scroll to adjust "
             "the occlusion balance and see cameras reposition instantly. "
             "Only affects Normal-Weighted mode; other modes fall back to Full Occlusion Matrix"),
        ],
        default='none'
    ) # type: ignore

    clamp_elevation: bpy.props.BoolProperty(
        name="Clamp Elevation",
        description="Restrict camera elevation to avoid extreme top-down or bottom-up views "
                    "that diffusion models often struggle with",
        default=False
    ) # type: ignore

    max_elevation_angle: bpy.props.FloatProperty(
        name="Max Elevation",
        description="Maximum upward elevation angle (degrees above horizon). "
                    "Cameras looking further up will be clamped to this angle",
        default=1.2217,  # 70 degrees in radians
        min=0.0,
        max=1.5708,      # 90 degrees
        subtype='ANGLE',
        unit='ROTATION'
    ) # type: ignore

    min_elevation_angle: bpy.props.FloatProperty(
        name="Min Elevation",
        description="Minimum downward elevation angle (degrees below horizon). "
                    "Cameras looking further down will be clamped to this angle",
        default=-0.1745,  # -10 degrees in radians
        min=-1.5708,      # -90 degrees
        max=0.0,
        subtype='ANGLE',
        unit='ROTATION'
    ) # type: ignore

    _timer = None
    _camera_index = 0
    _cameras = []
    _initial_camera = None
    _draw_handle = None
    # Occlusion modal state
    _occ_phase = False
    _occ_gen = None
    _occ_progress = 0.0
    _occ_state = None
    # Interactive visibility preview state
    _vis_preview_phase = False
    _vis_count = None
    _vis_n_candidates = 200
    _vis_balance = 0.2
    _vis_state = None
    _vis_directions = None

    def draw(self, context):
        """Custom dialog layout for the placement mode selector."""
        layout = self.layout
        layout.prop(self, "placement_mode")
        layout.separator()
        is_auto = self.placement_mode in ('hemisphere', 'normal_weighted', 'pca_axes', 'greedy_coverage')
        if self.placement_mode == 'greedy_coverage':
            layout.prop(self, "coverage_target")
            layout.prop(self, "max_auto_cameras")
        else:
            layout.prop(self, "num_cameras")
        if not is_auto:
            layout.prop(self, "center_type")
        if self.placement_mode == 'fan_from_camera':
            layout.prop(self, "fan_angle")
        if is_auto:
            if self.placement_mode != 'hemisphere':
                layout.prop(self, "occlusion_mode")
            layout.prop(self, "auto_aspect")
            layout.prop(self, "exclude_bottom")
            if self.exclude_bottom:
                layout.prop(self, "exclude_bottom_angle")
            layout.prop(self, "auto_prompts")
            layout.prop(self, "clamp_elevation")
            if self.clamp_elevation:
                row = layout.row(align=True)
                row.prop(self, "min_elevation_angle", text="Min")
                row.prop(self, "max_elevation_angle", text="Max")
        layout.prop(self, "review_placement")
        layout.prop(self, "purge_others")
        if not self.purge_others and is_auto:
            layout.prop(self, "consider_existing")

    def draw_callback(self, context):
        # Guard against operator being freed while draw handler is still registered
        try:
            _ = self._occ_phase  # probe attribute access
        except ReferenceError:
            # Operator freed — remove the dangling draw handler
            if AddCameras._draw_handle:
                bpy.types.SpaceView3D.draw_handler_remove(
                    AddCameras._draw_handle, 'WINDOW')
                AddCameras._draw_handle = None
            return

        # ── Occlusion progress display ───────────────────────────────────
        if self._occ_phase:
            font_id = 0
            region = context.region
            rw, rh = region.width, region.height
            pct = self._occ_progress

            # Progress bar background (dark)
            bar_w, bar_h = 300, 18
            bar_x = (rw - bar_w) / 2
            bar_y = rh * 0.10

            # Draw progress text above bar
            msg = f"Computing occlusion… {pct * 100:.0f}%"
            blf.size(font_id, 18)
            tw, _th = blf.dimensions(font_id, msg)
            blf.position(font_id, (rw - tw) / 2, bar_y + bar_h + 6, 0)
            blf.color(font_id, 1.0, 0.85, 0.35, 0.95)
            blf.draw(font_id, msg)

            hint = "Press ESC to cancel"
            blf.size(font_id, 13)
            hw, _hh = blf.dimensions(font_id, hint)
            blf.position(font_id, (rw - hw) / 2, bar_y - 18, 0)
            blf.color(font_id, 0.7, 0.7, 0.7, 0.8)
            blf.draw(font_id, hint)
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            return

        # ── Interactive visibility preview HUD ───────────────────────────
        if self._vis_preview_phase:
            font_id = 0
            region = context.region
            rw, rh = region.width, region.height
            balance = self._vis_balance

            msg = f"Occlusion Balance: {balance:.0%}"
            blf.size(font_id, 22)
            tw, _th = blf.dimensions(font_id, msg)
            blf.position(font_id, (rw - tw) / 2, rh * 0.13, 0)
            blf.color(font_id, 1.0, 0.85, 0.35, 0.95)
            blf.draw(font_id, msg)

            n_cams = len(self._cameras) if self._cameras else 0
            info = f"{n_cams} cameras"
            blf.size(font_id, 16)
            iw, _ih = blf.dimensions(font_id, info)
            blf.position(font_id, (rw - iw) / 2, rh * 0.13 - 26, 0)
            blf.color(font_id, 0.9, 0.9, 0.9, 0.9)
            blf.draw(font_id, info)

            hint = "Scroll \u2191\u2193 to adjust  |  ENTER to confirm  |  ESC to cancel"
            blf.size(font_id, 13)
            hw, _hh = blf.dimensions(font_id, hint)
            blf.position(font_id, (rw - hw) / 2, rh * 0.13 - 48, 0)
            blf.color(font_id, 0.7, 0.7, 0.7, 0.8)
            blf.draw(font_id, hint)
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            return

        # ── Fly-through HUD ──────────────────────────────────────────────
        try:
            count = len(self._cameras)
            idx = self._camera_index
        except Exception:
            return
        if count == 0:
            return
        font_id = 0
        region = context.region
        rw, rh = region.width, region.height

        msg = f"Camera: {idx+1}/{count}  |  Press SPACE to confirm"
        blf.size(font_id, 20)
        text_width, text_height = blf.dimensions(font_id, msg)
        x = (rw - text_width) / 2
        y = rh * 0.10
        blf.position(font_id, x, y, 0)
        blf.draw(font_id, msg)

        # Show auto-generated view label below the main HUD line
        if idx < count:
            cam_obj = self._cameras[idx]
            view_label = cam_obj.get('sg_view_label', '')
            if view_label:
                blf.size(font_id, 16)
                blf.color(font_id, 1.0, 0.85, 0.35, 0.95)
                lw, _lh = blf.dimensions(font_id, view_label)
                blf.position(font_id, (rw - lw) / 2, y - 28, 0)
                blf.draw(font_id, view_label)
                blf.color(font_id, 1.0, 1.0, 1.0, 1.0)

    def execute(self, context):
        # --- Delete existing cameras if requested ---
        if self.purge_others:
            scene = context.scene
            to_remove = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
            for cam in to_remove:
                for col in list(cam.users_collection):
                    col.objects.unlink(cam)
                bpy.data.objects.remove(cam, do_unlink=True)
            for cam_data in list(bpy.data.cameras):
                if not cam_data.users:
                    bpy.data.cameras.remove(cam_data)
            scene.camera = None

        # --- Validate mesh requirement for mesh-based modes ---
        if self.placement_mode in ('hemisphere', 'normal_weighted', 'pca_axes', 'greedy_coverage'):
            target_meshes = _gather_target_meshes(context)
            if not target_meshes:
                self.report({'ERROR'}, "No mesh objects found for this placement mode. Select meshes or ensure the scene has meshes.")
                return {'CANCELLED'}
            total_faces = sum(len(o.data.polygons) for o in target_meshes)
            if total_faces == 0:
                self.report({'ERROR'}, "Target meshes have no faces.")
                return {'CANCELLED'}

        # --- Fallback for center_type ---
        if self.center_type == 'object':
            obj = context.object
            if not obj:
                self.report({'WARNING'}, "No active object found. Using view center instead.")
                self.center_type = 'view center'

        # --- Add draw handler ---
        if AddCameras._draw_handle is None:
            AddCameras._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback, (context,), 'WINDOW', 'POST_PIXEL')

        # --- Determine if this is a "manual" mode (needs ref camera for orbit)
        #     or an "auto" mode (computes its own positions) ---
        is_auto_mode = self.placement_mode in ('hemisphere', 'normal_weighted', 'pca_axes', 'greedy_coverage')

        # --- Determine center location ---
        # Auto modes compute their own center from mesh vertices later, so
        # only the manual modes (orbit_ring, fan_from_camera) need this.
        if is_auto_mode:
            # Placeholder – auto branch overrides center_location from verts
            rv3d = getattr(context, 'region_data', None)
            if rv3d is not None:
                center_location = rv3d.view_location.copy()
            else:
                center_location = mathutils.Vector((0.0, 0.0, 0.0))
        elif self.center_type == 'object':
            obj = context.object
            cursor_loc = context.scene.cursor.location.copy()
            context.scene.cursor.location = obj.location.copy()
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            center_location = obj.location.copy()
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            context.scene.cursor.location = cursor_loc
            center_location = obj.matrix_world.translation + obj.matrix_world.to_3x3() @ (center_location - obj.matrix_world.translation)
        else:
            rv3d = getattr(context, 'region_data', None)
            if rv3d is not None:
                center_location = rv3d.view_location.copy()
            else:
                center_location = mathutils.Vector((0.0, 0.0, 0.0))

        # --- Set or create reference camera ---
        self._initial_camera = context.scene.camera
        using_viewport_ref = False
        if not self._initial_camera and not is_auto_mode:
            # Manual modes: create a camera from the viewport as Camera_0
            rv3d = getattr(context, 'region_data', None)
            if rv3d is None:
                self.report({'ERROR'}, "No 3D viewport available for camera placement")
                return {'CANCELLED'}
            cam_data = bpy.data.cameras.new(name='Camera_0')
            cam_obj = bpy.data.objects.new('Camera_0', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.matrix_world = rv3d.view_matrix.inverted()
            context.scene.camera = cam_obj
            self._initial_camera = cam_obj
            using_viewport_ref = True

        # --- Capture reference camera settings (for lens/sensor/clip copy) ---
        cam_settings = self._initial_camera.data if self._initial_camera else None

        self._cameras.clear()
        self._camera_index = 0

        # --- Branch by placement mode ---
        if self.placement_mode == 'orbit_ring':
            ref_mat = self._initial_camera.matrix_world.copy()
            initial_pos = self._initial_camera.location.copy()
            radius = (initial_pos - center_location).length
            if radius < 0.001:
                radius = 5.0
            self._cameras.append(self._initial_camera)
            self._place_orbit_ring(context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref)
        elif self.placement_mode == 'fan_from_camera':
            ref_mat = self._initial_camera.matrix_world.copy()
            initial_pos = self._initial_camera.location.copy()
            radius = (initial_pos - center_location).length
            if radius < 0.001:
                radius = 5.0
            self._cameras.append(self._initial_camera)
            self._place_fan(context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref)
        else:
            # ============================================================
            # Auto modes: per-camera optimal distance from mesh silhouette
            # ============================================================
            # target_meshes was validated above

            # Borrow settings from existing camera or create temp defaults
            temp_cam_data = None
            if not cam_settings:
                temp_cam_data = bpy.data.cameras.new(name='_sg_temp_cam')
                cam_settings = temp_cam_data

            # Pre-compute combined mesh data across all target meshes
            verts_world = _get_mesh_verts_world(target_meshes)
            mesh_center = verts_world.mean(axis=0)
            center_location = mathutils.Vector(mesh_center.tolist())
            mesh_names = ', '.join(o.name for o in target_meshes)
            self.report({'INFO'}, f"Target meshes ({len(target_meshes)}): {mesh_names}")

            # --- Collect existing camera directions (for "consider existing") ---
            existing_dirs = None
            if not self.purge_others and self.consider_existing:
                existing_dirs = _existing_camera_directions(mesh_center)
                if existing_dirs:
                    self.report({'INFO'},
                        f"Considering {len(existing_dirs)} existing camera(s)")

            # --- Determine directions ---
            if self.placement_mode == 'greedy_coverage':
                normals, areas, centers = _get_mesh_face_data(target_meshes)
                if self.exclude_bottom:
                    normals, areas, centers = _filter_bottom_faces(
                        normals, areas, centers, self.exclude_bottom_angle)

                occ = self.occlusion_mode
                if occ == 'none':
                    directions, coverage = _greedy_coverage_directions(
                        normals, areas,
                        max_cameras=self.max_auto_cameras,
                        coverage_target=self.coverage_target,
                        existing_dirs=existing_dirs)
                    directions = [np.array(d) for d in directions]
                    self.report({'INFO'},
                        f"Greedy coverage (no occlusion): "
                        f"{len(directions)} cameras, {coverage*100:.1f}% coverage")
                else:
                    # Greedy doesn't use K-means; vis modes fall back to FOM
                    eff_occ = occ if occ in ('full_matrix', 'two_pass') else 'full_matrix'
                    # ── Async occlusion via modal ────────────────────────
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    gen_func = (_occ_fom_generator if eff_occ == 'full_matrix'
                                else _occ_tpr_generator)
                    self._occ_gen = gen_func(
                        normals, areas, centers, bvh_trees,
                        max_cameras=self.max_auto_cameras,
                        coverage_target=self.coverage_target,
                        existing_dirs=existing_dirs)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    self._occ_state = {
                        'result_type': 'greedy',
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'occ_label': ('full occlusion' if occ == 'full_matrix'
                                      else 'two-pass'),
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
            elif self.placement_mode == 'hemisphere':
                points = _fibonacci_sphere_points(self.num_cameras)
                directions = [np.array(p) for p in points]
                if self.exclude_bottom:
                    # Remove directions that point more than the threshold below the horizon
                    threshold_z = -math.cos(self.exclude_bottom_angle)
                    directions = [d for d in directions if d[2] >= threshold_z]
                if existing_dirs:
                    directions = _filter_near_existing(directions, existing_dirs)
            elif self.placement_mode == 'normal_weighted':
                normals, areas, centers = _get_mesh_face_data(target_meshes)
                if self.exclude_bottom:
                    normals, areas, centers = _filter_bottom_faces(
                        normals, areas, centers, self.exclude_bottom_angle)

                occ = self.occlusion_mode
                if occ in ('vis_weighted', 'vis_interactive'):
                    # Visibility-count generator (shared by both modes)
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    self._occ_gen = _occ_vis_count_generator(
                        normals, areas, centers, bvh_trees)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    rt = 'vis_kmeans' if occ == 'vis_weighted' else 'vis_interactive'
                    self._occ_state = {
                        'result_type': rt,
                        'normals': normals,
                        'areas': areas,
                        'centers': centers,
                        'num_cameras': self.num_cameras,
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'existing_dirs': existing_dirs,
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
                elif occ != 'none':
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    self._occ_gen = _occ_filter_faces_generator(
                        normals, areas, centers, bvh_trees)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    self._occ_state = {
                        'result_type': 'filter_kmeans',
                        'normals': normals,
                        'areas': areas,
                        'centers': centers,
                        'num_cameras': self.num_cameras,
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'existing_dirs': existing_dirs,
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
                else:
                    k = min(self.num_cameras, len(normals))
                    cluster_dirs = _kmeans_on_sphere(normals, areas, k)
                    directions = [cluster_dirs[i] for i in range(len(cluster_dirs))]
                    if existing_dirs:
                        directions = _filter_near_existing(directions, existing_dirs)
            elif self.placement_mode == 'pca_axes':
                normals, areas, centers = _get_mesh_face_data(target_meshes)
                if self.exclude_bottom:
                    normals, areas, centers = _filter_bottom_faces(
                        normals, areas, centers, self.exclude_bottom_angle)

                occ = self.occlusion_mode
                # PCA doesn't use K-means; vis modes fall back to filter
                if occ not in ('none',):
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    self._occ_gen = _occ_filter_faces_generator(
                        normals, areas, centers, bvh_trees)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    self._occ_state = {
                        'result_type': 'filter_pca',
                        'normals': normals,
                        'areas': areas,
                        'centers': centers,
                        'num_cameras': self.num_cameras,
                        'exclude_bottom': self.exclude_bottom,
                        'exclude_bottom_angle': self.exclude_bottom_angle,
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'existing_dirs': existing_dirs,
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
                else:
                    axes = _compute_pca_axes(verts_world)
                    directions = []
                    for axis in axes:
                        directions.append(axis)
                        directions.append(-axis)
                    if self.exclude_bottom:
                        threshold_z = -math.cos(self.exclude_bottom_angle)
                        directions = [d for d in directions if d[2] >= threshold_z]
                    directions = directions[:min(self.num_cameras, len(directions))]
                    if existing_dirs:
                        directions = _filter_near_existing(directions, existing_dirs)

            # Finalize: sort, aspect ratio, camera creation, auto prompts
            self._finalize_auto_cameras(
                context, directions, verts_world, mesh_center,
                cam_settings, temp_cam_data)

        # --- Start fly-through review ---
        if self.review_placement:
            if not self._start_fly_review(context):
                return {'CANCELLED'}
            return {'RUNNING_MODAL'}
        else:
            # Skip review — just finish immediately
            self._finish_without_review(context)
            return {'FINISHED'}

    # -------------------------------------------------------
    # Auto-mode finalization helpers
    # -------------------------------------------------------

    def _finalize_auto_cameras(self, context, directions, verts_world,
                               mesh_center, cam_settings, temp_cam_data):
        """Sort directions, compute aspect ratios, create cameras, and
        generate auto prompts.  Shared by both the synchronous (no-occlusion)
        and asynchronous (modal occlusion) code paths."""
        ref_dir = None
        if directions:
            rv3d = context.region_data
            if rv3d:
                view_dir = rv3d.view_rotation @ mathutils.Vector((0, 0, -1))
                ref_dir = np.array([-view_dir.x, -view_dir.y, -view_dir.z])
            directions = _sort_directions_spatially(directions, ref_dir)

        # --- Clamp elevation angles to avoid extreme top/bottom views ---
        if self.clamp_elevation and directions:
            min_rad = self.min_elevation_angle
            max_rad = self.max_elevation_angle
            clamped = []
            for d in directions:
                d_np = np.array(d, dtype=float)
                norm = np.linalg.norm(d_np)
                if norm < 1e-8:
                    continue
                d_np /= norm
                elev = math.asin(float(np.clip(d_np[2], -1.0, 1.0)))
                if min_rad <= elev <= max_rad:
                    clamped.append(d_np)
                    continue
                elev = max(min_rad, min(max_rad, elev))
                horiz = math.sqrt(float(d_np[0]**2 + d_np[1]**2))
                if horiz < 1e-8:
                    # Pure vertical direction – pick an arbitrary azimuth
                    azimuth = 0.0
                else:
                    azimuth = math.atan2(float(d_np[1]), float(d_np[0]))
                cos_e = math.cos(elev)
                d_np = np.array([cos_e * math.cos(azimuth),
                                 cos_e * math.sin(azimuth),
                                 math.sin(elev)])
                clamped.append(d_np)

            # Deduplicate near-identical directions that may have been created
            # by clamping several cameras to the same elevation limit.
            _MERGE_COS = math.cos(math.radians(8.0))  # merge within 8°
            deduped = []
            for d in clamped:
                if all(float(np.dot(d, existing)) < _MERGE_COS
                       for existing in deduped):
                    deduped.append(d)
            n_before = len(directions)
            directions = deduped
            if len(directions) < n_before:
                self.report({'INFO'},
                    f"Elevation clamp: {n_before} → {len(directions)} cameras "
                    f"(merged {n_before - len(directions)} near-duplicates)")

        render = context.scene.render
        total_px = render.resolution_x * render.resolution_y
        center_np_for_aspect = verts_world.mean(axis=0)

        if self.auto_aspect == 'shared' and directions:
            dirs_np = [d / np.linalg.norm(d) for d in directions]
            old_x, old_y = render.resolution_x, render.resolution_y
            new_x, new_y = _apply_auto_aspect(dirs_np, context, verts_world)
            if (new_x, new_y) != (old_x, old_y):
                self.report({'INFO'},
                    f"Aspect ratio adjusted: {old_x}x{old_y} → {new_x}x{new_y}")

        if self.auto_aspect == 'per_camera' and directions:
            self._create_cameras_per_aspect(
                context, directions, mesh_center, verts_world,
                cam_settings, total_px, center_np_for_aspect)
        else:
            fov_x, fov_y = _get_fov(cam_settings, context)
            self._create_cameras_from_directions(
                context, directions, mesh_center, verts_world,
                cam_settings, fov_x, fov_y)

        if temp_cam_data:
            bpy.data.cameras.remove(temp_cam_data)

        for cam_obj in self._cameras:
            fit_display_size(cam_obj, verts_world, context.scene)

        if self.auto_prompts and self._cameras:
            ref_front = (ref_dir if ref_dir is not None
                         else np.array([0.0, 1.0, 0.0]))
            mesh_center_np = np.array(mesh_center, dtype=float)
            for cam_obj in self._cameras:
                cam_pos = np.array(cam_obj.location, dtype=float)
                cam_dir = cam_pos - mesh_center_np
                label = _classify_camera_direction(cam_dir, ref_front)
                cam_obj["sg_view_label"] = label
                prompt_item = next(
                    (item for item in context.scene.camera_prompts
                     if item.name == cam_obj.name), None)
                if not prompt_item:
                    prompt_item = context.scene.camera_prompts.add()
                    prompt_item.name = cam_obj.name
                prompt_item.prompt = label
            context.scene.use_camera_prompts = True
            self.report({'INFO'},
                        f"Auto-prompts: assigned view labels to "
                        f"{len(self._cameras)} cameras")
        elif not self.auto_prompts:
            # Clear any pre-existing auto-prompts so stale labels don't persist
            context.scene.camera_prompts.clear()
            context.scene.use_camera_prompts = False
            for cam_obj in self._cameras:
                if 'sg_view_label' in cam_obj:
                    del cam_obj['sg_view_label']
            _sg_remove_label_overlay()

    def _update_vis_cameras(self, context):
        """Regenerate cameras based on current visibility balance setting."""
        # Remove existing preview cameras
        for cam in list(self._cameras):
            cam_data = cam.data
            bpy.data.objects.remove(cam, do_unlink=True)
            if cam_data and not cam_data.users:
                bpy.data.cameras.remove(cam_data)
        self._cameras.clear()

        state = self._vis_state
        vis_count = self._vis_count
        n_cand = self._vis_n_candidates
        vis_fraction = vis_count.astype(float) / n_cand
        balance = self._vis_balance

        exterior = vis_count > 0
        normals = state['normals'][exterior]
        areas_base = state['areas'][exterior]
        vf = vis_fraction[exterior]
        # weight = area × (vis_fraction + balance × (1 - vis_fraction))
        # balance=0: weight = area × vis_fraction (full vis-weighting)
        # balance=1: weight = area (no vis-weighting, all exterior equal)
        weighted_areas = areas_base * (vf + balance * (1.0 - vf))

        existing_dirs = state.get('existing_dirs')
        k = min(state['num_cameras'], len(normals))
        if k > 0 and len(normals) > 0:
            cluster_dirs = _kmeans_on_sphere(normals, weighted_areas, k)
            directions = [cluster_dirs[i] for i in range(len(cluster_dirs))]
            if existing_dirs:
                directions = _filter_near_existing(directions, existing_dirs)
        else:
            directions = []

        # Sort directions
        ref_dir = None
        rv3d = context.region_data
        if rv3d:
            view_dir = rv3d.view_rotation @ mathutils.Vector((0, 0, -1))
            ref_dir = np.array([-view_dir.x, -view_dir.y, -view_dir.z])
        directions = _sort_directions_spatially(directions, ref_dir)
        self._vis_directions = directions

        # Quick camera creation (without full finalize overhead)
        if directions:
            cam_settings = state['cam_settings']
            fov_x, fov_y = _get_fov(cam_settings, context)
            self._create_cameras_from_directions(
                context, directions, state['mesh_center'],
                state['verts_world'], cam_settings, fov_x, fov_y)
            if self._cameras:
                context.scene.camera = self._cameras[0]

        n_ext = int(exterior.sum())
        context.area.header_text_set(
            f"Occlusion Balance: {balance:.0%}  |  "
            f"{len(directions)} cameras, {n_ext} visible faces  |  "
            f"Scroll to adjust  |  ENTER to confirm  |  ESC to cancel")

    def _cleanup_vis_preview(self, context):
        """Clean up interactive visibility preview state."""
        # Delete preview cameras
        for cam in list(self._cameras):
            cam_data = cam.data
            bpy.data.objects.remove(cam, do_unlink=True)
            if cam_data and not cam_data.users:
                bpy.data.cameras.remove(cam_data)
        self._cameras.clear()

        # Clean up temp camera data
        state = self._vis_state
        if state and state.get('temp_cam_data'):
            bpy.data.cameras.remove(state['temp_cam_data'])

        self._vis_preview_phase = False
        self._vis_state = None
        self._vis_count = None
        context.area.header_text_set(None)
        if AddCameras._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(
                AddCameras._draw_handle, 'WINDOW')
            AddCameras._draw_handle = None

    def _start_fly_review(self, context, add_modal_handler=True):
        """Frame the first camera and start fly-through review.

        *add_modal_handler* should be False when called from an already-running
        modal (e.g. after occlusion computation finishes).
        Returns True if fly-through started, False if no cameras were created.
        """
        if not self._cameras:
            self.report({'WARNING'}, "No cameras were created.")
            if AddCameras._draw_handle:
                bpy.types.SpaceView3D.draw_handler_remove(
                    AddCameras._draw_handle, 'WINDOW')
                AddCameras._draw_handle = None
            return False

        rv3d = context.region_data
        context.scene.camera = self._cameras[0]
        if rv3d.view_perspective != 'CAMERA':
            bpy.ops.view3d.view_camera()
        bpy.ops.view3d.view_center_camera()
        try:
            rv3d.view_camera_zoom = 1.0
        except Exception:
            pass

        _sg_hide_label_overlay()
        if add_modal_handler:
            context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(
            0.5, window=context.window)
        self._last_time = time.time()
        bpy.ops.view3d.fly('INVOKE_DEFAULT')
        return True

    def _finish_without_review(self, context):
        """Clean up draw handler and restore state without fly-through."""
        if AddCameras._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(
                AddCameras._draw_handle, 'WINDOW')
            AddCameras._draw_handle = None
        if self._cameras:
            context.scene.camera = self._cameras[0]
        _sg_restore_label_overlay()
        self.report({'INFO'},
                    f"Cameras added successfully ({len(self._cameras)} cameras).")

    # -------------------------------------------------------
    # Placement methods
    # -------------------------------------------------------

    def _place_orbit_ring(self, context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref):
        """Place cameras in a circle (original AddCameras behaviour)."""
        total = self.num_cameras
        count = total - 1 if using_viewport_ref else total
        angle_initial = math.atan2(initial_pos.y - center_location.y, initial_pos.x - center_location.x)
        angle_step = 2 * math.pi / (count + 1)

        for i in range(count):
            angle = (i + 1) * angle_step + angle_initial
            cam_data_new = bpy.data.cameras.new(name=f'Camera_{i+1}')
            cam_obj_new = bpy.data.objects.new(f'Camera_{i+1}', cam_data_new)
            context.collection.objects.link(cam_obj_new)
            if using_viewport_ref:
                delta = (i + 1) * angle_step
                T1 = Matrix.Translation(-center_location)
                Rz = Matrix.Rotation(delta, 4, 'Z')
                T2 = Matrix.Translation(center_location)
                cam_obj_new.matrix_world = T2 @ Rz @ T1 @ ref_mat
            else:
                x = center_location.x + radius * math.cos(angle)
                y = center_location.y + radius * math.sin(angle)
                z = initial_pos.z
                cam_obj_new.location = (x, y, z)
                direction = center_location - cam_obj_new.location
                rot = direction.to_track_quat('-Z', 'Y')
                cam_obj_new.rotation_euler = rot.to_euler()
            self._copy_camera_settings(cam_obj_new, cam_settings)
            self._cameras.append(cam_obj_new)

    def _place_fan(self, context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref):
        """Spread cameras in an arc around the active camera's position."""
        fan_rad = math.radians(self.fan_angle)
        angle_initial = math.atan2(initial_pos.y - center_location.y, initial_pos.x - center_location.x)
        total = self.num_cameras
        count = total - 1 if using_viewport_ref else total
        if count <= 0:
            return
        for i in range(count):
            if count == 1:
                t = 0.0
            else:
                t = (i / (count - 1)) - 0.5  # range -0.5 .. 0.5
            angle = angle_initial + t * fan_rad
            cam_data_new = bpy.data.cameras.new(name=f'Camera_fan_{i}')
            cam_obj_new = bpy.data.objects.new(f'Camera_fan_{i}', cam_data_new)
            context.collection.objects.link(cam_obj_new)
            if using_viewport_ref:
                delta = t * fan_rad
                T1 = Matrix.Translation(-center_location)
                Rz = Matrix.Rotation(delta, 4, 'Z')
                T2 = Matrix.Translation(center_location)
                cam_obj_new.matrix_world = T2 @ Rz @ T1 @ ref_mat
            else:
                x = center_location.x + radius * math.cos(angle)
                y = center_location.y + radius * math.sin(angle)
                z = initial_pos.z
                cam_obj_new.location = (x, y, z)
                direction = center_location - cam_obj_new.location
                rot = direction.to_track_quat('-Z', 'Y')
                cam_obj_new.rotation_euler = rot.to_euler()
            self._copy_camera_settings(cam_obj_new, cam_settings)
            self._cameras.append(cam_obj_new)

    def _create_cameras_from_directions(self, context, directions, center_np,
                                         verts_world, cam_settings, fov_x, fov_y):
        """Shared camera creation for all auto-placement modes.
        Each camera gets its own optimal distance computed from the mesh's
        silhouette extent in that viewing direction."""
        center_vec = mathutils.Vector(center_np.tolist())
        prefix_map = {
            'hemisphere': 'Camera_sphere',
            'normal_weighted': 'Camera_auto',
            'pca_axes': 'Camera_pca',
            'greedy_coverage': 'Camera',
        }
        prefix = prefix_map.get(self.placement_mode, 'Camera')

        for i, d in enumerate(directions):
            d_np = np.array(d, dtype=float)
            dist, aim_off = _compute_silhouette_distance(verts_world, center_np, d_np, fov_x, fov_y)
            dir_vec = mathutils.Vector(d_np.tolist()).normalized()
            aim_point = center_vec + mathutils.Vector(aim_off.tolist())
            pos = aim_point + dir_vec * dist

            cam_data = bpy.data.cameras.new(name=f'{prefix}_{i}')
            cam_obj = bpy.data.objects.new(f'{prefix}_{i}', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.location = pos
            right, up_v, d_unit = _camera_basis(d_np)
            cam_obj.rotation_euler = _rotation_from_basis(right, up_v, d_unit)
            self._copy_camera_settings(cam_obj, cam_settings)
            self._cameras.append(cam_obj)

    def _create_cameras_per_aspect(self, context, directions, center_np,
                                    verts_world, cam_settings, total_px,
                                    center_for_aspect):
        """Create cameras with per-camera optimal aspect ratio and distance.
        Uses an iterative refinement: first computes aspect from the
        orthographic silhouette, then refines it using the actual perspective
        projection from the computed camera position."""
        center_vec = mathutils.Vector(center_np.tolist())
        prefix_map = {
            'hemisphere': 'Camera_sphere',
            'normal_weighted': 'Camera_auto',
            'pca_axes': 'Camera_pca',
            'greedy_coverage': 'Camera',
        }
        prefix = prefix_map.get(self.placement_mode, 'Camera')

        for i, d in enumerate(directions):
            d_np = np.array(d, dtype=float)
            d_unit = d_np / np.linalg.norm(d_np)

            # --- Pass 1: orthographic aspect as initial estimate ---
            align = _get_resolution_align(context)
            aspect = _compute_per_camera_aspect(d_unit, verts_world, center_for_aspect)
            res_x, res_y = _resolution_from_aspect(aspect, total_px, align=align)
            fov_x, fov_y = _get_fov(cam_settings, context, res_x, res_y)
            dist, aim_off = _compute_silhouette_distance(
                verts_world, center_np, d_np, fov_x, fov_y)

            # --- Pass 2: refine aspect from actual perspective camera pos ---
            aim_point_np = center_np + aim_off
            cam_pos_np = aim_point_np + d_unit * dist
            aspect = _perspective_aspect(verts_world, cam_pos_np, d_np)
            res_x, res_y = _resolution_from_aspect(aspect, total_px, align=align)
            fov_x, fov_y = _get_fov(cam_settings, context, res_x, res_y)
            dist, aim_off = _compute_silhouette_distance(
                verts_world, center_np, d_np, fov_x, fov_y)

            dir_vec = mathutils.Vector(d_np.tolist()).normalized()
            aim_point = center_vec + mathutils.Vector(aim_off.tolist())
            pos = aim_point + dir_vec * dist

            cam_data = bpy.data.cameras.new(name=f'{prefix}_{i}')
            cam_obj = bpy.data.objects.new(f'{prefix}_{i}', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.location = pos
            right, up_v, d_unit_cam = _camera_basis(d_np)
            cam_obj.rotation_euler = _rotation_from_basis(right, up_v, d_unit_cam)
            self._copy_camera_settings(cam_obj, cam_settings)

            # Store per-camera resolution
            _store_per_camera_resolution(cam_obj, res_x, res_y)
            _setup_square_camera_display(cam_obj, res_x, res_y)
            self._cameras.append(cam_obj)

        # Set scene to square resolution (max side length) for viewport display
        max_side = max(
            max(int(c.get('sg_res_x', 0)), int(c.get('sg_res_y', 0)))
            for c in self._cameras if 'sg_res_x' in c
        ) if self._cameras else max(context.scene.render.resolution_x,
                                     context.scene.render.resolution_y)
        if max_side > 0:
            context.scene.render.resolution_x = max_side
            context.scene.render.resolution_y = max_side

    @staticmethod
    def _copy_camera_settings(cam_obj, cam_settings):
        """Copy lens / sensor / clip settings from a reference camera data block."""
        cam_obj.data.type = cam_settings.type
        cam_obj.data.lens = cam_settings.lens
        cam_obj.data.sensor_width = cam_settings.sensor_width
        cam_obj.data.sensor_height = cam_settings.sensor_height
        cam_obj.data.clip_start = cam_settings.clip_start
        cam_obj.data.clip_end = cam_settings.clip_end

    def modal(self, context, event):
        # ── Occlusion computation phase ──────────────────────────────────
        if self._occ_phase:
            if event.type in {'ESC', 'RIGHTMOUSE'}:
                # Cancel occlusion computation
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
                self._occ_gen = None
                self._occ_phase = False
                context.area.header_text_set(None)
                if AddCameras._draw_handle:
                    bpy.types.SpaceView3D.draw_handler_remove(
                        AddCameras._draw_handle, 'WINDOW')
                    AddCameras._draw_handle = None
                # Clean up temp camera data stored in state
                state = self._occ_state
                if state and state.get('temp_cam_data'):
                    bpy.data.cameras.remove(state['temp_cam_data'])
                self._occ_state = None
                self.report({'WARNING'}, "Occlusion computation cancelled.")
                return {'CANCELLED'}

            if event.type == 'TIMER':
                try:
                    progress = next(self._occ_gen)
                    self._occ_progress = progress
                    context.area.header_text_set(
                        f"Computing occlusion visibility… "
                        f"{progress * 100:.0f}%   (ESC to cancel)")
                    # Force viewport redraw so the GPU overlay updates too
                    for area in context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                except StopIteration as done:
                    # Generator finished – retrieve result
                    context.area.header_text_set(None)

                    # Clean up occlusion timer
                    context.window_manager.event_timer_remove(self._timer)
                    self._timer = None
                    self._occ_gen = None
                    self._occ_phase = False

                    state = self._occ_state
                    result_type = state.get('result_type', 'greedy')

                    if result_type == 'greedy':
                        # Greedy occlusion: result is (directions, coverage)
                        directions, coverage = done.value
                        directions = [np.array(d) for d in directions]
                        occ_label = state['occ_label']
                        self.report({'INFO'},
                            f"Greedy coverage ({occ_label}): "
                            f"{len(directions)} cameras, "
                            f"{coverage * 100:.1f}% coverage")
                    elif result_type == 'filter_kmeans':
                        # Face filter for K-means: result is exterior mask
                        exterior = done.value
                        normals = state['normals'][exterior]
                        areas = state['areas'][exterior]
                        n_removed = int((~exterior).sum())
                        self.report({'INFO'},
                            f"Occlusion filter: removed {n_removed} "
                            f"interior faces, {len(normals)} remain")
                        k = min(state['num_cameras'], len(normals))
                        if k > 0 and len(normals) > 0:
                            cluster_dirs = _kmeans_on_sphere(normals, areas, k)
                            directions = [cluster_dirs[i]
                                          for i in range(len(cluster_dirs))]
                        else:
                            directions = []
                    elif result_type == 'filter_pca':
                        # Face filter for PCA: result is exterior mask
                        exterior = done.value
                        # PCA uses filtered verts (only exterior faces)
                        f_centers = state['centers'][exterior]
                        n_removed = int((~exterior).sum())
                        self.report({'INFO'},
                            f"Occlusion filter: removed {n_removed} "
                            f"interior faces, {len(f_centers)} remain")
                        if len(f_centers) >= 3:
                            axes = _compute_pca_axes(f_centers)
                        else:
                            axes = _compute_pca_axes(
                                state['verts_world'])
                        directions = []
                        for axis in axes:
                            directions.append(axis)
                            directions.append(-axis)
                        if state.get('exclude_bottom'):
                            threshold_z = -math.cos(
                                state['exclude_bottom_angle'])
                            directions = [
                                d for d in directions
                                if d[2] >= threshold_z]
                        directions = directions[
                            :min(state['num_cameras'],
                                 len(directions))]

                    elif result_type == 'vis_kmeans':
                        # Visibility-weighted K-means: weight by fraction
                        vis_count = done.value
                        n_cand = 200
                        vis_fraction = vis_count.astype(float) / n_cand
                        exterior = vis_count > 0
                        normals = state['normals'][exterior]
                        areas_base = state['areas'][exterior]
                        weighted_areas = areas_base * vis_fraction[exterior]
                        n_ext = int(exterior.sum())
                        n_removed = len(vis_count) - n_ext
                        self.report({'INFO'},
                            f"Visibility-weighted: {n_ext} visible faces, "
                            f"{n_removed} fully occluded removed")
                        k = min(state['num_cameras'], len(normals))
                        if k > 0 and len(normals) > 0:
                            cluster_dirs = _kmeans_on_sphere(
                                normals, weighted_areas, k)
                            directions = [cluster_dirs[i]
                                          for i in range(len(cluster_dirs))]
                        else:
                            directions = []

                    elif result_type == 'vis_interactive':
                        # Enter interactive preview phase
                        vis_count = done.value
                        self._vis_count = vis_count
                        self._vis_n_candidates = 200
                        self._vis_balance = 0.2
                        self._vis_preview_phase = True
                        self._vis_state = state
                        self._occ_state = None
                        self._update_vis_cameras(context)
                        return {'RUNNING_MODAL'}

                    # Angular dedup against existing cameras (filter paths)
                    ex_dirs = state.get('existing_dirs')
                    if ex_dirs and result_type in (
                            'filter_kmeans', 'filter_pca', 'vis_kmeans'):
                        directions = _filter_near_existing(
                            directions, ex_dirs)

                    # Finalize cameras (sort, aspect, create, prompts)
                    self._finalize_auto_cameras(
                        context, directions,
                        state['verts_world'], state['mesh_center'],
                        state['cam_settings'], state['temp_cam_data'])
                    self._occ_state = None

                    # Transition to fly-through review
                    if self.review_placement:
                        if not self._start_fly_review(
                                context, add_modal_handler=False):
                            return {'CANCELLED'}
                        # Stay in modal for the fly-through phase
                    else:
                        self._finish_without_review(context)
                        return {'FINISHED'}
                return {'RUNNING_MODAL'}

            # Let Blender process UI events during occlusion
            return {'RUNNING_MODAL'}

        # ── Interactive visibility preview phase ─────────────────────────
        if self._vis_preview_phase:
            if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
                self._cleanup_vis_preview(context)
                self.report({'WARNING'}, "Interactive visibility cancelled.")
                return {'CANCELLED'}

            changed = False
            if event.type == 'WHEELUPMOUSE':
                self._vis_balance = min(1.0, self._vis_balance + 0.05)
                changed = True
            elif event.type == 'WHEELDOWNMOUSE':
                self._vis_balance = max(0.0, self._vis_balance - 0.05)
                changed = True
            elif event.type == 'NUMPAD_PLUS' and event.value == 'PRESS':
                self._vis_balance = min(1.0, self._vis_balance + 0.05)
                changed = True
            elif event.type == 'NUMPAD_MINUS' and event.value == 'PRESS':
                self._vis_balance = max(0.0, self._vis_balance - 0.05)
                changed = True

            if changed:
                self._update_vis_cameras(context)
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type in {'RET', 'SPACE'} and event.value == 'PRESS':
                # Confirm: finalize with current directions
                self._vis_preview_phase = False
                context.area.header_text_set(None)

                # Delete preview cameras
                for cam in list(self._cameras):
                    cam_data = cam.data
                    bpy.data.objects.remove(cam, do_unlink=True)
                    if cam_data and not cam_data.users:
                        bpy.data.cameras.remove(cam_data)
                self._cameras.clear()

                state = self._vis_state
                self._finalize_auto_cameras(
                    context, self._vis_directions,
                    state['verts_world'], state['mesh_center'],
                    state['cam_settings'], state['temp_cam_data'])
                self._vis_state = None
                self._vis_count = None

                # Transition to fly-through or finish
                if self.review_placement:
                    if not self._start_fly_review(
                            context, add_modal_handler=False):
                        return {'CANCELLED'}
                else:
                    self._finish_without_review(context)
                    return {'FINISHED'}
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        # ── Fly-through review phase ─────────────────────────────────────
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if time.time() - self._last_time < 0.2:
                return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            fly_running = any(op.bl_idname == 'VIEW3D_OT_fly'
                              for w in context.window_manager.windows
                              for op in w.modal_operators)
            if not fly_running:
                self._camera_index += 1
                if self._camera_index >= len(self._cameras):
                    context.window_manager.event_timer_remove(self._timer)
                    if AddCameras._draw_handle:
                        bpy.types.SpaceView3D.draw_handler_remove(AddCameras._draw_handle, 'WINDOW')
                        AddCameras._draw_handle = None
                    # Restore to initial camera, or first placed camera for auto modes
                    if self._initial_camera and self._initial_camera.name in [o.name for o in context.scene.objects]:
                        context.scene.camera = self._initial_camera
                    elif self._cameras:
                        context.scene.camera = self._cameras[0]
                    # Enable the floating prompt labels now that review is over
                    _sg_restore_label_overlay()
                    self.report({'INFO'}, "Cameras added successfully.")
                    return {'FINISHED'}
                context.scene.camera = self._cameras[self._camera_index]
                self._last_time = time.time()
                bpy.ops.view3d.fly('INVOKE_DEFAULT')
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    