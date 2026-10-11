"""Camera utility operators – switch viewport, clone, mirror, toggle labels."""

import bpy  # pylint: disable=import-error
import mathutils
from ..utils import sg_modal_active
from . import overlays as _overlays
from .overlays import (
    _sg_ensure_label_overlay, _sg_remove_label_overlay,
    _sg_ensure_crop_overlay, _sg_remove_crop_overlay,
    _setup_square_camera_display,
)
from .geometry import (
    _store_per_camera_resolution,
    _gather_target_meshes, _get_mesh_verts_world,
    _compute_per_camera_aspect, _resolution_from_aspect,
    _get_fov, _compute_silhouette_distance, _perspective_aspect,
    _camera_basis, _rotation_from_basis, _get_resolution_align
)
import numpy as np

class ApplyAutoAspect(bpy.types.Operator):
    """Apply automatic aspect ratio and framing to selected cameras based on object silhouette.
    
    If no camera is selected, applies to all cameras in the 'sg_cameras' collection."""
    bl_idname = "object.apply_auto_aspect"
    bl_label = "Apply Auto Aspect"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(o.type == 'CAMERA' for o in context.scene.objects) and not sg_modal_active(context)

    def execute(self, context):
        cameras = [o for o in context.selected_objects if o.type == 'CAMERA']
        if not cameras:
            col = bpy.data.collections.get("sg_cameras")
            if col:
                cameras = [o for o in col.objects if o.type == 'CAMERA']
            else:
                cameras = [o for o in context.scene.objects if o.type == 'CAMERA']
                
        if not cameras:
            self.report({'WARNING'}, "No cameras found")
            return {'CANCELLED'}
            
        target_meshes = _gather_target_meshes(context)
        if not target_meshes:
            self.report({'WARNING'}, "No target meshes found for silhouette calculation")
            return {'CANCELLED'}
            
        verts_world = _get_mesh_verts_world(target_meshes)
        center_for_aspect = verts_world.mean(axis=0)
        center_np = center_for_aspect
        center_vec = mathutils.Vector(center_np.tolist())
        
        render = context.scene.render
        total_px = render.resolution_x * render.resolution_y
        align = _get_resolution_align(context)
        
        updated_count = 0
        for cam_obj in cameras:
            cam_settings = cam_obj.data
            mat = cam_obj.matrix_world
            # Extract direction: 'd' in the placement geometry logic points FROM the object TO the camera.
            # Since the camera looks down its local -Z, the local +Z points back towards the camera.
            d_vec = mathutils.Vector(mat.col[2][:3])
            d_np = np.array([d_vec.x, d_vec.y, d_vec.z], dtype=float)
            norm = np.linalg.norm(d_np)
            if norm < 1e-6:
                continue
            d_unit = d_np / norm
            
            # --- Pass 1: orthographic aspect as initial estimate ---
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

            # Apply new transform (world-space — works even with parented cameras)
            right, up_v, d_unit_cam = _camera_basis(d_np)
            cam_obj.matrix_world = mathutils.Matrix.LocRotScale(
                pos,
                _rotation_from_basis(right, up_v, d_unit_cam).to_quaternion(),
                None,
            )

            # Store per-camera resolution
            _store_per_camera_resolution(cam_obj, res_x, res_y)
            _setup_square_camera_display(cam_obj, res_x, res_y)
            updated_count += 1
            
        # Set scene to max square resolution for viewport display
        all_sg_cams = [o for o in context.scene.objects if o.type == 'CAMERA' and 'sg_res_x' in o]
        if all_sg_cams:
            max_side = max(
                max(int(c.get('sg_res_x', 0)), int(c.get('sg_res_y', 0)))
                for c in all_sg_cams
            )
            if max_side > 0:
                context.scene.render.resolution_x = max_side
                context.scene.render.resolution_y = max_side
                
        self.report({'INFO'}, f"Applied Auto Aspect to {updated_count} camera(s)")
        return {'FINISHED'}

def switch_viewport_to_camera(context, camera):
    """Switches the first found 3D viewport to the specified camera's view."""
    if not camera:
        return
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            # Ensure we are in the right space and region
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
                    # Make sure the scene's active camera is set
                    context.scene.camera = camera
                    area.tag_redraw()
                    break
            break 

class CloneCamera(bpy.types.Operator):
    """Create a new camera at the active camera's position and enter fly mode to reposition it.
    
    If no camera exists, one is created from the current viewport.
    Useful for incrementally adding cameras one at a time."""
    bl_idname = "object.clone_camera"
    bl_label = "Clone Camera"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _camera = None
    _original_camera = None

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def execute(self, context):
        ref_cam = context.scene.camera
        self._original_camera = ref_cam

        if not ref_cam:
            # Create from viewport
            rv3d = context.region_data
            cam_data = bpy.data.cameras.new(name='Camera_clone')
            cam_obj = bpy.data.objects.new('Camera_clone', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.matrix_world = rv3d.view_matrix.inverted()
        else:
            # Clone from active camera
            cam_data = bpy.data.cameras.new(name=f'{ref_cam.name}_clone')
            cam_obj = bpy.data.objects.new(cam_data.name, cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.matrix_world = ref_cam.matrix_world.copy()
            cam_obj.data.type = ref_cam.data.type
            cam_obj.data.lens = ref_cam.data.lens
            cam_obj.data.sensor_width = ref_cam.data.sensor_width
            cam_obj.data.sensor_height = ref_cam.data.sensor_height
            cam_obj.data.clip_start = ref_cam.data.clip_start
            cam_obj.data.clip_end = ref_cam.data.clip_end

        self._camera = cam_obj
        context.scene.camera = cam_obj

        rv3d = context.region_data
        if rv3d.view_perspective != 'CAMERA':
            bpy.ops.view3d.view_camera()
        bpy.ops.view3d.view_center_camera()
        try:
            rv3d.view_camera_zoom = 1.0
        except Exception:
            pass

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        bpy.ops.view3d.fly('INVOKE_DEFAULT')
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            fly_running = any(op.bl_idname == 'VIEW3D_OT_fly'
                              for w in context.window_manager.windows
                              for op in w.modal_operators)
            if not fly_running:
                context.window_manager.event_timer_remove(self._timer)
                if self._original_camera:
                    context.scene.camera = self._original_camera
                self.report({'INFO'}, f"Camera cloned: {self._camera.name}")
                return {'FINISHED'}
            return {'PASS_THROUGH'}
        return {'PASS_THROUGH'}


class MirrorCamera(bpy.types.Operator):
    """Create a mirror of the active camera across a chosen axis through the object/scene center.
    
    The new camera is placed symmetrically on the opposite side and oriented to look at the center."""
    bl_idname = "object.mirror_camera"
    bl_label = "Mirror Camera"
    bl_options = {'REGISTER', 'UNDO'}

    mirror_axis: bpy.props.EnumProperty(
        name="Mirror Axis",
        description="Axis to mirror across",
        items=[
            ('X', "X Axis", "Mirror left / right"),
            ('Y', "Y Axis", "Mirror front / back"),
            ('Z', "Z Axis", "Mirror top / bottom"),
        ],
        default='X'
    ) # type: ignore

    @classmethod
    def poll(cls, context):
        return context.scene.camera is not None and not sg_modal_active(context)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        cam = context.scene.camera
        obj = context.object

        # Determine center point
        if obj and obj.type == 'MESH':
            center = obj.matrix_world.translation.copy()
        else:
            center = mathutils.Vector((0, 0, 0))

        # Mirror position
        pos = cam.location.copy()
        axis_idx = 'XYZ'.index(self.mirror_axis)
        delta = pos - center
        delta[axis_idx] = -delta[axis_idx]
        new_pos = center + delta

        # Create new camera
        cam_data = bpy.data.cameras.new(name=f'{cam.name}_mirror_{self.mirror_axis}')
        cam_obj = bpy.data.objects.new(cam_data.name, cam_data)
        context.collection.objects.link(cam_obj)

        # Copy settings
        cam_obj.data.type = cam.data.type
        cam_obj.data.lens = cam.data.lens
        cam_obj.data.sensor_width = cam.data.sensor_width
        cam_obj.data.sensor_height = cam.data.sensor_height
        cam_obj.data.clip_start = cam.data.clip_start
        cam_obj.data.clip_end = cam.data.clip_end

        cam_obj.location = new_pos
        direction = center - new_pos
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        self.report({'INFO'}, f"Mirrored camera created: {cam_obj.name}")
        return {'FINISHED'}


class ToggleCameraLabels(bpy.types.Operator):
    """Toggle numbered camera labels in the 3D viewport.

    Draws short camera names and their generation-order numbers.
    Full prompts remain in the camera prompt editor."""
    bl_idname = "object.toggle_camera_labels"
    bl_label = "Toggle Camera Labels"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return any(o.type == 'CAMERA' for o in context.scene.objects) and not sg_modal_active(context)

    def execute(self, context):
        if context.scene.sg_camera_display.labels != 'NONE':
            context.scene.sg_camera_display.labels = 'NONE'
            _sg_remove_label_overlay()
            self.report({'INFO'}, "Camera labels hidden")
        else:
            context.scene.sg_camera_display.labels = 'ALL'
            _sg_ensure_label_overlay()
            self.report({'INFO'}, "Camera labels visible")
        return {'FINISHED'}
