"""Camera prompt items, ordering UI, and collection operators."""

import math
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import numpy as np
from ..utils import get_dir_path, sg_modal_active
from .operators import switch_viewport_to_camera
from .overlays import _sg_ensure_label_overlay

class CameraPromptItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        name="Camera Name",
        description="Name of the camera object"
    ) # type: ignore
    prompt: bpy.props.StringProperty(
        name="View Prompt",
        description="View-specific wording included with the shared generation prompt. Describe direction or details belonging to this camera, such as 'front view' or 'close-up on face'"
    ) # type: ignore

# ── Camera Generation Order ──────────────────────────────────────────────────

class CameraOrderItem(bpy.types.PropertyGroup):
    """One entry in the generation-order list. Stores the Blender camera name."""
    name: bpy.props.StringProperty(
        name="Camera Name",
        description="Name of the camera object"
    ) # type: ignore


class SG_UL_CameraOrderList(bpy.types.UIList):
    """UIList that shows cameras in their current generation order."""
    bl_idname = "SG_UL_CameraOrderList"

    def draw_item(self, _context, layout, _data, item, _icon,
                  _active_data, _active_propname, index):
        cam_obj = bpy.data.objects.get(item.name)
        if cam_obj is None:
            layout.label(text=f"{item.name} (missing)", icon='ERROR')
            return
        # Show index, camera icon, name, and prompt if any
        row = layout.row(align=True)
        row.label(text=f"{index + 1}.")
        row.label(text=item.name, icon='CAMERA_DATA')
        # Show prompt preview if one exists
        prompt_item = next(
            (p for p in _context.scene.camera_prompts if p.name == item.name), None)
        if prompt_item and prompt_item.prompt:
            sub = row.row()
            sub.scale_x = 1.5
            sub.label(text=prompt_item.prompt, icon='SHORTDISPLAY')


class SyncCameraOrder(bpy.types.Operator):
    """Rebuild the generation order list from all cameras in the scene (sorted alphabetically)"""
    bl_idname = "stablegen.sync_camera_order"
    bl_label = "Sync Camera Order"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        scene = context.scene
        cameras = sorted(
            [obj for obj in scene.objects if obj.type == 'CAMERA'],
            key=lambda x: x.name)
        scene.sg_camera_order.clear()
        for cam in cameras:
            item = scene.sg_camera_order.add()
            item.name = cam.name
        scene.sg_camera_order_index = 0
        self.report({'INFO'}, f"Synced {len(cameras)} cameras to generation order list.")
        return {'FINISHED'}


class MoveCameraOrder(bpy.types.Operator):
    """Move the selected camera up or down in the generation order"""
    bl_idname = "stablegen.move_camera_order"
    bl_label = "Move Camera"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[('UP', "Up", ""), ('DOWN', "Down", "")],
        name="Direction"
    ) # type: ignore

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        scene = context.scene
        idx = scene.sg_camera_order_index
        total = len(scene.sg_camera_order)
        if total < 2:
            return {'CANCELLED'}
        new_idx = idx - 1 if self.direction == 'UP' else idx + 1
        if new_idx < 0 or new_idx >= total:
            return {'CANCELLED'}
        scene.sg_camera_order.move(idx, new_idx)
        scene.sg_camera_order_index = new_idx
        return {'FINISHED'}


class ApplyCameraOrderPreset(bpy.types.Operator):
    """Sort the generation order list using a preset strategy.
    Cameras are first synced from the scene, then reordered."""
    bl_idname = "stablegen.apply_camera_order_preset"
    bl_label = "Apply Camera Order Preset"
    bl_options = {'REGISTER', 'UNDO'}

    strategy: bpy.props.EnumProperty(
        name="Strategy",
        items=[
            ('ALPHABETICAL', "Alphabetical (Original)",
             "Sort cameras alphabetically by name (original order for auto-generated cameras)"),
            ('FRONT_FIRST', "Front → Back → Sides",
             "Front-facing cameras first, then back, then sides. "
             "Prioritises the main view for sequential inpainting"),
            ('BACK_FIRST', "Back → Front → Sides",
             "Back-facing cameras first, then front, then sides"),
            ('ALTERNATING', "Alternating (Opposites)",
             "Alternate between opposing cameras (front, back, left, right…) "
             "for maximum context spread between consecutive views"),
            ('TOP_DOWN', "Top → Bottom",
             "Cameras sorted from highest elevation to lowest"),
            ('REVERSE', "Reverse Current",
             "Reverse the current generation order"),
        ],
        default='FRONT_FIRST'
    ) # type: ignore

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        scene = context.scene

        # Always sync from scene first (picks up new/removed cameras)
        cameras = sorted(
            [obj for obj in scene.objects if obj.type == 'CAMERA'],
            key=lambda x: x.name)

        if not cameras:
            self.report({'ERROR'}, "No cameras in the scene.")
            return {'CANCELLED'}

        if self.strategy == 'REVERSE':
            # Reverse whatever is currently in the list
            names = [item.name for item in scene.sg_camera_order]
            if not names:
                names = [c.name for c in cameras]
            names.reverse()
        elif self.strategy == 'ALPHABETICAL':
            names = [c.name for c in cameras]
        else:
            # Compute mesh centre for direction classification
            mesh_objects = [obj for obj in scene.objects
                           if obj.type == 'MESH' and not obj.hide_get()]
            if mesh_objects:
                centres = [obj.matrix_world @ (
                    sum((mathutils.Vector(v.co) for v in obj.data.vertices),
                        mathutils.Vector()) / max(len(obj.data.vertices), 1))
                    for obj in mesh_objects]
                mesh_center = sum(centres, mathutils.Vector()) / len(centres)
            else:
                mesh_center = mathutils.Vector((0, 0, 0))

            mc = np.array(mesh_center, dtype=float)

            # Use the first camera (alphabetically) as the reference
            # "front" direction — avoids assuming any world axis is front.
            ref_pos = np.array(cameras[0].location, dtype=float)
            ref_d = ref_pos - mc
            ref_d_xy = ref_d[:2]
            ref_xy_len = np.linalg.norm(ref_d_xy)
            if ref_xy_len < 1e-8:
                ref_d_xy = np.array([0.0, 1.0])
            else:
                ref_d_xy = ref_d_xy / ref_xy_len

            # Compute direction angles for each camera
            cam_data = []
            for cam in cameras:
                pos = np.array(cam.location, dtype=float)
                d = pos - mc
                norm = np.linalg.norm(d)
                if norm < 1e-8:
                    d = np.array([0, 1, 0], dtype=float)
                else:
                    d /= norm
                # Elevation (angle above XY plane)
                elev = math.degrees(math.asin(np.clip(d[2], -1, 1)))
                # Azimuth relative to the first camera's XY direction
                d_xy = d[:2]
                d_xy_len = np.linalg.norm(d_xy)
                if d_xy_len < 1e-8:
                    azimuth = 0.0
                else:
                    d_xy = d_xy / d_xy_len
                    cos_a = float(np.clip(np.dot(d_xy, ref_d_xy), -1, 1))
                    cross = float(ref_d_xy[0] * d_xy[1]
                                  - ref_d_xy[1] * d_xy[0])
                    azimuth = math.degrees(math.atan2(cross, cos_a))
                cam_data.append((cam.name, elev, azimuth))

            if self.strategy == 'FRONT_FIRST':
                # Priority: front (small |azimuth|), then back (large |azimuth|), then sides
                def _front_key(item):
                    _name, _elev, az = item
                    abs_az = abs(az)
                    if abs_az <= 45:
                        bucket = 0  # front
                    elif abs_az >= 135:
                        bucket = 1  # back
                    else:
                        bucket = 2  # sides
                    return (bucket, abs_az, -_elev)
                cam_data.sort(key=_front_key)

            elif self.strategy == 'BACK_FIRST':
                def _back_key(item):
                    _name, _elev, az = item
                    abs_az = abs(az)
                    if abs_az >= 135:
                        bucket = 0  # back
                    elif abs_az <= 45:
                        bucket = 1  # front
                    else:
                        bucket = 2  # sides
                    return (bucket, -abs_az, -_elev)
                cam_data.sort(key=_back_key)

            elif self.strategy == 'ALTERNATING':
                # Sort by azimuth, then pick from alternating ends
                cam_data.sort(key=lambda x: x[2])
                alternated = []
                left, right = 0, len(cam_data) - 1
                pick_left = True
                while left <= right:
                    if pick_left:
                        alternated.append(cam_data[left])
                        left += 1
                    else:
                        alternated.append(cam_data[right])
                        right -= 1
                    pick_left = not pick_left
                cam_data = alternated

            elif self.strategy == 'TOP_DOWN':
                cam_data.sort(key=lambda x: -x[1])  # highest elevation first

            names = [item[0] for item in cam_data]

        # Apply to collection
        scene.sg_camera_order.clear()
        for n in names:
            item = scene.sg_camera_order.add()
            item.name = n
        scene.sg_camera_order_index = 0
        self.report({'INFO'},
                    f"Applied '{self.strategy}' order to {len(names)} cameras.")
        return {'FINISHED'}


class CollectCameraPrompts(bpy.types.Operator):
    """Collect viewpoint description prompts for the selected cameras (or all cameras if none selected).
    
    - These prompts will be appended before the main prompt for each camera generation.
    - Applicable for separate, sequential and refine (also within grid mode) modes.
    - Select one or more cameras in the viewport to only prompt for those, or run with no selection to cycle through all cameras.
    - Examples: 'front view', 'close-up on face', 'from above'."""
    bl_idname = "object.collect_camera_prompts"
    bl_label = "Collect Camera View Prompts"
    bl_options = {'REGISTER', 'UNDO'}

    camera_prompt: bpy.props.StringProperty(
        name="View Prompt",
        description="View-specific generation wording for this camera, such as 'front view', 'close-up on face', or 'from above'. This dialog steps through selected cameras, or all cameras if none are selected",
        default=""
    ) # type: ignore

    # Internal state
    _cameras: list = []
    _camera_index: int = 0

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        if not any(obj.type == 'CAMERA' for obj in context.scene.objects):
            cls.poll_message_set("No cameras in the scene — use 'Add Cameras' first")
            return False
        return True

    def invoke(self, context, event):
        # Use selected cameras if any are selected, otherwise fall back to all
        selected_cams = sorted(
            [obj for obj in context.selected_objects if obj.type == 'CAMERA'],
            key=lambda x: x.name,
        )
        if selected_cams:
            self._cameras = selected_cams
        else:
            self._cameras = sorted(
                [obj for obj in context.scene.objects if obj.type == 'CAMERA'],
                key=lambda x: x.name,
            )

        if not self._cameras:
            self.report({'ERROR'}, "No cameras found in the scene.")
            return {'CANCELLED'}

        # Initialize state
        self._camera_index = 0

        # Set the first camera and pre-fill prompt if exists
        current_cam = self._cameras[self._camera_index]

        # Find existing prompt or set default
        existing_item = next((item for item in context.scene.camera_prompts if item.name == current_cam.name), None)
        self.camera_prompt = existing_item.prompt if existing_item else ""

        context.scene.camera = current_cam
        switch_viewport_to_camera(context, current_cam) # Switch viewport
        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        if self._camera_index < len(self._cameras):
            layout.label(text=f"Camera: {self._cameras[self._camera_index].name} ({self._camera_index + 1}/{len(self._cameras)})")
            layout.prop(self, "camera_prompt")

    def execute(self, context):
        cam_name = self._cameras[self._camera_index].name

        # Find existing item or add a new one
        prompt_item = next((item for item in context.scene.camera_prompts if item.name == cam_name), None)
        if not prompt_item:
            prompt_item = context.scene.camera_prompts.add()
            prompt_item.name = cam_name

        prompt_item.prompt = self.camera_prompt.strip() # Store trimmed prompt

        self._camera_index += 1
        if self._camera_index < len(self._cameras):
            next_cam = self._cameras[self._camera_index]
            # Pre-fill next prompt
            existing_item = next((item for item in context.scene.camera_prompts if item.name == next_cam.name), None)
            self.camera_prompt = existing_item.prompt if existing_item else ""
            # Ensure scene camera is set for next dialog and switch view
            context.scene.camera = next_cam
            switch_viewport_to_camera(context, next_cam) # Switch viewport
            return context.window_manager.invoke_props_dialog(self, width=400) # Show next dialog
        else:
            # Refresh the floating labels so edited prompts appear
            _sg_ensure_label_overlay()
            self.report({'INFO'}, f"Collected prompts for {len(self._cameras)} cameras.")
            return {'FINISHED'}

