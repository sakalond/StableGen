import bpy
import os
import numpy as np
from PIL import Image
import mathutils
from mathutils import Vector

from ..utils import get_file_path, get_last_material_index
from ..project import project_image
from ..color_match import color_match_single

def _build_mirrored_camera_matrix(
    cam: bpy.types.Object,
    mirror_x: bool,
    mirror_y: bool,
    mirror_z: bool,
) -> mathutils.Matrix:
    """
    Mirror a camera across the selected global axes (X=0, Y=0, Z=0 planes),
    then rebuild a proper rotation so we don't end up with negative scale
    / flipped handedness.
    """
    m = cam.matrix_world.copy()
    loc = m.to_translation()
    basis = m.to_3x3()

    # Camera conventions: looks along local -Z, local +Y is "up"
    forward = -(basis @ Vector((0.0, 0.0, 1.0)))  # world-space view dir
    up      =  (basis @ Vector((0.0, 1.0, 0.0)))  # world-space up

    # Reflection factors per axis
    sx = -1.0 if mirror_x else 1.0
    sy = -1.0 if mirror_y else 1.0
    sz = -1.0 if mirror_z else 1.0

    # Apply reflection to position and direction vectors
    loc_m = Vector((sx * loc.x, sy * loc.y, sz * loc.z))
    f_m   = Vector((sx * forward.x, sy * forward.y, sz * forward.z))
    u_m   = Vector((sx * up.x,      sy * up.y,      sz * up.z))

    f = f_m.normalized()
    u0 = u_m.normalized()

    # Build an orthonormal basis: right, up, -forward
    r = f.cross(u0)
    if r.length < 1e-6:
        # Degenerate case: fall back to world Z as "up"
        r = f.cross(Vector((0.0, 0.0, 1.0)))
    r.normalize()
    u = r.cross(f).normalized()

    # Columns: X = right, Y = up, Z = -forward (camera looks along -Z)
    ori3 = mathutils.Matrix((r, u, -f)).transposed()
    ori4 = ori3.to_4x4()
    ori4.translation = loc_m
    return ori4


def mirror_camera_matrix_x(cam: bpy.types.Object) -> mathutils.Matrix:
    """
    Convenience wrapper: mirror a camera across the global X=0 plane.
    Uses the shared helper to avoid code duplication.
    """
    return _build_mirrored_camera_matrix(cam, mirror_x=True, mirror_y=False, mirror_z=False)


class MirrorReproject(bpy.types.Operator):
    """Duplicate & mirror the last projection camera and image.

    - In standard / non-refine workflows: mirrors the image and camera, then calls Reproject.
    - In refine+preserve workflows: mirrors as a *new refine layer* using project_image directly.
    """
    bl_idname = "object.stablegen_mirror_reproject"
    bl_label = "Mirror Last Projection"
    bl_options = {'REGISTER', 'UNDO'}

    mirror_axis_x: bpy.props.BoolProperty(name="X", default=True)
    mirror_axis_y: bpy.props.BoolProperty(name="Y", default=False)
    mirror_axis_z: bpy.props.BoolProperty(name="Z", default=False)

    _original_method = None
    _original_overwrite_material = None
    _timer = None

    @classmethod
    def poll(cls, context):
        # Same checks as Reproject
        scene = context.scene
        if scene.output_timestamp == "":
            return False

        if scene.generation_status == 'waiting':
            return False

        blocked_ids = {
            'OBJECT_OT_add_cameras',
            'OBJECT_OT_bake_textures',
            'OBJECT_OT_collect_camera_prompts',
            'OBJECT_OT_test_stable',
            'OBJECT_OT_stablegen_reproject',
            'OBJECT_OT_stablegen_regenerate',
            'OBJECT_OT_stablegen_mirror_reproject',
        }

        wm = context.window_manager
        for window in wm.windows:
            for op in window.modal_operators:
                if op.bl_idname in blocked_ids:
                    return False

        return True

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Mirror Axes:")
        row = layout.row()
        row.prop(self, "mirror_axis_x")
        row.prop(self, "mirror_axis_y")
        row.prop(self, "mirror_axis_z")

    def _mirror_image_file(
        self,
        src_path: str,
        dst_path: str,
        flip_x: bool = True,
        flip_y: bool = False,
    ):
        """
        Load an image from src_path, mirror it according to flip_x / flip_y, save to dst_path.

        flip_x: mirror horizontally (left/right)
        flip_y: mirror vertically   (top/bottom)
        """
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        img = bpy.data.images.load(src_path)
        try:
            w, h = img.size
            pixels = list(img.pixels[:])  # flat RGBA array
            mirrored = [0.0] * len(pixels)
            row_stride = w * 4  # RGBA per pixel

            # y in *destination* space
            for y in range(h):
                src_y = h - 1 - y if flip_y else y
                src_row_start = src_y * row_stride
                dst_row_start = y * row_stride

                # x in *destination* space
                for x in range(w):
                    src_x = w - 1 - x if flip_x else x
                    src_idx = src_row_start + src_x * 4
                    dst_idx = dst_row_start + x * 4
                    mirrored[dst_idx:dst_idx + 4] = pixels[src_idx:src_idx + 4]

            img.pixels[:] = mirrored
            img.filepath_raw = dst_path
            img.file_format = 'PNG'
            img.save()
        finally:
            bpy.data.images.remove(img)

    def _mirror_camera_matrix(
        self,
        cam: bpy.types.Object,
        mirror_x: bool,
        mirror_y: bool,
        mirror_z: bool,
    ) -> mathutils.Matrix:
        """
        Thin wrapper around _build_mirrored_camera_matrix so it can be used
        from methods (and swapped if needed later).
        """
        return _build_mirrored_camera_matrix(cam, mirror_x, mirror_y, mirror_z)

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
        Special path for generation_method == 'local_edit' (or qwen_generation_method == 'local_edit').

        For each selected axis (X / Y / Z), we:
        - Duplicate & mirror the active camera around that axis.
        - Create a mirrored image for that camera in a NEW refine layer (material_id = base_id + 1).
        Then we call project_image once for that new layer.
        """
        import shutil  # (optionally move this to the top of the file)

        scene = context.scene

        # 1) Base material id (from existing stack)
        base_id = self._compute_latest_material_id(to_texture)
        if base_id < 0:
            self.report({'ERROR'}, "No StableGen materials found to mirror.")
            return {'CANCELLED'}

        # New refine layer id
        new_material_id = base_id + 1

        # 2) Original cameras & active camera index (before adding mirrored cameras)
        cameras_orig = self._get_sorted_cameras(context)
        if not cameras_orig:
            self.report({'ERROR'}, "No cameras found in the scene.")
            return {'CANCELLED'}

        orig_cam = scene.camera if scene.camera in cameras_orig else cameras_orig[0]
        src_cam_idx = cameras_orig.index(orig_cam)

        # --- Mirror axes from Operator properties ---
        axis_configs = [
            (
                "X",
                self.mirror_axis_x,
                dict(mirror_x=True,  mirror_y=False, mirror_z=False),
                (True, False),   # flip_x, flip_y
            ),
            (
                "Y",
                self.mirror_axis_y,
                dict(mirror_x=False, mirror_y=True,  mirror_z=False),
                (False, True),
            ),
            (
                "Z",
                self.mirror_axis_z,
                dict(mirror_x=False, mirror_y=False, mirror_z=True),
                (True, True),
            ),
        ]

        axes_to_mirror = [cfg for cfg in axis_configs if cfg[1]]
        if not axes_to_mirror:
            # Safety: default to X only if everything is off
            axes_to_mirror = [axis_configs[0]]

        # 3) Copy base layer images -> new refine layer for ALL original cameras
        for cam_idx, _cam_obj in enumerate(cameras_orig):
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
                print(
                    f"[MirrorReproject] No base image for cam {cam_idx} at {src_path}, "
                    f"skipping copy."
                )
                continue

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            print(
                f"[MirrorReproject] Copied base image for cam {cam_idx}\n"
                f"  {src_path}\n-> {dst_path}"
            )

        # We'll always mirror from the original camera's base image
        src_path_for_mirror = get_file_path(
            context,
            "generated",
            camera_id=src_cam_idx,
            material_id=base_id,
        )
        if not os.path.exists(src_path_for_mirror):
            self.report(
                {'ERROR'},
                (
                    f"No generated image found for active camera index {src_cam_idx} at:\n"
                    f"{src_path_for_mirror}"
                ),
            )
            return {'CANCELLED'}

        created_cams = []

        # 4) For each selected axis, create a mirrored camera + mirrored image
        for axis_name, _flag, matrix_kwargs, (flip_x, flip_y) in axes_to_mirror:
            # Duplicate & mirror from the original camera
            mir_cam = orig_cam.copy()
            mir_cam.data = orig_cam.data.copy()
            scene.collection.objects.link(mir_cam)

            mir_cam.matrix_world = self._mirror_camera_matrix(orig_cam, **matrix_kwargs)
            mir_cam.name = f"{orig_cam.name}_MIRROR_{axis_name}"
            created_cams.append(mir_cam)

            # After linking, get its index in the updated camera list
            cameras_all = self._get_sorted_cameras(context)
            dst_cam_idx = cameras_all.index(mir_cam)

            dst_path = get_file_path(
                context,
                "generated",
                camera_id=dst_cam_idx,
                material_id=new_material_id,
            )
            print(
                f"[MirrorReproject] Refine mirror ({axis_name})\n"
                f"  {src_path_for_mirror}\n-> {dst_path}"
            )

            self._mirror_image_file(
                src_path_for_mirror,
                dst_path,
                flip_x=flip_x,
                flip_y=flip_y,
            )

        # 5) Call project_image once as a new refine layer
        project_image(context, to_texture, new_material_id)

        # 6) Delete the temporary mirrored cameras
        for cam in created_cams:
            try:
                bpy.data.objects.remove(cam, do_unlink=True)
            except Exception:
                pass

        axis_labels = ", ".join(cfg[0] for cfg in axes_to_mirror)
        self.report(
            {'INFO'},
            (
                f"Refine mirror applied as new layer (material id {new_material_id}) "
                f"with {len(created_cams)} mirrored camera(s) ({axis_labels}). "
                f"Temporary cameras removed."
            ),
        )
        return {'FINISHED'}

    def _schedule_delete_camera_later(self, cam_name: str):
        """Delete the temporary camera after StableGen reprojection finishes."""
        import bpy
        from ..generator import ComfyUIGenerate

        def _delete():
            # Wait until the generate/reproject operator is done
            if ComfyUIGenerate._is_running:
                return 0.5  # check again in 0.5 seconds

            cam = bpy.data.objects.get(cam_name)
            if cam is not None:
                try:
                    bpy.data.objects.remove(cam, do_unlink=True)
                    print(f"[MirrorReproject] Deleted temporary camera '{cam_name}'")
                except Exception as exc:
                    print(f"[MirrorReproject] Failed to delete camera '{cam_name}': {exc}")
            return None  # stop the timer

        bpy.app.timers.register(_delete, first_interval=1.0)

    def _do_standard_mirror(self, context, to_texture):
        """
        Standard / non-refine workflow:
        - For each selected axis (X / Y / Z), duplicate & mirror the active camera
        around that single axis.
        - For each mirrored camera, mirror the image on disk for the same material id.
        - Then let the existing Reproject operator do its thing once.
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

        # --- Mirror axes from Operator properties ---
        axis_configs = [
            (
                "X",
                self.mirror_axis_x,
                dict(mirror_x=True,  mirror_y=False, mirror_z=False),
                (True, False),   # flip_x, flip_y
            ),
            (
                "Y",
                self.mirror_axis_y,
                dict(mirror_x=False, mirror_y=True,  mirror_z=False),
                (False, True),
            ),
            (
                "Z",
                self.mirror_axis_z,
                dict(mirror_x=False, mirror_y=False, mirror_z=True),
                (True, True),
            ),
        ]

        axes_to_mirror = [cfg for cfg in axis_configs if cfg[1]]

        # Safety: if user turned everything off, default back to X-only
        if not axes_to_mirror:
            axes_to_mirror = [axis_configs[0]]  # X

        # Source image for the active camera
        src_path = get_file_path(
            context,
            "generated",
            camera_id=src_cam_idx,
            material_id=base_id,
        )
        if not os.path.exists(src_path):
            self.report(
                {'ERROR'},
                (
                    f"No generated image found for active camera index {src_cam_idx} at:\n"
                    f"{src_path}"
                ),
            )
            return {'CANCELLED'}

        created_cam_names = []

        # 3) For each selected axis, create a mirrored camera + mirrored image
        for axis_name, _flag, matrix_kwargs, (flip_x, flip_y) in axes_to_mirror:
            # Duplicate & mirror from the original camera each time
            mir_cam = orig_cam.copy()
            mir_cam.data = orig_cam.data.copy()
            scene.collection.objects.link(mir_cam)

            mir_cam.matrix_world = self._mirror_camera_matrix(orig_cam, **matrix_kwargs)
            mir_cam.name = f"{orig_cam.name}_MIRROR_{axis_name}"
            created_cam_names.append(mir_cam.name)

            # Rebuild sorted list so we know this mirrored camera's index
            cameras = self._get_sorted_cameras(context)
            dst_cam_idx = cameras.index(mir_cam)

            dst_path = get_file_path(
                context,
                "generated",
                camera_id=dst_cam_idx,
                material_id=base_id,
            )
            print(
                f"[MirrorReproject] Standard mirror ({axis_name})\n"
                f"  {src_path}\n-> {dst_path}"
            )

            # Flip image for this axis
            self._mirror_image_file(src_path, dst_path, flip_x=flip_x, flip_y=flip_y)

        # 4) Hand off to the existing Reproject operator once
        try:
            bpy.ops.object.stablegen_reproject('INVOKE_DEFAULT')
        except Exception as exc:
            self.report(
                {'ERROR'},
                f"Failed to start StableGen reprojection after mirroring: {exc}",
            )
            return {'CANCELLED'}

        # 5) Schedule deletion of all temporary mirrored cameras
        for cam_name in created_cam_names:
            self._schedule_delete_camera_later(cam_name)

        axis_labels = ", ".join(cfg[0] for cfg in axes_to_mirror)
        self.report(
            {'INFO'},
            (
                f"Standard mirror: created {len(created_cam_names)} mirrored camera(s) "
                f"({axis_labels}), mirrored images, and started reprojection "
                f"(temporary cameras will be removed afterwards)."
            ),
        )
        return {'FINISHED'}

    def execute(self, context):
        scene = context.scene
        to_texture = self._get_to_texture(context)

        if not to_texture:
            self.report({'ERROR'}, "No mesh objects found for texturing.")
            return {'CANCELLED'}

        # Branch: local_edit vs everything else
        if scene.generation_method == 'local_edit' or (scene.model_architecture.startswith('qwen') and scene.qwen_generation_method == 'local_edit'):
            return self._do_refine_mirror(context, to_texture)
        else:
            return self._do_standard_mirror(context, to_texture)

def _get_viewport_ref_np(obj):
    """
    Try to grab the current viewport/base color texture for this object as [H, W, 3] float32 [0,1].
    Returns None if no suitable image is found.
    """
    mat = getattr(obj, "active_material", None)
    if not mat or not mat.use_nodes:
        return None

    nt = mat.node_tree
    if not nt:
        return None

    img = None
    # Try the first image texture in the node tree that is plugged into Base Color
    for node in nt.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None:
            img = node.image
            break

    if img is None:
        return None

    # Ensure the image has pixels loaded
    if not img.has_data:
        img.pixels[0]  # force load

    w, h = img.size
    # Blender stores pixels as linear RGBA floats in [0,1]
    buf = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    rgb_lin = buf[..., :3]

    # Convert from linear to sRGB so it matches the PNGs we read via Pillow
    x = np.clip(rgb_lin, 0.0, 1.0)
    a = 0.055
    rgb_srgb = np.where(
        x <= 0.0031308,
        12.92 * x,
        (1.0 + a) * np.power(x, 1.0 / 2.4) - a,
    )

    return rgb_srgb

def _apply_color_match_to_file(image_path, ref_rgb, scene):
    """
    Load image from disk, color-match to ref_rgb if enabled, and overwrite on disk.
    """
    if not scene.view_blend_use_color_match:
        return image_path  # no-op

    try:
        with Image.open(image_path) as pil_img:
            pil_img = pil_img.convert("RGBA")
            w, h = pil_img.size
            tgt_np = np.array(pil_img, dtype=np.float32) / 255.0  # [H, W, 4]

        # Resize ref_rgb to match target resolution if needed
        if ref_rgb.shape[:2] != (h, w):
            ref_uint8 = np.clip(ref_rgb * 255.0, 0, 255).astype("uint8")
            ref_img = Image.fromarray(ref_uint8, mode="RGB")
            ref_img = ref_img.resize((w, h), Image.BILINEAR)
            ref_resized = np.array(ref_img, dtype=np.float32) / 255.0
        else:
            ref_resized = ref_rgb

        # Build a fake alpha channel of 1s so ref has 4 channels
        ref_rgba = np.concatenate(
            [ref_resized, np.ones((*ref_resized.shape[:2], 1), dtype=np.float32)],
            axis=-1,
        )

        matched_np = color_match_single(
            ref=ref_rgba,
            target=tgt_np,
            method=scene.view_blend_color_match_method,
            strength=scene.view_blend_color_match_strength,
        )

        matched_np = np.clip(matched_np * 255.0, 0, 255).astype("uint8")
        matched_img = Image.fromarray(matched_np, mode="RGBA")
        matched_img.save(image_path)
        print(f"[StableGen] Color matched: {image_path}")
        return image_path

    except Exception as e:
        print(f"[StableGen] Color match failed for {image_path}: {e}")
        return image_path
