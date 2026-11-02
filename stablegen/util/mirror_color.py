import bpy
import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import mathutils
from mathutils import Vector

from ..utils import get_dir_path, get_file_path, get_generation_dirs, get_last_material_index, sg_modal_active
from ..texturing.projection import project_image
from ..texturing.view_plan import generation_cameras, order_cameras, order_cameras_from_manifest
from .color_match import color_match_single
from ..core.architectures import QWEN_ARCHITECTURES

# A reflection reverses handedness, so the mirrored camera (rebuilt right-handed,
# its up vector reflected with the scene) sees the reflected surface mirrored
# left to right, whichever plane is mirrored: every axis flips the image
# horizontally. Flipping vertically for Y turned its projection upside down.
MIRROR_AXES = (
    ("X", dict(mirror_x=True, mirror_y=False, mirror_z=False)),
    ("Y", dict(mirror_x=False, mirror_y=True, mirror_z=False)),
    ("Z", dict(mirror_x=False, mirror_y=False, mirror_z=True)),
)

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

        if sg_modal_active(context):
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

    def _mirror_image_file(self, src_path: str, dst_path: str):
        """Save the image at src_path mirrored left to right (see MIRROR_AXES) to dst_path."""
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with Image.open(src_path) as image:
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(dst_path)

    def _add_mirrored_camera(self, scene, orig_cam, axis_name, matrix_kwargs):
        mir_cam = orig_cam.copy()
        mir_cam.data = orig_cam.data.copy()
        # A view of its own: the copied view ID put it in the original camera's
        # slot of the run's view plan, so Reproject projected the original image
        # from the mirrored camera.
        if "sg_view_uuid" in mir_cam:
            del mir_cam["sg_view_uuid"]
        scene.collection.objects.link(mir_cam)
        mir_cam.matrix_world = self._mirror_camera_matrix(orig_cam, **matrix_kwargs)
        # '~' sorts after the usual names, so a run without a view plan also
        # numbers the mirror after its own cameras.
        mir_cam.name = f"~{orig_cam.name}_MIRROR_{axis_name}"
        return mir_cam

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

    def _get_run_cameras(self, context):
        """The cameras in the order the run numbered its files (generated_image<i>,
        render<i>_edgefeather), the order a generation or Reproject uses. Sorting
        by name put a mirrored camera between the run's cameras, shifting the
        ones after it onto other views' images and masks."""
        scene = context.scene
        names = ([item.name for item in scene.sg_camera_order]
                 if scene.sg_use_custom_camera_order else [])
        cameras = order_cameras(generation_cameras(scene), names)
        manifest = Path(get_generation_dirs(context)["revision"]) / "view_plan.json"
        if manifest.is_file():
            try:
                cameras = order_cameras_from_manifest(
                    cameras, json.loads(manifest.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as error:
                print(f"[MirrorReproject] Could not read the view plan: {error}")
        return cameras

    def _axes_to_mirror(self):
        flags = (self.mirror_axis_x, self.mirror_axis_y, self.mirror_axis_z)
        # Safety: default to X only if everything is off
        return [axis for axis, flag in zip(MIRROR_AXES, flags) if flag] or [MIRROR_AXES[0]]

    def _compute_latest_material_id(self, to_texture, include_edits=True):
        max_id = -1
        for obj in to_texture:
            mat_id = get_last_material_index(obj, include_edits=include_edits)
            if mat_id > max_id:
                max_id = mat_id
        return max_id

    def _do_refine_mirror(self, context, to_texture):
        """
        Special path for generation_method == 'local_edit' (or qwen_generation_method == 'local_edit').

        For each selected axis (X / Y / Z), we:
        - Duplicate & mirror the active camera around that axis.
        - Create a mirrored image for that camera in a NEW refine layer (material_id = base_id + 1),
          with its edit's edge-feather mask mirrored too.
        Then we call project_image once for that new layer, with the run's cameras
        first and the mirrored ones after them.
        """
        scene = context.scene

        # 1) Base material id (from existing stack)
        base_id = self._compute_latest_material_id(to_texture)
        if base_id < 0:
            self.report({'ERROR'}, "No StableGen materials found to mirror.")
            return {'CANCELLED'}

        # New refine layer id
        new_material_id = base_id + 1

        # 2) Original cameras & active camera index (before adding mirrored cameras)
        cameras_orig = self._get_run_cameras(context)
        if not cameras_orig:
            self.report({'ERROR'}, "No cameras found in the scene.")
            return {'CANCELLED'}

        orig_cam = scene.camera if scene.camera in cameras_orig else cameras_orig[0]
        src_cam_idx = cameras_orig.index(orig_cam)
        axes_to_mirror = self._axes_to_mirror()

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
        # A local edit gates each view with its edit's edge-feather mask, by camera index
        visibility_dir = get_dir_path(context, "inpaint")["visibility"]
        src_mask = os.path.join(visibility_dir, f"render{src_cam_idx}_edgefeather.png")

        created_cams = []

        # 4) For each selected axis, create a mirrored camera + mirrored image (and mask)
        for axis_name, matrix_kwargs in axes_to_mirror:
            mir_cam = self._add_mirrored_camera(scene, orig_cam, axis_name, matrix_kwargs)
            dst_cam_idx = len(cameras_orig) + len(created_cams)
            created_cams.append(mir_cam)

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
            self._mirror_image_file(src_path_for_mirror, dst_path)
            if os.path.exists(src_mask):
                self._mirror_image_file(
                    src_mask, os.path.join(visibility_dir, f"render{dst_cam_idx}_edgefeather.png"))

        # 5) Call project_image once as a new refine layer
        project_image(context, to_texture, new_material_id, cameras=cameras_orig + created_cams)

        # 6) Delete the temporary mirrored cameras
        for cam in created_cams:
            try:
                bpy.data.objects.remove(cam, do_unlink=True)
            except Exception:
                pass

        axis_labels = ", ".join(axis for axis, _kwargs in axes_to_mirror)
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
        from ..texturing.generator import ComfyUIGenerate

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

        # 1) The generation's material id: Reproject reads generated views, which
        #    Masked Edit layers do not have
        base_id = self._compute_latest_material_id(to_texture, include_edits=False)
        if base_id < 0:
            self.report({'ERROR'}, "No StableGen materials found to mirror.")
            return {'CANCELLED'}

        # 2) Get cameras in the order Reproject numbers them
        cameras = self._get_run_cameras(context)
        if not cameras:
            self.report({'ERROR'}, "No cameras found in the scene.")
            return {'CANCELLED'}

        # Use active scene camera if possible, otherwise first camera
        orig_cam = scene.camera if scene.camera in cameras else cameras[0]
        src_cam_idx = cameras.index(orig_cam)
        axes_to_mirror = self._axes_to_mirror()

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

        # 3) For each selected axis, create a mirrored camera + mirrored image.
        #    Mirrored cameras are new views, numbered after the run's own: a
        #    mirror numbered among them overwrote another view's image.
        for axis_name, matrix_kwargs in axes_to_mirror:
            mir_cam = self._add_mirrored_camera(scene, orig_cam, axis_name, matrix_kwargs)
            dst_cam_idx = len(cameras) + len(created_cam_names)
            created_cam_names.append(mir_cam.name)

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
            self._mirror_image_file(src_path, dst_path)
        run_order = self._get_run_cameras(context)
        if [c.name for c in run_order[len(cameras):]] != created_cam_names:
            print("[MirrorReproject] WARNING: Reproject numbers the mirrored cameras differently: "
                  f"{[c.name for c in run_order]}")

        # 4) Hand off to the existing Reproject operator once
        #    Bypass modal poll guard since we are already a modal operator
        from .. import utils as _sg_utils
        _sg_utils._sg_bypass_modal_check = True
        try:
            bpy.ops.object.stablegen_reproject('INVOKE_DEFAULT')
        except Exception as exc:
            self.report(
                {'ERROR'},
                f"Failed to start StableGen reprojection after mirroring: {exc}",
            )
            return {'CANCELLED'}
        finally:
            _sg_utils._sg_bypass_modal_check = False

        # 5) Schedule deletion of all temporary mirrored cameras
        for cam_name in created_cam_names:
            self._schedule_delete_camera_later(cam_name)

        axis_labels = ", ".join(axis for axis, _kwargs in axes_to_mirror)
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
        if scene.generation_method == 'local_edit' or (scene.model_architecture in QWEN_ARCHITECTURES and scene.qwen_generation_method == 'local_edit'):
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

def _unmatched_path(image_path):
    """Where the generated image is kept as generated while a color-matched copy takes its place."""
    return os.path.join(os.path.dirname(image_path), "unmatched", os.path.basename(image_path))


def restore_unmatched_image(image_path):
    """Put the generated image back after color matching was turned off, so a reprojection uses the original view."""
    original = _unmatched_path(image_path)
    if os.path.isfile(original):
        shutil.copyfile(original, image_path)
        os.remove(original)
        print(f"[StableGen] Restored unmatched view: {image_path}")


def _apply_color_match_to_file(image_path, ref_rgb, scene):
    """
    Color-match the generated image on disk to ref_rgb if enabled. The image as generated is kept in an
    'unmatched' folder beside it and every match starts from it, so matching never compounds and can be undone.
    """
    if not scene.view_blend_use_color_match:
        return image_path  # no-op

    try:
        original = _unmatched_path(image_path)
        if not os.path.isfile(original) and os.path.isfile(image_path):
            os.makedirs(os.path.dirname(original), exist_ok=True)
            shutil.copyfile(image_path, original)
        with Image.open(original if os.path.isfile(original) else image_path) as pil_img:
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
