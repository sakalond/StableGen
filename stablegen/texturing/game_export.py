"""Game engine export operator (glTF/FBX)."""

import os
import bpy  # pylint: disable=import-error
from ..utils import get_dir_path, sg_modal_active

_ADDON_PKG = __package__.rsplit('.', 1)[0]


def _is_gltf_not_exported_helper(obj):
    for collection in getattr(obj, "users_collection", []):
        if "not_exported" in collection.name.lower():
            return True
    return False


def _is_visible_export_object(obj):
    if not obj or obj.hide_get() or _is_gltf_not_exported_helper(obj):
        return False
    try:
        return obj.visible_get(view_layer=bpy.context.view_layer)
    except TypeError:
        try:
            return obj.visible_get()
        except Exception:
            return True
    except Exception:
        return True


def _armatures_for_mesh(mesh):
    armatures = []
    parent = mesh.parent
    if parent and parent.type == 'ARMATURE':
        armatures.append(parent)

    for modifier in mesh.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object and modifier.object.type == 'ARMATURE':
            armatures.append(modifier.object)

    deduped = []
    seen = set()
    for armature in armatures:
        if armature.name not in seen:
            deduped.append(armature)
            seen.add(armature.name)
    return deduped


def _mesh_uses_armature(mesh, armature):
    if mesh.parent == armature:
        return True
    return any(
        modifier.type == 'ARMATURE' and modifier.object == armature
        for modifier in mesh.modifiers
    )


def _collect_game_export_objects(context, scope):
    if scope == 'ALL_MESH':
        meshes = [
            obj for obj in context.view_layer.objects
            if obj.type == 'MESH' and _is_visible_export_object(obj)
        ]
        selected_armatures = []
    else:
        selected = list(context.selected_objects)
        meshes = [obj for obj in selected if obj.type == 'MESH']
        selected_armatures = [
            obj for obj in selected
            if obj.type == 'ARMATURE' and _is_visible_export_object(obj)
        ]

    for armature in selected_armatures:
        for obj in context.view_layer.objects:
            if (
                obj.type == 'MESH'
                and _is_visible_export_object(obj)
                and _mesh_uses_armature(obj, armature)
                and obj not in meshes
            ):
                meshes.append(obj)

    armatures = []
    for mesh in meshes:
        for armature in _armatures_for_mesh(mesh):
            if _is_visible_export_object(armature) and armature not in armatures:
                armatures.append(armature)

    export_objects = []
    for obj in [*armatures, *meshes]:
        if obj not in export_objects:
            export_objects.append(obj)

    return meshes, armatures, export_objects


class ExportForGameEngine(bpy.types.Operator):
    """Export textured objects for game engines with PBR textures.

    Wraps Blender's glTF / FBX exporters with game-engine-optimal settings
    and auto-saves to the StableGen output directory."""
    bl_idname = "object.export_game_engine"
    bl_label = "Export for Game Engine"
    bl_options = {'REGISTER'}

    export_format: bpy.props.EnumProperty(
        name="Format",
        description="Export file format",
        items=[
            ('GLB', 'glTF Binary (.glb)',
             'Single-file binary glTF — best for web, Unity, Godot'),
            ('GLTF_SEPARATE', 'glTF + Textures (.gltf)',
             'Separate .gltf, .bin and texture files'),
            ('FBX', 'FBX (.fbx)',
             'Autodesk FBX with embedded textures — Unreal Engine'),
        ],
        default='GLB'
    ) # type: ignore

    export_scope: bpy.props.EnumProperty(
        name="Objects",
        description="Which objects to export",
        items=[
            ('SELECTED', 'Selected', 'Export selected meshes and detected rigs'),
            ('ALL_MESH', 'All Mesh', 'Export all visible meshes and detected rigs'),
        ],
        default='SELECTED'
    ) # type: ignore

    apply_transforms: bpy.props.BoolProperty(
        name="Apply Transforms",
        description="Apply location, rotation and scale before export. Skipped automatically for rigged exports",
        default=True
    ) # type: ignore

    export_animations: bpy.props.BoolProperty(
        name="Export Animations",
        description="Include animations in the export",
        default=False
    ) # type: ignore

    @classmethod
    def poll(cls, context):
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        if not os.path.exists(addon_prefs.output_dir):
            cls.poll_message_set("Output directory not set or does not exist (check addon preferences)")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_format")
        layout.prop(self, "export_scope")
        layout.prop(self, "apply_transforms")
        layout.prop(self, "export_animations")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        # ── Prepare selection ─────────────────────────────────────────
        original_selection = list(context.selected_objects)
        original_active = context.view_layer.objects.active

        meshes, armatures, export_objects = _collect_game_export_objects(
            context, self.export_scope)
        if not meshes:
            self.report({'ERROR'}, "No mesh objects to export.")
            return {'CANCELLED'}

        rigged_export = bool(armatures)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in export_objects:
            obj.select_set(True)
        context.view_layer.objects.active = meshes[0]

        # ── Output path ───────────────────────────────────────────────
        output_base = get_dir_path(context, "baked")
        os.makedirs(output_base, exist_ok=True)

        # Build filename from first object name
        base_name = meshes[0].name if meshes else "export"

        if self.export_format == 'FBX':
            ext = ".fbx"
        elif self.export_format == 'GLTF_SEPARATE':
            ext = ".gltf"
        else:
            ext = ".glb"
        filepath = os.path.join(output_base, f"{base_name}{ext}")

        # ── Apply transforms if requested ─────────────────────────────
        if self.apply_transforms and rigged_export:
            print(
                "[StableGen] Rigged export detected; skipping transform apply "
                "to preserve armature bind pose"
            )
        elif self.apply_transforms:
            for o in meshes:
                o.select_set(True)
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True)

        # ── Export ────────────────────────────────────────────────────
        try:
            if self.export_format in ('GLB', 'GLTF_SEPARATE'):
                bpy.ops.export_scene.gltf(
                    filepath=filepath,
                    export_format=self.export_format,
                    use_selection=True,
                    export_materials='EXPORT',
                    export_image_format='AUTO',
                    export_tangents=True,
                    export_yup=True,
                    export_apply=False,
                    export_animations=self.export_animations,
                )
            else:  # FBX
                bpy.ops.export_scene.fbx(
                    filepath=filepath,
                    use_selection=True,
                    object_types={'MESH', 'ARMATURE'},
                    apply_scale_options='FBX_SCALE_ALL',
                    path_mode='COPY',
                    embed_textures=True,
                    add_leaf_bones=False,
                    bake_anim=self.export_animations,
                    bake_anim_use_all_bones=True,
                    mesh_smooth_type='FACE',
                )

            if rigged_export:
                print(
                    f"[StableGen] Exported rigged asset with "
                    f"{len(meshes)} mesh object(s) and {len(armatures)} armature(s)"
                )
            self.report({'INFO'},
                        f"Exported to {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            import traceback
            traceback.print_exc()

        # ── Restore selection ─────────────────────────────────────────
        bpy.ops.object.select_all(action='DESELECT')
        for o in original_selection:
            try:
                o.select_set(True)
            except ReferenceError:
                pass
        if original_active:
            try:
                context.view_layer.objects.active = original_active
            except ReferenceError:
                pass

        return {'FINISHED'}
