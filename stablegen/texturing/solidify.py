"""General-purpose StableGen mesh solidification.

Unlike the 3D-print debug/export path, this operates on base mesh data and
preserves the rig and live projection-material inputs.
"""

import bpy  # pylint: disable=import-error

from ..utils import sg_modal_active
from .print_export import _make_solid_mesh_object


def _weighted_membership_count(obj):
    return sum(len(vertex.groups) for vertex in obj.data.vertices)


def _projection_attribute_names(mesh):
    return {
        attribute.name
        for attribute in mesh.attributes
        if (
            attribute.name.startswith("ProjectionUV_")
            or attribute.name.startswith("_SG_")
        )
    }


def _mesh_topology_counts(mesh):
    """Return boundary/non-manifold edge counts without changing mesh state."""
    face_counts = {
        tuple(sorted(edge.vertices[:])): 0
        for edge in mesh.edges
    }
    for polygon in mesh.polygons:
        for edge_key in polygon.edge_keys:
            key = tuple(sorted(edge_key))
            face_counts[key] = face_counts.get(key, 0) + 1
    counts = tuple(face_counts.values())
    return {
        "boundary": sum(count == 1 for count in counts),
        "non_manifold": sum(count > 2 for count in counts),
        "loose": sum(count == 0 for count in counts),
    }


def _restore_mesh_metadata(source, target):
    """Restore datablock metadata BMesh does not carry itself."""
    for material in source.materials:
        target.materials.append(material)

    active_uv = source.uv_layers.active.name if source.uv_layers.active else None
    render_uv = next(
        (
            layer.name for layer in source.uv_layers
            if getattr(layer, "active_render", False)
        ),
        None,
    )
    if active_uv and target.uv_layers.get(active_uv):
        target.uv_layers.active = target.uv_layers[active_uv]
    if render_uv and target.uv_layers.get(render_uv):
        target.uv_layers[render_uv].active_render = True


def _restore_vertex_group_names(obj, names):
    """Reconnect preserved deform indices to their original bone names."""
    existing = [group.name for group in obj.vertex_groups]
    if existing == names:
        return
    if existing:
        raise RuntimeError(
            "replacement mesh exposed an unexpected vertex-group table "
            f"{existing!r}"
        )
    for name in names:
        obj.vertex_groups.new(name=name)


def _validate_solid_result(
        obj, source, target, weighted_before, *, fill_gaps):
    """Reject a replacement that would damage rig or projection state."""
    if len(target.vertices) == 0 or len(target.polygons) == 0:
        raise RuntimeError("solidification produced an empty mesh")

    source_uvs = {layer.name for layer in source.uv_layers}
    target_uvs = {layer.name for layer in target.uv_layers}
    missing_uvs = sorted(source_uvs - target_uvs)
    if missing_uvs:
        raise RuntimeError(
            "solidification lost UV layer(s): " + ", ".join(missing_uvs)
        )

    missing_projection = sorted(
        _projection_attribute_names(source)
        - _projection_attribute_names(target)
    )
    if missing_projection:
        raise RuntimeError(
            "solidification lost projection attribute(s): "
            + ", ".join(missing_projection)
        )

    source_materials = list(source.materials)
    target_materials = list(target.materials)
    if source_materials != target_materials:
        raise RuntimeError("solidification changed the material slot layout")
    if any(
        polygon.material_index >= len(target_materials)
        for polygon in target.polygons
    ):
        raise RuntimeError("solidification produced an invalid face material")

    cap_attribute = target.attributes.get("_SG_SolidCap")
    if fill_gaps and cap_attribute is None:
        raise RuntimeError("solidification did not mark its generated caps")
    if cap_attribute is not None:
        if cap_attribute.domain != 'FACE':
            raise RuntimeError("_SG_SolidCap has the wrong attribute domain")
        cap_faces = [
            polygon
            for polygon, value in zip(target.polygons, cap_attribute.data)
            if bool(value.value)
        ]
        if any(polygon.area <= 1.0e-12 for polygon in cap_faces):
            raise RuntimeError("solidification produced a degenerate cap face")

    source_topology = _mesh_topology_counts(source)
    target_topology = _mesh_topology_counts(target)
    if (
        target_topology["non_manifold"]
        > source_topology["non_manifold"]
    ):
        raise RuntimeError(
            "solidification increased non-manifold edges "
            f"({source_topology['non_manifold']} -> "
            f"{target_topology['non_manifold']})"
        )
    if (
        fill_gaps
        and target_topology["boundary"] > source_topology["boundary"]
    ):
        raise RuntimeError(
            "solidification increased open boundary edges "
            f"({source_topology['boundary']} -> "
            f"{target_topology['boundary']})"
        )

    weighted_after = _weighted_membership_count(obj)
    if weighted_before and not weighted_after:
        raise RuntimeError("solidification removed all skinning weights")


class StableGenMakeSolid(bpy.types.Operator):
    """Remove hidden interior surfaces and close resulting mesh boundaries."""

    bl_idname = "object.stablegen_make_solid"
    bl_label = "Make Solid"
    bl_description = (
        "Make the active mesh solid while preserving its rig, skinning "
        "weights, UV/projection data, and live StableGen materials"
    )
    bl_options = {'REGISTER', 'UNDO'}

    fill_gaps: bpy.props.BoolProperty(
        name="Fill Gaps",
        description="Close boundary loops left after hidden geometry removal",
        default=True,
    )
    keep_largest_island: bpy.props.BoolProperty(
        name="Keep Largest Island",
        description=(
            "Discard smaller disconnected components before closing gaps; "
            "disable this for characters with intentional detached parts"
        ),
        default=False,
    )
    raycast_count: bpy.props.IntProperty(
        name="Raycast Directions",
        description=(
            "Number of radial directions used to distinguish exterior from "
            "interior surfaces"
        ),
        default=12,
        min=4,
        max=64,
    )

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Wait for the current StableGen operation to finish")
            return False
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            cls.poll_message_set("Select an active mesh object")
            return False
        return True

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.active_object
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        source = obj.data
        if source.shape_keys and len(source.shape_keys.key_blocks) > 1:
            self.report(
                {'ERROR'},
                "Make Solid cannot safely preserve shape keys because it "
                "changes topology. Remove or apply them first.",
            )
            return {'CANCELLED'}

        materials_before = list(source.materials)
        active_material_index = obj.active_material_index
        vertex_groups_before = [group.name for group in obj.vertex_groups]
        weighted_before = _weighted_membership_count(obj)
        modifiers_before = list(obj.modifiers)

        solid = None
        try:
            solid = _make_solid_mesh_object(
                obj,
                fill_gaps=self.fill_gaps,
                keep_largest_island=self.keep_largest_island,
                use_evaluated_mesh=False,
                raycast_count=self.raycast_count,
                weld_distance=0.0,
                robust_fill=True,
                smooth_caps=True,
            )
            _restore_mesh_metadata(source, solid)
            obj.data = solid
            _restore_vertex_group_names(obj, vertex_groups_before)
            obj.active_material_index = min(
                active_material_index,
                max(0, len(materials_before) - 1),
            )
            _validate_solid_result(
                obj,
                source,
                solid,
                weighted_before,
                fill_gaps=self.fill_gaps,
            )

            vertex_groups_after = [
                group.name for group in obj.vertex_groups
            ]
            if vertex_groups_after != vertex_groups_before:
                raise RuntimeError(
                    "solidification changed the vertex groups "
                    f"({vertex_groups_before!r} -> {vertex_groups_after!r})"
                )
            if list(obj.modifiers) != modifiers_before:
                raise RuntimeError("solidification changed the modifier stack")

            if source.users == 0:
                bpy.data.meshes.remove(source)
            obj.data.update()
            self.report(
                {'INFO'},
                f"Made {obj.name} solid while preserving rig and materials",
            )
            return {'FINISHED'}
        except Exception as error:
            if obj.data == solid:
                obj.data = source
            if solid and solid.name in bpy.data.meshes and solid.users == 0:
                bpy.data.meshes.remove(solid)
            self.report({'ERROR'}, f"Make Solid failed safely: {error}")
            return {'CANCELLED'}
