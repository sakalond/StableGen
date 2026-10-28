"""Viewport presentation helpers for imported SkinTokens armatures."""

import json

import bpy  # pylint: disable=import-error
from mathutils import Matrix  # pylint: disable=import-error


_AUTORIG_IMPORTED_DISPLAY_PROP = "sg_autorig_imported_bone_display"
_AUTORIG_IMPORTED_SHAPES_PROP = "sg_autorig_imported_custom_shapes"
AUTORIG_GLTF_BONE_HEURISTIC = "FORTUNE"
AUTORIG_GLTF_BONE_HEURISTICS = {"FORTUNE", "BLENDER", "TEMPERANCE"}


def autorig_gltf_import_kwargs(filepath, bone_heuristic=AUTORIG_GLTF_BONE_HEURISTIC):
    """Return glTF import settings suitable for SkinTokens joint rigs.

    SkinTokens exports glTF joints, which do not contain Blender-style bone
    tails.  Blender's default import heuristic points each synthetic tail down
    the joint's local +Y axis.  Fortune instead points one-child bones at the
    child joint while compensating the imported transforms so the skin remains
    in place.
    """
    if bone_heuristic not in AUTORIG_GLTF_BONE_HEURISTICS:
        bone_heuristic = AUTORIG_GLTF_BONE_HEURISTIC
    return {
        "filepath": filepath,
        "bone_heuristic": bone_heuristic,
        # Blender creates glTF_not_exported plus an Icosphere whenever bone
        # shapes are enabled, but only its legacy BLENDER heuristic actually
        # assigns that shape to pose bones.
        "disable_bone_shape": bone_heuristic != "BLENDER",
    }


def configure_gltf_not_exported_visibility(context=None):
    """Hide Blender's glTF helper collection using normal Outliner controls.

    Blender's importer globally disables this collection in viewports via
    ``Collection.hide_viewport``.  That is the monitor/restriction setting,
    which is hard to discover and override.  Keep the collection globally
    enabled, use each view layer's eye toggle instead, and use only the
    collection render toggle rather than burying duplicate flags on objects.
    """
    context = context or bpy.context
    scene = getattr(context, "scene", None) or bpy.context.scene
    stats = {
        "collections": 0,
        "layer_collections": 0,
        "object_flags_cleared": 0,
    }

    def iter_objects(collection):
        seen = set()

        def visit(current):
            for obj in current.objects:
                if obj.name in seen:
                    continue
                seen.add(obj.name)
                yield obj
            for child in current.children:
                yield from visit(child)

        yield from visit(collection)

    def matching_layer_collections(root, target):
        if root.collection == target:
            yield root
        for child in root.children:
            yield from matching_layer_collections(child, target)

    for collection in bpy.data.collections:
        if "not_exported" not in collection.name.lower():
            continue
        stats["collections"] += 1
        collection.hide_viewport = False
        collection.hide_render = True

        for obj in iter_objects(collection):
            if obj.hide_viewport or obj.hide_render:
                stats["object_flags_cleared"] += 1
            obj.hide_viewport = False
            obj.hide_render = False

        for view_layer in scene.view_layers:
            for layer_collection in matching_layer_collections(
                    view_layer.layer_collection, collection):
                layer_collection.exclude = False
                layer_collection.hide_viewport = True
                stats["layer_collections"] += 1

    view_layer = getattr(context, "view_layer", None)
    if view_layer:
        view_layer.update()
    return stats


def record_autorig_bone_heuristic(objects, bone_heuristic):
    """Record the glTF joint-direction heuristic on imported armatures."""
    bone_heuristic = autorig_gltf_import_kwargs(
        "", bone_heuristic)["bone_heuristic"]
    armatures = []
    for obj in objects:
        if obj.type == 'ARMATURE' and obj not in armatures:
            armatures.append(obj)
        if obj.type != 'MESH':
            continue
        if obj.parent and obj.parent.type == 'ARMATURE' and obj.parent not in armatures:
            armatures.append(obj.parent)
        for modifier in obj.modifiers:
            if (
                modifier.type == 'ARMATURE'
                and modifier.object
                and modifier.object not in armatures
            ):
                armatures.append(modifier.object)
    for armature in armatures:
        armature["sg_autorig_bone_heuristic"] = bone_heuristic
    return len(armatures)


def fortune_conversion_risks(armature):
    """Return reasons why rewriting an existing rig's rest axes is risky."""
    risks = []
    animation_data = armature.animation_data
    if animation_data:
        if animation_data.action is not None:
            risks.append("an active animation action")
        if animation_data.nla_tracks:
            risks.append("NLA animation tracks")
        if animation_data.drivers:
            risks.append("animation drivers")
    if any(pose_bone.constraints for pose_bone in armature.pose.bones):
        risks.append("pose-bone constraints")

    identity = Matrix.Identity(4)
    if any(
        max(
            abs(pose_bone.matrix_basis[row][column] - identity[row][column])
            for row in range(4)
            for column in range(4)
        ) > 1e-6
        for pose_bone in armature.pose.bones
    ):
        risks.append("a non-rest pose")
    return risks


def convert_armature_bones_to_fortune(armature):
    """Rebuild existing deform-bone axes using Blender's Fortune geometry.

    Joint heads and hierarchy are preserved.  Non-leaf tails point toward the
    centroid of their child joints and use the nearest child distance, matching
    Blender's glTF Fortune heuristic.  Leaf directions inherit their parent's
    direction correction.  The operation is intended for unanimated generated
    rigs; callers should inspect :func:`fortune_conversion_risks` first.
    """
    if not armature or armature.type != 'ARMATURE':
        raise TypeError("Expected an armature object")

    context = bpy.context
    active_object = getattr(context, "object", None)
    if active_object is None:
        active_object = context.view_layer.objects.active
    if active_object and active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    changed = 0
    one_child = 0
    branches = 0
    leaves = 0
    try:
        edit_bones = armature.data.edit_bones
        snapshots = {
            bone.name: {
                "head": bone.head.copy(),
                "tail": bone.tail.copy(),
                "matrix": bone.matrix.copy(),
                "parent": bone.parent.name if bone.parent else None,
                "children": [child.name for child in bone.children],
            }
            for bone in edit_bones
        }

        def bone_depth(name):
            depth = 0
            parent = snapshots[name]["parent"]
            while parent:
                depth += 1
                parent = snapshots[parent]["parent"]
            return depth

        corrections = {}
        for name in sorted(snapshots, key=bone_depth):
            snapshot = snapshots[name]
            head = snapshot["head"]
            old_vector = snapshot["tail"] - head
            if old_vector.length <= 1e-8:
                continue
            old_direction = old_vector.normalized()
            children = snapshot["children"]

            if children:
                offsets = [
                    snapshots[child]["head"] - head
                    for child in children
                ]
                offsets = [offset for offset in offsets if offset.length > 1e-8]
                if not offsets:
                    continue
                direction_sum = sum(offsets, offsets[0].copy() * 0.0)
                if direction_sum.length <= 1e-8:
                    continue
                new_direction = direction_sum.normalized()
                new_length = min(offset.length for offset in offsets)
                if len(offsets) == 1:
                    one_child += 1
                else:
                    branches += 1
            else:
                parent_name = snapshot["parent"]
                parent_correction = corrections.get(parent_name)
                if parent_correction is None:
                    continue
                new_direction = (parent_correction @ old_direction).normalized()
                new_length = old_vector.length
                leaves += 1

            correction = old_direction.rotation_difference(new_direction)
            corrections[name] = correction
            old_z = snapshot["matrix"].to_3x3().col[2].normalized()
            roll_reference = correction @ old_z

            edit_bone = edit_bones[name]
            edit_bone.tail = head + (new_direction * new_length)
            if (
                roll_reference.length > 1e-8
                and abs(roll_reference.normalized().dot(new_direction)) < 0.9999
            ):
                edit_bone.align_roll(roll_reference)
            changed += 1
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    apply_autorig_bone_display([armature], 'OCTAHEDRAL')
    armature["sg_autorig_bone_heuristic"] = "FORTUNE_CONVERTED"
    return {
        "changed": changed,
        "one_child": one_child,
        "branches": branches,
        "leaves": leaves,
    }


def apply_autorig_bone_display(objects, display_type='OCTAHEDRAL'):
    """Apply a selectable Blender bone display to imported SkinTokens rigs.

    The incoming GLB commonly uses Blender's Stick display, whose circular
    joints can look like icospheres. Non-imported modes clear pose-bone custom
    shapes as well, while recording enough information to restore the imported
    presentation later.
    """
    armatures = []
    for obj in objects:
        if obj.type == 'ARMATURE' and obj not in armatures:
            armatures.append(obj)
        if obj.type != 'MESH':
            continue
        if obj.parent and obj.parent.type == 'ARMATURE' and obj.parent not in armatures:
            armatures.append(obj.parent)
        for modifier in obj.modifiers:
            if (
                modifier.type == 'ARMATURE'
                and modifier.object
                and modifier.object not in armatures
            ):
                armatures.append(modifier.object)
    for armature in armatures:
        if _AUTORIG_IMPORTED_DISPLAY_PROP not in armature:
            armature[_AUTORIG_IMPORTED_DISPLAY_PROP] = armature.data.display_type
        if _AUTORIG_IMPORTED_SHAPES_PROP not in armature:
            armature[_AUTORIG_IMPORTED_SHAPES_PROP] = json.dumps({
                bone.name: bone.custom_shape.name
                for bone in armature.pose.bones
                if bone.custom_shape is not None
            }, sort_keys=True)

        if display_type == 'IMPORTED':
            imported_type = armature.get(
                _AUTORIG_IMPORTED_DISPLAY_PROP, armature.data.display_type)
            valid_types = {'OCTAHEDRAL', 'STICK', 'BBONE', 'ENVELOPE', 'WIRE'}
            if imported_type in valid_types:
                armature.data.display_type = imported_type
            try:
                imported_shapes = json.loads(
                    armature.get(_AUTORIG_IMPORTED_SHAPES_PROP, "{}"))
            except (TypeError, ValueError):
                imported_shapes = {}
            for bone in armature.pose.bones:
                shape_name = imported_shapes.get(bone.name)
                bone.custom_shape = bpy.data.objects.get(shape_name) if shape_name else None
        else:
            armature.data.display_type = display_type
            for bone in armature.pose.bones:
                bone.custom_shape = None

        # Generated rigs are primarily inspected inside the character mesh.
        armature.show_in_front = True
        armature["sg_autorig_bone_display"] = display_type

    return len(armatures)


def selected_armatures_for_display(context):
    """Resolve selected armatures, including those driving selected meshes."""
    armatures = []
    for obj in context.selected_objects:
        if obj.type == 'ARMATURE' and obj not in armatures:
            armatures.append(obj)
        if obj.type != 'MESH':
            continue
        if obj.parent and obj.parent.type == 'ARMATURE' and obj.parent not in armatures:
            armatures.append(obj.parent)
        for modifier in obj.modifiers:
            if (
                modifier.type == 'ARMATURE'
                and modifier.object
                and modifier.object not in armatures
            ):
                armatures.append(modifier.object)
    return armatures
