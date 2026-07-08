"""Humanoid animation retargeting for generated SkinTokens rigs.

This module intentionally treats generated armatures as arbitrary skeletons:
bone names may be generic (``bone_0``), so the first step is to infer a
semantic profile from the rest pose and store it on the armature.
"""

import json
import os
import traceback
from datetime import datetime

import blf  # pylint: disable=import-error
import bpy  # pylint: disable=import-error
import gpu  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
from bpy_extras import view3d_utils  # pylint: disable=import-error
from gpu_extras.batch import batch_for_shader  # pylint: disable=import-error

from ..utils import sg_modal_active

_PROFILE_PROP = "sg_retarget_profile"
_REFERENCE_POSE_PROP = "sg_retarget_reference_pose"
_PROFILE_VERSION = 12
_REFERENCE_POSE_VERSION = 1
_CONSTRAINT_PREFIX = "SG_Retarget_"
_BONE_ENUM_ITEMS_CACHE = {}
_rig_mapping_draw_handle = None
_rig_mapping_label_draw_handle = None
_rig_mapping_armature_name = None

_CHAIN_COLORS = {
    "spine": (0.45, 0.75, 1.0, 0.95),
    "left_arm": (0.2, 0.9, 0.45, 0.95),
    "right_arm": (1.0, 0.45, 0.25, 0.95),
    "left_leg": (0.2, 0.95, 0.95, 0.95),
    "right_leg": (1.0, 0.75, 0.2, 0.95),
}

_CHAIN_LABELS = {
    "spine": "Spine",
    "left_arm": "L Arm",
    "right_arm": "R Arm",
    "left_leg": "L Leg",
    "right_leg": "R Leg",
}

_BASIS_COLORS = {
    "forward": (1.0, 0.25, 0.95, 0.98),
    "left": (0.15, 0.55, 1.0, 0.98),
}

_BASIS_LABELS = {
    "forward": "Forward",
    "left": "Left",
}


def _norm_name(name):
    value = name.lower()
    for prefix in ("mixamorig:", "mixamo:", "manny:", "quinn:"):
        value = value.replace(prefix, "")
    for char in ("_", "-", ".", " ", ":"):
        value = value.replace(char, "")
    return value


def _named_bone(armature, candidates):
    bones_by_norm = {_norm_name(bone.name): bone for bone in armature.data.bones}
    for candidate in candidates:
        bone = bones_by_norm.get(_norm_name(candidate))
        if bone:
            return bone
    for bone in armature.data.bones:
        normalized = _norm_name(bone.name)
        if any(_norm_name(candidate) in normalized for candidate in candidates):
            return bone
    return None


def _existing_names(*bones):
    return [bone.name for bone in bones if bone]


def _standard_name_profile(armature):
    """Return a profile for common animation skeletons when names are useful."""
    hips = _named_bone(armature, ("hips", "pelvis", "root"))
    head = _named_bone(armature, ("head", "head01"))
    spine = [
        _named_bone(armature, ("spine", "spine01")),
        _named_bone(armature, ("spine1", "spine02")),
        _named_bone(armature, ("spine2", "spine03")),
        _named_bone(armature, ("chest", "spine04")),
        _named_bone(armature, ("neck", "neck01")),
        head,
    ]

    left_arm = _existing_names(
        _named_bone(armature, ("leftshoulder", "claviclel", "clavicle_l")),
        _named_bone(armature, ("leftarm", "upperarml", "upperarm_l")),
        _named_bone(armature, ("leftforearm", "lowerarml", "lowerarm_l")),
        _named_bone(armature, ("lefthand", "handl", "hand_l")),
    )
    right_arm = _existing_names(
        _named_bone(armature, ("rightshoulder", "clavicler", "clavicle_r")),
        _named_bone(armature, ("rightarm", "upperarmr", "upperarm_r")),
        _named_bone(armature, ("rightforearm", "lowerarmr", "lowerarm_r")),
        _named_bone(armature, ("righthand", "handr", "hand_r")),
    )
    left_leg = _existing_names(
        _named_bone(armature, ("leftupleg", "thighl", "thigh_l")),
        _named_bone(armature, ("leftleg", "calfl", "calf_l")),
        _named_bone(armature, ("leftfoot", "footl", "foot_l")),
        _named_bone(armature, ("lefttoebase", "balll", "ball_l", "toe_l")),
    )
    right_leg = _existing_names(
        _named_bone(armature, ("rightupleg", "thighr", "thigh_r")),
        _named_bone(armature, ("rightleg", "calfr", "calf_r")),
        _named_bone(armature, ("rightfoot", "footr", "foot_r")),
        _named_bone(armature, ("righttoebase", "ballr", "ball_r", "toe_r")),
    )

    chains = {
        "spine": _existing_names(hips, *spine),
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_leg": left_leg,
        "right_leg": right_leg,
    }
    useful_chains = sum(1 for chain in chains.values() if len(chain) >= 2)
    if not hips or useful_chains < 3:
        return None

    return {
        "version": _PROFILE_VERSION,
        "method": "standard_names",
        "armature": armature.name,
        "pelvis": hips.name,
        "head": head.name if head else chains["spine"][-1],
        "chains": chains,
    }


def _largest_root(armature):
    roots = [bone for bone in armature.data.bones if bone.parent is None]
    if not roots:
        return None

    def count_descendants(bone):
        return 1 + sum(count_descendants(child) for child in bone.children)

    return max(roots, key=count_descendants)


def _path_from_root(bone):
    path = []
    cur = bone
    while cur:
        path.append(cur)
        cur = cur.parent
    path.reverse()
    return path


def _limb_chain_for_leaf(leaf):
    path = _path_from_root(leaf)
    if len(path) <= 1:
        return [leaf.name]

    start = 0
    for idx, bone in enumerate(path[:-1]):
        if len(bone.children) > 1:
            start = idx + 1
    return [bone.name for bone in path[start:]]


def _terminal_limb_bone(leaf):
    path = _path_from_root(leaf)
    if len(path) < 4:
        return leaf

    last_index = len(path) - 1
    for idx in range(last_index - 1, 1, -1):
        bone = path[idx]
        if len(bone.children) < 2:
            continue

        remaining_depth = last_index - idx
        path_fraction = idx / max(last_index, 1)
        if remaining_depth > 4 or path_fraction < 0.55:
            continue

        # Finger/toe fans create many short distal leaves.  For humanoid
        # retargeting, the semantic limb endpoint should be the palm/foot base,
        # not whichever digit happened to be farthest from the torso.
        if len(bone.children) >= 3 or path_fraction >= 0.7:
            return bone

    return leaf


def _chain_to_leaf(leaf):
    return [bone.name for bone in _path_from_root(leaf)]


def _side_from_x(bone):
    # Fallback only.  Generated rigs are normally assigned by foot-derived
    # forward/up axes so arbitrary TRELLIS orientations do not matter.
    return "left" if bone.tail_local.x > 0.0 else "right"


def _leaf_limb_features(leaf):
    terminal = _terminal_limb_bone(leaf)
    chain_names = _limb_chain_for_leaf(terminal)
    chain_bones = []
    cur = terminal
    by_name = {}
    while cur:
        by_name[cur.name] = cur
        cur = cur.parent
    for name in chain_names:
        bone = by_name.get(name)
        if bone:
            chain_bones.append(bone)

    attach = chain_bones[0] if chain_bones else leaf
    return {
        "leaf": leaf,
        "terminal": terminal,
        "chain": chain_names,
        "attach_z": attach.head_local.z,
        "end_z": terminal.tail_local.z,
        "abs_x": abs(terminal.tail_local.x),
    }


def _dedupe_limb_features(features):
    by_terminal = {}
    for item in features:
        terminal = item.get("terminal") or item.get("leaf")
        if not terminal:
            continue
        existing = by_terminal.get(terminal.name)
        if not existing or item["abs_x"] > existing["abs_x"]:
            by_terminal[terminal.name] = item
    return list(by_terminal.values())


def _feature_endpoint(item):
    terminal = item.get("terminal") or item.get("leaf")
    if not terminal:
        return None
    return terminal.tail_local.copy()


def _world_up():
    return mathutils.Vector((0.0, 0.0, 1.0))


def _profile_uses_world_up(profile):
    return (profile or {}).get("method") != "standard_names"


def _horizontal_projection(vector, up):
    if vector is None:
        return None
    projected = vector - (up * vector.dot(up))
    if projected.length < 1e-6:
        return None
    return projected


def _snap_to_world_cardinal_forward(vector):
    if vector is None:
        return None
    flat = mathutils.Vector((vector.x, vector.y, 0.0))
    if flat.length < 1e-6:
        return None
    flat.normalize()
    candidates = (
        mathutils.Vector((1.0, 0.0, 0.0)),
        mathutils.Vector((-1.0, 0.0, 0.0)),
        mathutils.Vector((0.0, 1.0, 0.0)),
        mathutils.Vector((0.0, -1.0, 0.0)),
    )
    return max(candidates, key=lambda axis: flat.dot(axis)).copy()


def _distal_chain_front_hint(armature, chain, up):
    for bone_name in reversed(chain[1:]):
        bone = armature.data.bones.get(bone_name)
        if not bone:
            continue
        direction = bone.tail_local - bone.head_local
        if direction.length < 1e-6:
            continue
        projected = _horizontal_projection(direction, up)
        if projected is None:
            continue
        horizontal = projected.length / direction.length
        vertical = abs(direction.normalized().dot(up))
        if horizontal < 0.5 or vertical > 0.75 or projected.length < 1e-6:
            continue
        return projected.normalized()
    return None


def _front_hint_from_chain_names(armature, chain_names_list, up, origin=None, snap_cardinal=False):
    best = None
    distal_hints = []
    fallback_hints = []
    for chain in chain_names_list:
        if len(chain) < 2:
            continue

        distal_hint = _distal_chain_front_hint(armature, chain, up)
        if distal_hint is not None:
            distal_hints.append(distal_hint)

        trimmed_chain = _leg_pose_solve_chain_from_up(armature, chain, up)
        for reverse_index, bone_name in enumerate(reversed(chain[1:]), start=1):
            bone = armature.data.bones.get(bone_name)
            if not bone:
                continue
            direction = bone.tail_local - bone.head_local
            if direction.length < 1e-6:
                continue
            projected = _horizontal_projection(direction, up)
            if projected is None:
                continue
            horizontal = projected.length / direction.length
            vertical = abs(direction.normalized().dot(up))
            if horizontal < 0.5 or vertical > 0.75 or projected.length < 1e-6:
                continue

            if origin is not None:
                distance_vector = _horizontal_projection(
                    ((bone.head_local + bone.tail_local) * 0.5) - origin,
                    up,
                )
                distance = distance_vector.length if distance_vector is not None else 0.0
            else:
                distance = float(reverse_index)
            score = (distance, -reverse_index, horizontal)
            if best is None or score > best[0]:
                best = (score, projected.normalized())
            break

        if len(trimmed_chain) < 2:
            continue
        start = _profile_point(armature, trimmed_chain[0], use_tail=False)
        end = _profile_point(armature, trimmed_chain[-1], use_tail=True)
        if start is None or end is None:
            continue
        direction = end - start
        projected = _horizontal_projection(direction, up)
        if projected is None:
            continue
        horizontal = projected.length / max(direction.length, 1e-6)
        if horizontal > 0.04:
            fallback_hints.append(projected.normalized() * horizontal)

    if snap_cardinal and distal_hints:
        hint = mathutils.Vector((0.0, 0.0, 0.0))
        for item in distal_hints:
            hint += item
        snapped = _snap_to_world_cardinal_forward(hint)
        if snapped is not None:
            return snapped

    if best is not None:
        if snap_cardinal:
            snapped = _snap_to_world_cardinal_forward(best[1])
            if snapped is not None:
                return snapped
        return best[1]
    if not fallback_hints:
        return None
    hint = mathutils.Vector((0.0, 0.0, 0.0))
    for item in fallback_hints:
        hint += item
    if snap_cardinal:
        snapped = _snap_to_world_cardinal_forward(hint)
        if snapped is not None:
            return snapped
    return hint.normalized() if hint.length >= 1e-6 else None


def _assign_feature_pair_by_side(chains, features, suffix, left_axis):
    if len(features) < 2 or left_axis.length < 1e-6:
        for item in features:
            terminal = item.get("terminal") or item["leaf"]
            chains[f"{_side_from_x(terminal)}_{suffix}"] = _limb_chain_for_leaf(terminal)
        return

    center = mathutils.Vector((0.0, 0.0, 0.0))
    usable = []
    for item in features[:2]:
        endpoint = _feature_endpoint(item)
        if endpoint is None:
            continue
        center += endpoint
        usable.append((item, endpoint))
    if len(usable) < 2:
        return
    center /= len(usable)

    scored = [
        ((endpoint - center).dot(left_axis), item)
        for item, endpoint in usable
    ]
    scored.sort(key=lambda value: value[0])
    right_item = scored[0][1]
    left_item = scored[-1][1]
    right_terminal = right_item.get("terminal") or right_item["leaf"]
    left_terminal = left_item.get("terminal") or left_item["leaf"]
    chains[f"left_{suffix}"] = _limb_chain_for_leaf(left_terminal)
    chains[f"right_{suffix}"] = _limb_chain_for_leaf(right_terminal)


def _geometry_profile(armature):
    root = _largest_root(armature)
    bones = list(armature.data.bones)
    leaves = [bone for bone in bones if not bone.children]
    if not root or len(leaves) < 3:
        return None

    head_leaf = max(leaves, key=lambda bone: bone.tail_local.z)
    remaining = _dedupe_limb_features(
        _leaf_limb_features(bone) for bone in leaves if bone != head_leaf
    )
    if len(remaining) < 4:
        return None

    # A dangling hand or held prop can be lower than a foot.  Legs are more
    # reliably identified by low torso attachment; arms attach higher and
    # usually end farther away from the body's center line.
    foot_features = sorted(remaining, key=lambda item: (item["attach_z"], item["end_z"]))[:2]
    remaining = [item for item in remaining if item not in foot_features]
    hand_features = sorted(remaining, key=lambda item: item["abs_x"], reverse=True)[:2]

    chains = {"spine": _chain_to_leaf(head_leaf)}
    up = _world_up()
    foot_chains = [
        _limb_chain_for_leaf(item.get("terminal") or item["leaf"])
        for item in foot_features
    ]
    forward = _front_hint_from_chain_names(
        armature,
        foot_chains,
        up,
        origin=root.head_local,
        snap_cardinal=True,
    )
    if forward is None:
        left_axis = mathutils.Vector((1.0, 0.0, 0.0))
    else:
        left_axis = _safe_normalized(up.cross(forward), mathutils.Vector((1.0, 0.0, 0.0)))

    _assign_feature_pair_by_side(chains, foot_features, "leg", left_axis)
    _assign_feature_pair_by_side(chains, hand_features, "arm", left_axis)

    profile = {
        "version": _PROFILE_VERSION,
        "method": "rest_pose_geometry",
        "armature": armature.name,
        "pelvis": root.name,
        "head": head_leaf.name,
        "chains": chains,
    }
    return _correct_profile_sides_from_forward(armature, profile)


def characterize_armature(armature):
    """Infer a semantic retarget profile for an armature."""
    if not armature or armature.type != "ARMATURE":
        return None
    return _standard_name_profile(armature) or _geometry_profile(armature)


def _profile_summary(profile):
    if not profile:
        return "no profile"
    chains = profile.get("chains", {})
    parts = []
    for name in ("spine", "left_arm", "right_arm", "left_leg", "right_leg"):
        chain = chains.get(name, [])
        if chain:
            parts.append(f"{name}:{chain[0]}->{chain[-1]}")
    return "; ".join(parts) if parts else "empty chains"


def _pair_summary(pairs):
    grouped = {}
    for source_name, target_name, role in pairs:
        grouped.setdefault(role, []).append((source_name, target_name))

    parts = []
    for role in ("pelvis", "spine", "left_arm", "right_arm", "left_leg", "right_leg"):
        role_pairs = grouped.get(role, [])
        if not role_pairs:
            continue
        first = role_pairs[0]
        last = role_pairs[-1]
        if first == last:
            parts.append(f"{role}:{first[0]}->{first[1]}")
        else:
            parts.append(
                f"{role}:{first[0]}->{first[1]} ... "
                f"{last[0]}->{last[1]}"
            )
    return "; ".join(parts) if parts else "no bone pairs"


def _save_profile(armature, profile):
    armature[_PROFILE_PROP] = json.dumps(profile, sort_keys=True)


def _load_profile(armature):
    raw = armature.get(_PROFILE_PROP)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _matrix_to_rows(matrix):
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def _matrix3_to_rows(matrix):
    matrix = matrix.to_3x3()
    return [[float(matrix[row][col]) for col in range(3)] for row in range(3)]


def _vector_to_list(vector):
    if vector is None:
        return None
    return [float(value) for value in vector]


def _safe_file_stem(value):
    cleaned = []
    for char in str(value or "rig"):
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    stem = "".join(cleaned).strip("_")
    return stem or "rig"


def _rows_to_matrix(rows):
    return mathutils.Matrix(rows)


def _capture_reference_pose(armature, bone_names=None, label="current_pose"):
    if not armature or armature.type != "ARMATURE":
        return None
    names = list(bone_names) if bone_names else [bone.name for bone in armature.pose.bones]
    matrices = {}
    for name in names:
        pose_bone = armature.pose.bones.get(name)
        if not pose_bone:
            continue
        matrices[name] = _matrix_to_rows(pose_bone.matrix_basis.copy())
    return {
        "version": _REFERENCE_POSE_VERSION,
        "armature": armature.name,
        "label": label,
        "bones": matrices,
    }


def _rest_reference_pose(armature, bone_names):
    identity = _matrix_to_rows(mathutils.Matrix.Identity(4))
    return {
        "version": _REFERENCE_POSE_VERSION,
        "armature": armature.name,
        "label": "rest_pose",
        "bones": {name: identity for name in bone_names},
    }


def _save_reference_pose(armature, reference_pose):
    armature[_REFERENCE_POSE_PROP] = json.dumps(reference_pose, sort_keys=True)


def _load_reference_pose(armature):
    raw = armature.get(_REFERENCE_POSE_PROP)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _reference_pose_is_usable(armature, reference_pose, bone_names):
    if not armature or not reference_pose:
        return False
    if reference_pose.get("version") != _REFERENCE_POSE_VERSION:
        return False
    matrices = reference_pose.get("bones", {})
    available = {bone.name for bone in armature.pose.bones}
    return all(name in available and name in matrices for name in bone_names)


def _profile_is_usable(armature, profile):
    if not profile:
        return False
    if profile.get("version") != _PROFILE_VERSION:
        return False
    available = {bone.name for bone in armature.data.bones}
    required = [profile.get("pelvis")]
    for chain in profile.get("chains", {}).values():
        required.extend(chain)
    return all(name in available for name in required if name)


def _profile_for_armature(armature):
    profile = _load_profile(armature)
    if _profile_is_usable(armature, profile):
        return profile
    profile = characterize_armature(armature)
    if profile:
        _save_profile(armature, profile)
    return profile


def _active_rig_mapping_armature():
    if not _rig_mapping_armature_name:
        return None
    obj = bpy.data.objects.get(_rig_mapping_armature_name)
    if obj and obj.type == "ARMATURE":
        return obj
    return None


def _chain_world_points(armature, chain):
    points = []
    for bone_name in chain:
        bone = armature.data.bones.get(bone_name)
        if not bone:
            continue
        points.append(armature.matrix_world @ bone.head_local)
    return points


def _profile_world_points(armature, profile):
    points = []
    chains = profile.get("chains", {}) if profile else {}
    names = {profile.get("pelvis"), profile.get("head")} if profile else set()
    for chain in chains.values():
        names.update(chain)

    for name in names:
        bone = armature.data.bones.get(name) if name else None
        if not bone:
            continue
        points.append(armature.matrix_world @ bone.head_local)
        points.append(armature.matrix_world @ bone.tail_local)
    return points


def _basis_overlay_scale(armature, profile):
    points = _profile_world_points(armature, profile)
    if len(points) < 2:
        return 1.0
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    min_z = min(point.z for point in points)
    max_z = max(point.z for point in points)
    diagonal = mathutils.Vector((max_x - min_x, max_y - min_y, max_z - min_z)).length
    return max(diagonal * 0.28, 0.15)


def _rig_basis_overlay_vectors(armature, profile):
    pelvis = _profile_point(armature, profile.get("pelvis"), use_tail=False)
    if pelvis is None:
        return {}

    basis = _profile_body_basis(armature, profile)
    left, forward, up = _basis_axes(basis)
    world_basis = armature.matrix_world.to_3x3()
    origin = armature.matrix_world @ pelvis
    scale = _basis_overlay_scale(armature, profile)
    up_offset = (world_basis @ up).normalized() * (scale * 0.08)
    origin = origin + up_offset

    vectors = {}
    for name, vector in (("forward", forward), ("left", left)):
        world_vector = world_basis @ vector
        if world_vector.length < 1e-6:
            continue
        direction = world_vector.normalized()
        vectors[name] = {
            "origin": origin,
            "tip": origin + (direction * scale),
            "direction": direction,
            "length": scale,
        }
    return vectors


def _basis_arrow_segments(origin, tip, direction, length):
    segments = [origin, tip]
    up = mathutils.Vector((0.0, 0.0, 1.0))
    side = direction.cross(up)
    if side.length < 1e-6:
        side = direction.cross(mathutils.Vector((1.0, 0.0, 0.0)))
    if side.length < 1e-6:
        return segments

    side.normalize()
    back = -direction
    arrow_len = length * 0.18
    arrow_width = length * 0.08
    left_wing = tip + (back * arrow_len) + (side * arrow_width)
    right_wing = tip + (back * arrow_len) - (side * arrow_width)
    segments.extend([tip, left_wing, tip, right_wing])
    return segments


def _tag_view3d_redraw():
    wm = bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _draw_rig_mapping_overlay():
    armature = _active_rig_mapping_armature()
    profile = _load_profile(armature) if armature else None
    if not _profile_is_usable(armature, profile):
        return

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(4.0)

    try:
        for chain_name, chain in profile.get("chains", {}).items():
            points = _chain_world_points(armature, chain)
            if len(points) < 2:
                continue
            shader.uniform_float(
                "color",
                _CHAIN_COLORS.get(chain_name, (1.0, 1.0, 1.0, 0.9)),
            )
            batch = batch_for_shader(shader, "LINE_STRIP", {"pos": points})
            batch.draw(shader)

        gpu.state.line_width_set(5.0)
        for name, vector in _rig_basis_overlay_vectors(armature, profile).items():
            segments = _basis_arrow_segments(
                vector["origin"],
                vector["tip"],
                vector["direction"],
                vector["length"],
            )
            if len(segments) < 2:
                continue
            shader.uniform_float(
                "color",
                _BASIS_COLORS.get(name, (1.0, 1.0, 1.0, 0.95)),
            )
            batch = batch_for_shader(shader, "LINES", {"pos": segments})
            batch.draw(shader)
    finally:
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")
        gpu.state.line_width_set(1.0)


def _draw_rig_mapping_labels():
    armature = _active_rig_mapping_armature()
    profile = _load_profile(armature) if armature else None
    if not _profile_is_usable(armature, profile):
        return

    context = bpy.context
    area = context.area
    region = context.region
    rv3d = context.region_data
    if not area or area.type != "VIEW_3D" or not region or not rv3d:
        return

    font_id = 0
    blf.size(font_id, 19)
    gpu.state.blend_set("ALPHA")
    try:
        for chain_name, chain in profile.get("chains", {}).items():
            points = _chain_world_points(armature, chain)
            if not points:
                continue
            center = sum(points, mathutils.Vector((0.0, 0.0, 0.0))) / len(points)
            co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, center)
            if co_2d is None:
                continue
            label = _CHAIN_LABELS.get(chain_name, chain_name)
            color = _CHAIN_COLORS.get(chain_name, (1.0, 1.0, 1.0, 0.9))
            blf.color(font_id, 0.0, 0.0, 0.0, 0.75)
            blf.position(font_id, co_2d.x + 8, co_2d.y + 4, 0)
            blf.draw(font_id, label)
            blf.color(font_id, color[0], color[1], color[2], 0.98)
            blf.position(font_id, co_2d.x + 6, co_2d.y + 6, 0)
            blf.draw(font_id, label)

        for name, vector in _rig_basis_overlay_vectors(armature, profile).items():
            co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, vector["tip"])
            if co_2d is None:
                continue
            label = _BASIS_LABELS.get(name, name)
            color = _BASIS_COLORS.get(name, (1.0, 1.0, 1.0, 0.95))
            blf.color(font_id, 0.0, 0.0, 0.0, 0.8)
            blf.position(font_id, co_2d.x + 9, co_2d.y + 5, 0)
            blf.draw(font_id, label)
            blf.color(font_id, color[0], color[1], color[2], 1.0)
            blf.position(font_id, co_2d.x + 7, co_2d.y + 7, 0)
            blf.draw(font_id, label)
    finally:
        gpu.state.blend_set("NONE")


def _ensure_rig_mapping_overlay(armature):
    global _rig_mapping_draw_handle, _rig_mapping_label_draw_handle
    global _rig_mapping_armature_name

    _rig_mapping_armature_name = armature.name
    if _rig_mapping_draw_handle is None:
        _rig_mapping_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_rig_mapping_overlay, (), "WINDOW", "POST_VIEW")
    if _rig_mapping_label_draw_handle is None:
        _rig_mapping_label_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_rig_mapping_labels, (), "WINDOW", "POST_PIXEL")
    _tag_view3d_redraw()


def _remove_rig_mapping_overlay():
    global _rig_mapping_draw_handle, _rig_mapping_label_draw_handle
    global _rig_mapping_armature_name

    if _rig_mapping_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_rig_mapping_draw_handle, "WINDOW")
        _rig_mapping_draw_handle = None
    if _rig_mapping_label_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_rig_mapping_label_draw_handle, "WINDOW")
        _rig_mapping_label_draw_handle = None
    _rig_mapping_armature_name = None
    _tag_view3d_redraw()


def _bone_enum_items(self, context):
    armature = _active_or_selected_armature(context) if context else None
    if not armature:
        return [('NONE', 'None', 'No armature selected')]
    key = (
        armature.name,
        tuple(bone.name for bone in armature.data.bones),
    )
    items = _BONE_ENUM_ITEMS_CACHE.get(key)
    if items is None:
        items = [(bone.name, bone.name, '') for bone in armature.data.bones]
        _BONE_ENUM_ITEMS_CACHE.clear()
        _BONE_ENUM_ITEMS_CACHE[key] = items
    return items


def _set_property_if_bone(operator, prop_name, armature, bone_name):
    if bone_name and armature.data.bones.get(bone_name):
        setattr(operator, prop_name, bone_name)


def _chain_from_profile(profile, name):
    return list(profile.get("chains", {}).get(name, [])) if profile else []


def _initialize_mapping_operator(operator, armature, profile):
    if not armature or not profile:
        return

    _set_property_if_bone(operator, "pelvis", armature, profile.get("pelvis"))
    _set_property_if_bone(operator, "head", armature, profile.get("head"))

    mapping = {
        "spine": ("spine_start", "spine_end"),
        "left_arm": ("left_arm_start", "left_arm_end"),
        "right_arm": ("right_arm_start", "right_arm_end"),
        "left_leg": ("left_leg_start", "left_leg_end"),
        "right_leg": ("right_leg_start", "right_leg_end"),
    }
    for chain_name, (start_prop, end_prop) in mapping.items():
        chain = _chain_from_profile(profile, chain_name)
        if chain:
            _set_property_if_bone(operator, start_prop, armature, chain[0])
            _set_property_if_bone(operator, end_prop, armature, chain[-1])


def _bone_path_between(armature, start_name, end_name):
    start = armature.data.bones.get(start_name)
    end = armature.data.bones.get(end_name)
    if not start or not end:
        return []

    end_path = _path_from_root(end)
    if start in end_path:
        idx = end_path.index(start)
        return [bone.name for bone in end_path[idx:]]

    start_path = _path_from_root(start)
    if end in start_path:
        idx = start_path.index(end)
        return [bone.name for bone in reversed(start_path[idx:])]

    return [start.name, end.name]


def _mesh_armature(obj):
    if not obj or obj.type != "MESH":
        return None
    if obj.parent and obj.parent.type == "ARMATURE":
        return obj.parent
    for modifier in obj.modifiers:
        if modifier.type == "ARMATURE" and modifier.object and modifier.object.type == "ARMATURE":
            return modifier.object
    return None


def _active_or_selected_armature(context):
    active = context.view_layer.objects.active
    if active:
        if active.type == "ARMATURE":
            return active
        armature = _mesh_armature(active)
        if armature:
            return armature

    for obj in context.selected_objects:
        if obj.type == "ARMATURE":
            return obj
        armature = _mesh_armature(obj)
        if armature:
            return armature
    return None


def _common_chain_pairs(
        source_profile, target_profile, swap_target_sides=False,
        source_armature=None, target_armature=None, trim_feet=False):
    source_chains = source_profile.get("chains", {})
    target_chains = target_profile.get("chains", {})
    pairs = []

    def target_key(key):
        if not swap_target_sides:
            return key
        if key.startswith("left_"):
            return "right_" + key[5:]
        if key.startswith("right_"):
            return "left_" + key[6:]
        return key

    if source_profile.get("pelvis") and target_profile.get("pelvis"):
        pairs.append((source_profile["pelvis"], target_profile["pelvis"], "pelvis"))

    for key in ("spine", "left_arm", "right_arm", "left_leg", "right_leg"):
        source = source_chains.get(key, [])
        mapped_target_key = target_key(key)
        target = target_chains.get(mapped_target_key, [])
        if not source or not target:
            continue
        if trim_feet and key in {"left_leg", "right_leg"}:
            source = _retarget_chain_for_role(source_armature, source_profile, key, source)
            target = _retarget_chain_for_role(
                target_armature,
                target_profile,
                mapped_target_key,
                target,
            )
            if not source or not target:
                continue

        if len(target) == 1:
            pairs.append((source[-1], target[0], key))
            continue

        max_src = max(1, len(source) - 1)
        max_tgt = max(1, len(target) - 1)
        for idx, target_bone in enumerate(target):
            source_idx = round((idx / max_tgt) * max_src)
            pairs.append((source[source_idx], target_bone, key))

    deduped = []
    seen = set()
    for source_bone, target_bone, key in pairs:
        marker = (source_bone, target_bone)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append((source_bone, target_bone, key))
    return deduped


def _import_animation_source(filepath):
    before = set(bpy.data.objects.keys())
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".bvh":
        bpy.ops.import_anim.bvh(filepath=filepath, update_scene_duration=True)
    else:
        raise RuntimeError("Animation source must be FBX or BVH")

    new_objects = [obj for obj in bpy.data.objects if obj.name not in before]
    armatures = [obj for obj in new_objects if obj.type == "ARMATURE"]
    if not armatures:
        armatures = [obj for obj in bpy.context.selected_objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature was imported from the animation source")
    source = armatures[0]

    if source.animation_data is None or source.animation_data.action is None:
        for obj in new_objects:
            if obj.animation_data and obj.animation_data.action:
                source.animation_data_create()
                source.animation_data.action = obj.animation_data.action
                break

    return source, new_objects


def _action_frame_range(armature, fallback_scene):
    action = armature.animation_data.action if armature.animation_data else None
    if action:
        start, end = action.frame_range
        return int(start), int(end)
    return int(fallback_scene.frame_start), int(fallback_scene.frame_end)


def _clear_retarget_constraints(target_armature):
    for pose_bone in target_armature.pose.bones:
        for constraint in list(pose_bone.constraints):
            if constraint.name.startswith(_CONSTRAINT_PREFIX):
                pose_bone.constraints.remove(constraint)


def _add_copy_constraints(source_armature, target_armature, pairs, copy_root_motion):
    added = 0
    for source_name, target_name, role in pairs:
        source_pose = source_armature.pose.bones.get(source_name)
        target_pose = target_armature.pose.bones.get(target_name)
        if not source_pose or not target_pose:
            continue

        constraint = target_pose.constraints.new(type="COPY_ROTATION")
        constraint.name = f"{_CONSTRAINT_PREFIX}Rot_{source_name}"
        constraint.target = source_armature
        constraint.subtarget = source_name
        constraint.target_space = "LOCAL_WITH_PARENT"
        constraint.owner_space = "LOCAL_WITH_PARENT"
        if hasattr(constraint, "mix_mode"):
            constraint.mix_mode = "REPLACE"
        added += 1

        if role == "pelvis" and copy_root_motion:
            loc_constraint = target_pose.constraints.new(type="COPY_LOCATION")
            loc_constraint.name = f"{_CONSTRAINT_PREFIX}Loc_{source_name}"
            loc_constraint.target = source_armature
            loc_constraint.subtarget = source_name
            loc_constraint.target_space = "LOCAL"
            loc_constraint.owner_space = "LOCAL"
            if hasattr(loc_constraint, "use_offset"):
                loc_constraint.use_offset = True
            added += 1
    return added


def _mapped_target_bones(pairs):
    seen = set()
    names = []
    for _source_name, target_name, _role in pairs:
        if target_name in seen:
            continue
        seen.add(target_name)
        names.append(target_name)
    return names


def _reset_target_pose(target_armature, bone_names):
    for name in bone_names:
        pose_bone = target_armature.pose.bones.get(name)
        if not pose_bone:
            continue
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)


def _apply_target_reference_pose(target_armature, bone_names, reference_pose):
    matrices = (reference_pose or {}).get("bones", {})
    for name in bone_names:
        pose_bone = target_armature.pose.bones.get(name)
        if not pose_bone:
            continue
        rows = matrices.get(name)
        if rows:
            pose_bone.matrix_basis = _rows_to_matrix(rows)
        else:
            pose_bone.matrix_basis = mathutils.Matrix.Identity(4)


def _target_reference_state(target_armature, bone_names):
    state = {}
    for name in bone_names:
        pose_bone = target_armature.pose.bones.get(name)
        if not pose_bone:
            continue
        state[name] = {
            "matrix": pose_bone.matrix.copy(),
            "location": pose_bone.location.copy(),
            "scale": pose_bone.scale.copy(),
        }
    return state


def _source_rest_pose_baseline(source_armature, pairs):
    baseline = {}
    for source_name, _target_name, _role in pairs:
        bone = source_armature.data.bones.get(source_name)
        if not bone:
            continue
        loc, rot, _scale = bone.matrix_local.decompose()
        baseline[source_name] = (loc.copy(), rot.copy())
    return baseline


def _height_from_profile(armature, profile):
    chains = profile.get("chains", {})
    names = set([profile.get("pelvis"), profile.get("head")])
    for chain in chains.values():
        names.update(chain)

    coords = []
    for name in names:
        bone = armature.data.bones.get(name)
        if not bone:
            continue
        coords.append(bone.head_local)
        coords.append(bone.tail_local)
    if not coords:
        return 1.0

    min_z = min(coord.z for coord in coords)
    max_z = max(coord.z for coord in coords)
    return max(max_z - min_z, 1e-5)


def _profile_point(armature, bone_name, use_tail=False):
    bone = armature.data.bones.get(bone_name) if bone_name else None
    if not bone:
        return None
    return bone.tail_local.copy() if use_tail else bone.head_local.copy()


def _pose_point(armature, bone_name, use_tail=False):
    pose_bone = armature.pose.bones.get(bone_name) if bone_name else None
    if not pose_bone:
        return None
    return pose_bone.tail.copy() if use_tail else pose_bone.head.copy()


def _chain_end_point(armature, chain):
    if not chain:
        return None
    return _profile_point(armature, chain[-1], use_tail=True)


def _pose_chain_end_point(armature, chain):
    if not chain:
        return None
    return _pose_point(armature, chain[-1], use_tail=True)


def _average_points(points):
    points = [point for point in points if point is not None]
    if not points:
        return None
    total = mathutils.Vector((0.0, 0.0, 0.0))
    for point in points:
        total += point
    return total / len(points)


def _safe_normalized(vector, fallback):
    if vector is None or vector.length < 1e-6:
        return fallback.normalized()
    return vector.normalized()


def _profile_up_axis(armature, profile):
    pelvis = _profile_point(armature, profile.get("pelvis"), use_tail=False)
    head = _profile_point(armature, profile.get("head"), use_tail=True)
    up = _safe_normalized(
        (head - pelvis) if head is not None and pelvis is not None else None,
        _world_up(),
    )
    if _profile_uses_world_up(profile) and up.dot(_world_up()) < 0.0:
        up = -up
    return up


def _pose_up_axis(armature, profile):
    pelvis = _pose_point(armature, profile.get("pelvis"), use_tail=False)
    head = _pose_point(armature, profile.get("head"), use_tail=True)
    up = _safe_normalized(
        (head - pelvis) if head is not None and pelvis is not None else None,
        _world_up(),
    )
    if _profile_uses_world_up(profile) and up.dot(_world_up()) < 0.0:
        up = -up
    return up


def _profile_side_axis(armature, profile, up):
    chains = profile.get("chains", {})
    left_point = _average_points([
        _chain_end_point(armature, chains.get("left_arm", [])),
        _chain_end_point(armature, chains.get("left_leg", [])),
    ])
    right_point = _average_points([
        _chain_end_point(armature, chains.get("right_arm", [])),
        _chain_end_point(armature, chains.get("right_leg", [])),
    ])
    side = (left_point - right_point) if left_point is not None and right_point is not None else None
    projected = _horizontal_projection(side, up)
    return _safe_normalized(projected, mathutils.Vector((1.0, 0.0, 0.0)))


def _pose_side_axis(armature, profile, up):
    chains = profile.get("chains", {})
    left_point = _average_points([
        _pose_chain_end_point(armature, chains.get("left_arm", [])),
        _pose_chain_end_point(armature, chains.get("left_leg", [])),
    ])
    right_point = _average_points([
        _pose_chain_end_point(armature, chains.get("right_arm", [])),
        _pose_chain_end_point(armature, chains.get("right_leg", [])),
    ])
    side = (left_point - right_point) if left_point is not None and right_point is not None else None
    projected = _horizontal_projection(side, up)
    return _safe_normalized(projected, mathutils.Vector((1.0, 0.0, 0.0)))


def _rest_foot_front_hint(armature, profile, up):
    chains = profile.get("chains", {})
    pelvis = _profile_point(armature, profile.get("pelvis"), use_tail=False)
    return _front_hint_from_chain_names(
        armature,
        [chains.get("left_leg", []), chains.get("right_leg", [])],
        up,
        origin=pelvis,
        snap_cardinal=_profile_uses_world_up(profile),
    )


def _body_axes_from_side_up(armature, profile, left, up):
    fallback_forward = mathutils.Vector((0.0, 1.0, 0.0))
    hint = _rest_foot_front_hint(armature, profile, up)
    if hint is not None:
        forward = _safe_normalized(_horizontal_projection(hint, up), fallback_forward)
        left = _safe_normalized(up.cross(forward), left)
        return left, forward

    forward = _safe_normalized(left.cross(up), fallback_forward)
    forward = _safe_normalized(forward - (up * forward.dot(up)), fallback_forward)
    return left, forward


def _profile_body_basis(armature, profile, flip_forward=False):
    up = _profile_up_axis(armature, profile)
    left = _profile_side_axis(armature, profile, up)
    left, forward = _body_axes_from_side_up(armature, profile, left, up)
    if flip_forward:
        forward = -forward
    return mathutils.Matrix((left, forward, up)).transposed()


def _pose_body_basis(armature, profile, flip_forward=False):
    up = _pose_up_axis(armature, profile)
    left = _pose_side_axis(armature, profile, up)
    left, forward = _body_axes_from_side_up(armature, profile, left, up)
    if flip_forward:
        forward = -forward
    return mathutils.Matrix((left, forward, up)).transposed()


def _basis_axes(basis):
    return (
        basis @ mathutils.Vector((1.0, 0.0, 0.0)),
        basis @ mathutils.Vector((0.0, 1.0, 0.0)),
        basis @ mathutils.Vector((0.0, 0.0, 1.0)),
    )


def _rotation_basis_from_direction_basis(direction_basis):
    left, forward, up = _basis_axes(direction_basis)
    up = _safe_normalized(up, mathutils.Vector((0.0, 0.0, 1.0)))
    forward = _safe_normalized(
        forward - (up * forward.dot(up)),
        mathutils.Vector((0.0, 1.0, 0.0)),
    )
    right = _safe_normalized(forward.cross(up), -left)
    forward = _safe_normalized(up.cross(right), forward)
    return mathutils.Matrix((right, forward, up)).transposed()


def _profile_rotation_basis(armature, profile, flip_forward=False):
    return _rotation_basis_from_direction_basis(
        _profile_body_basis(armature, profile, flip_forward=flip_forward)
    )


def _pose_rotation_basis(armature, profile, flip_forward=False):
    return _rotation_basis_from_direction_basis(
        _pose_body_basis(armature, profile, flip_forward=flip_forward)
    )


def _basis_diagnostics(armature, profile, flip_forward=False):
    direction_basis = _profile_body_basis(
        armature,
        profile,
        flip_forward=flip_forward,
    )
    rotation_basis = _profile_rotation_basis(
        armature,
        profile,
        flip_forward=flip_forward,
    )
    left, forward, up = _basis_axes(direction_basis)
    lateral, rotation_forward, rotation_up = _basis_axes(rotation_basis)
    hint = _rest_foot_front_hint(armature, profile, up)
    return {
        "direction_basis": {
            "left": _vector_to_list(left),
            "forward": _vector_to_list(forward),
            "up": _vector_to_list(up),
            "matrix": _matrix3_to_rows(direction_basis),
            "determinant": float(direction_basis.to_3x3().determinant()),
        },
        "rotation_basis": {
            "lateral": _vector_to_list(lateral),
            "forward": _vector_to_list(rotation_forward),
            "up": _vector_to_list(rotation_up),
            "matrix": _matrix3_to_rows(rotation_basis),
            "determinant": float(rotation_basis.to_3x3().determinant()),
        },
        "foot_front_hint": _vector_to_list(hint),
    }


def _axis_summary(label, armature, profile, flip_forward=False):
    basis = _profile_rotation_basis(armature, profile, flip_forward=flip_forward)
    lateral, forward, up = _basis_axes(basis)
    det = basis.to_3x3().determinant()
    return (
        f"{label} axes: lateral={tuple(round(v, 3) for v in lateral)}, "
        f"forward={tuple(round(v, 3) for v in forward)}, "
        f"up={tuple(round(v, 3) for v in up)}, det={det:.3f}"
    )


def _swap_profile_side_chains(profile):
    chains = profile.get("chains", {})
    for suffix in ("arm", "leg"):
        left_key = f"left_{suffix}"
        right_key = f"right_{suffix}"
        if left_key in chains and right_key in chains:
            chains[left_key], chains[right_key] = chains[right_key], chains[left_key]
    profile["side_corrected_from_forward"] = True
    return profile


def _correct_profile_sides_from_forward(armature, profile):
    try:
        up = _profile_up_axis(armature, profile)
        forward = _rest_foot_front_hint(armature, profile, up)
        if forward is None:
            return profile
        forward = _safe_normalized(_horizontal_projection(forward, up), mathutils.Vector((0.0, 1.0, 0.0)))
        current_left = _profile_side_axis(armature, profile, up)
        semantic_left = up.cross(forward)
        if semantic_left.length < 1e-6:
            return profile
        semantic_left.normalize()
        if current_left.normalized().dot(semantic_left) < -0.25:
            return _swap_profile_side_chains(profile)
    except Exception:
        return profile
    return profile


def _source_pose_baseline(source_armature, pairs):
    baseline = {}
    for source_name, _target_name, _role in pairs:
        source_pose = source_armature.pose.bones.get(source_name)
        if not source_pose:
            continue
        loc, rot, _scale = source_pose.matrix.decompose()
        baseline[source_name] = (loc.copy(), rot.copy())
    return baseline


def _source_baseline(source_armature, pairs, source_reference_mode):
    if source_reference_mode == "START_FRAME":
        return _source_pose_baseline(source_armature, pairs)
    return _source_rest_pose_baseline(source_armature, pairs)


def _apply_pose_delta(
        source_armature, target_armature, pairs, copy_root_motion,
        location_scale, source_baseline, body_alignment,
        target_reference_state, context):
    alignment_m = body_alignment.to_3x3()
    alignment_inv = alignment_m.inverted_safe()
    for source_name, target_name, role in pairs:
        source_pose = source_armature.pose.bones.get(source_name)
        target_pose = target_armature.pose.bones.get(target_name)
        if not source_pose or not target_pose:
            continue

        loc, rot, _scale = source_pose.matrix.decompose()
        base_loc, base_rot = source_baseline.get(source_name, (None, None))
        if base_rot:
            rot = rot @ base_rot.inverted()
            rot = (alignment_m @ rot.to_matrix() @ alignment_inv).to_quaternion()
        if base_loc:
            loc = loc - base_loc
            loc = alignment_m @ loc

        target_pose.rotation_mode = "QUATERNION"
        ref_state = target_reference_state.get(target_name, {})
        ref_matrix = ref_state.get("matrix")
        if ref_matrix is None:
            ref_matrix = target_pose.bone.matrix_local.copy()
        ref_basis_loc = ref_state.get("location", mathutils.Vector((0.0, 0.0, 0.0))).copy()
        ref_basis_scale = ref_state.get("scale", mathutils.Vector((1.0, 1.0, 1.0))).copy()
        ref_loc, ref_rot, _ref_scale = ref_matrix.decompose()
        desired_rot = rot @ ref_rot
        desired_matrix = desired_rot.to_matrix().to_4x4()
        desired_matrix.translation = ref_loc
        target_pose.matrix = desired_matrix
        if role == "pelvis" and copy_root_motion:
            target_pose.location = ref_basis_loc + (loc * location_scale)
        else:
            target_pose.location = ref_basis_loc
        target_pose.scale = ref_basis_scale
        context.view_layer.update()


def _keyframe_retarget_pose(
        target_armature, target_names, frame, copy_root_motion,
        key_locations=False, key_scale=False):
    for name in target_names:
        pose_bone = target_armature.pose.bones.get(name)
        if not pose_bone:
            continue
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if copy_root_motion or key_locations:
            pose_bone.keyframe_insert(data_path="location", frame=frame)
        if key_scale:
            pose_bone.keyframe_insert(data_path="scale", frame=frame)


def _bake_pose_delta_retarget(
        context, source_armature, target_armature, pairs,
        source_profile, target_profile, start, end, frame_step,
        copy_root_motion, target_reference_pose, flip_target_forward=False,
        source_reference_mode="START_FRAME"):
    target_names = _mapped_target_bones(pairs)
    source_height = _height_from_profile(source_armature, source_profile)
    target_height = _height_from_profile(target_armature, target_profile)
    location_scale = target_height / source_height if source_height > 1e-5 else 1.0
    context.scene.frame_set(start)
    context.view_layer.update()
    source_baseline = _source_baseline(source_armature, pairs, source_reference_mode)
    source_basis = _profile_rotation_basis(source_armature, source_profile)
    target_basis = _profile_rotation_basis(
        target_armature,
        target_profile,
        flip_forward=flip_target_forward,
    )
    body_alignment = target_basis @ source_basis.inverted()
    if target_reference_pose.get("label") == "rest_pose":
        _reset_target_pose(target_armature, target_names)
    else:
        _apply_target_reference_pose(target_armature, target_names, target_reference_pose)
    context.view_layer.update()
    reference_state = _target_reference_state(target_armature, target_names)

    for frame in range(start, end + 1, frame_step):
        context.scene.frame_set(frame)
        context.view_layer.update()
        if target_reference_pose.get("label") == "rest_pose":
            _reset_target_pose(target_armature, target_names)
        else:
            _apply_target_reference_pose(target_armature, target_names, target_reference_pose)
        context.view_layer.update()
        _apply_pose_delta(
            source_armature,
            target_armature,
            pairs,
            copy_root_motion=copy_root_motion,
            location_scale=location_scale,
            source_baseline=source_baseline,
            body_alignment=body_alignment,
            target_reference_state=reference_state,
            context=context,
        )
        context.view_layer.update()
        _keyframe_retarget_pose(
            target_armature,
            target_names,
            frame,
            copy_root_motion=copy_root_motion,
            key_locations=target_reference_pose.get("label") != "rest_pose",
            key_scale=target_reference_pose.get("label") != "rest_pose",
        )


def _target_chain_key(source_chain_key, swap_target_sides=False):
    if not swap_target_sides:
        return source_chain_key
    if source_chain_key.startswith("left_"):
        return "right_" + source_chain_key[5:]
    if source_chain_key.startswith("right_"):
        return "left_" + source_chain_key[6:]
    return source_chain_key


def _pose_chain_direction(armature, chain):
    if len(chain) < 2:
        return None
    start = _pose_point(armature, chain[0], use_tail=False)
    end = _pose_point(armature, chain[-1], use_tail=True)
    if start is None or end is None:
        return None
    direction = end - start
    if direction.length < 1e-6:
        return None
    return direction.normalized()


def _rotate_pose_bone_in_armature_space(pose_bone, delta_quat):
    loc, rot, scale = pose_bone.matrix.decompose()
    pose_bone.rotation_mode = "QUATERNION"
    desired_rot = delta_quat @ rot
    desired_matrix = desired_rot.to_matrix().to_4x4()
    desired_matrix.translation = loc
    pose_bone.matrix = desired_matrix
    pose_bone.scale = scale


def _match_limb_chain_directions(
        context, source_armature, target_armature,
        source_profile, target_profile, strength=1.0,
        swap_target_sides=False):
    source_basis = _pose_body_basis(source_armature, source_profile)
    target_basis = _pose_body_basis(target_armature, target_profile)
    source_chains = source_profile.get("chains", {})
    target_chains = target_profile.get("chains", {})
    strength = max(0.0, min(float(strength), 1.0))
    matched = 0

    for source_key in ("left_arm", "right_arm", "left_leg", "right_leg"):
        target_key = _target_chain_key(source_key, swap_target_sides=swap_target_sides)
        source_chain = source_chains.get(source_key, [])
        target_chain = target_chains.get(target_key, [])
        source_direction = _pose_chain_direction(source_armature, source_chain)
        target_direction = _pose_chain_direction(target_armature, target_chain)
        if source_direction is None or target_direction is None:
            continue

        source_body_direction = source_basis.inverted() @ source_direction
        desired_target_direction = target_basis @ source_body_direction
        if desired_target_direction.length < 1e-6:
            continue
        desired_target_direction.normalize()

        try:
            delta = target_direction.rotation_difference(desired_target_direction)
        except ValueError:
            continue
        if strength < 1.0:
            delta = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(delta, strength)

        root_pose = target_armature.pose.bones.get(target_chain[0])
        if not root_pose:
            continue
        _rotate_pose_bone_in_armature_space(root_pose, delta)
        context.view_layer.update()
        matched += 1

    return matched


def _match_source_reference_pose(
        context, source_armature, target_armature, pairs,
        source_profile, target_profile, strength=1.0,
        swap_target_sides=False):
    target_names = _mapped_target_bones(pairs)
    all_target_names = [pose_bone.name for pose_bone in target_armature.pose.bones]
    source_height = _height_from_profile(source_armature, source_profile)
    target_height = _height_from_profile(target_armature, target_profile)
    location_scale = target_height / source_height if source_height > 1e-5 else 1.0

    _reset_target_pose(target_armature, all_target_names)
    context.view_layer.update()
    source_baseline = _source_rest_pose_baseline(source_armature, pairs)
    source_basis = _profile_rotation_basis(source_armature, source_profile)
    target_basis = _profile_rotation_basis(target_armature, target_profile)
    body_alignment = target_basis @ source_basis.inverted()
    reference_state = _target_reference_state(target_armature, target_names)
    _apply_pose_delta(
        source_armature,
        target_armature,
        pairs,
        copy_root_motion=False,
        location_scale=location_scale,
        source_baseline=source_baseline,
        body_alignment=body_alignment,
        target_reference_state=reference_state,
        context=context,
    )
    context.view_layer.update()
    matched_chains = _match_limb_chain_directions(
        context,
        source_armature,
        target_armature,
        source_profile,
        target_profile,
        strength=strength,
        swap_target_sides=swap_target_sides,
    )
    return matched_chains


_POSE_PRESET_LABELS = {
    "REST": "Rest",
    "A_POSE": "A-Pose",
    "T_POSE": "T-Pose",
}


_POSE_PRESET_DIRECTIONS = {
    "T_POSE": {
        "left_arm": (1.0, 0.0, 0.0),
        "right_arm": (-1.0, 0.0, 0.0),
        "left_leg": (0.06, 0.0, -1.0),
        "right_leg": (-0.06, 0.0, -1.0),
    },
    "A_POSE": {
        "left_arm": (0.72, 0.0, -0.69),
        "right_arm": (-0.72, 0.0, -0.69),
        "left_leg": (0.08, 0.0, -1.0),
        "right_leg": (-0.08, 0.0, -1.0),
    },
}


def _leg_pose_solve_chain_from_up(armature, chain, up_axis):
    if len(chain) < 3:
        return chain

    up_axis = _safe_normalized(up_axis, mathutils.Vector((0.0, 0.0, 1.0)))
    first_foot_candidate = max(2, len(chain) - 2)
    for idx, bone_name in enumerate(chain[1:], start=1):
        if idx < first_foot_candidate:
            continue
        bone = armature.data.bones.get(bone_name)
        if not bone:
            continue
        direction = bone.tail_local - bone.head_local
        if direction.length < 1e-6:
            continue
        direction.normalize()
        vertical = abs(direction.dot(up_axis))
        horizontal = (direction - (up_axis * direction.dot(up_axis))).length
        if horizontal > 0.65 and vertical < 0.6:
            return chain[:idx]

    return chain[:-1]


def _leg_pose_solve_chain(armature, chain, basis):
    up_axis = basis @ mathutils.Vector((0.0, 0.0, 1.0))
    return _leg_pose_solve_chain_from_up(armature, chain, up_axis)


def _retarget_chain_for_role(armature, profile, role, chain):
    if role not in {"left_leg", "right_leg"}:
        return chain
    if not armature or not profile or len(chain) < 3:
        return chain
    try:
        basis = _profile_body_basis(armature, profile)
        trimmed = _leg_pose_solve_chain(armature, chain, basis)
    except Exception:
        return chain
    if len(trimmed) > 2:
        trimmed = trimmed[:-1]
    return trimmed if len(trimmed) >= 2 else chain


def _pose_preset_solve_chain(armature, chain, chain_key, basis):
    if chain_key in {"left_leg", "right_leg"}:
        return _leg_pose_solve_chain(armature, chain, basis)
    return chain


def _pose_preset_direction(current_direction, basis, body_direction, preserve_forward_limit=0.2):
    body = mathutils.Vector(body_direction)
    flat_body = mathutils.Vector((body.x, 0.0, body.z))
    if flat_body.length < 1e-6:
        flat_body = body.copy()
    if flat_body.length < 1e-6:
        return None

    side_up_direction = basis @ flat_body.normalized()
    forward_axis = (basis @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()
    forward_component = current_direction.dot(forward_axis)
    if preserve_forward_limit is not None:
        limit = abs(float(preserve_forward_limit))
        forward_component = max(-limit, min(limit, forward_component))
    side_up_scale = max(0.0, 1.0 - (forward_component * forward_component)) ** 0.5
    desired = (side_up_direction * side_up_scale) + (forward_axis * forward_component)
    if desired.length < 1e-6:
        return side_up_direction.normalized()
    return desired.normalized()


def _apply_pose_chain_direction(context, armature, chain, desired_direction, strength=1.0):
    if len(chain) < 2:
        return False
    current_direction = _pose_chain_direction(armature, chain)
    if current_direction is None or desired_direction.length < 1e-6:
        return False

    desired_direction = desired_direction.normalized()
    try:
        delta = current_direction.rotation_difference(desired_direction)
    except ValueError:
        return False

    strength = max(0.0, min(float(strength), 1.0))
    if strength < 1.0:
        delta = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(delta, strength)

    root_pose = armature.pose.bones.get(chain[0])
    if not root_pose:
        return False
    _rotate_pose_bone_in_armature_space(root_pose, delta)
    context.view_layer.update()
    return True


def _apply_rig_pose_preset(
        context, armature, profile, preset, strength=1.0,
        adjust_arms=True, adjust_legs=True):
    all_pose_bones = [pose_bone.name for pose_bone in armature.pose.bones]
    _reset_target_pose(armature, all_pose_bones)
    context.view_layer.update()

    if preset == "REST":
        _tag_view3d_redraw()
        return 0

    chains = profile.get("chains", {})
    directions = _POSE_PRESET_DIRECTIONS.get(preset, {})
    if not directions:
        return 0

    basis = _profile_body_basis(armature, profile)
    applied = 0
    for chain_key, body_direction in directions.items():
        if chain_key.endswith("_arm") and not adjust_arms:
            continue
        if chain_key.endswith("_leg") and not adjust_legs:
            continue

        chain = chains.get(chain_key, [])
        solve_chain = _pose_preset_solve_chain(armature, chain, chain_key, basis)
        current_direction = _pose_chain_direction(armature, solve_chain)
        if current_direction is None:
            continue
        preserve_forward_limit = None if chain_key.endswith("_leg") else 0.2
        desired = _pose_preset_direction(
            current_direction,
            basis,
            body_direction,
            preserve_forward_limit=preserve_forward_limit,
        )
        if desired is None:
            continue
        if _apply_pose_chain_direction(context, armature, solve_chain, desired, strength=strength):
            applied += 1

    _tag_view3d_redraw()
    return applied


def _delete_objects(objects):
    if not objects:
        return
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj.name in bpy.data.objects:
            obj.select_set(True)
    bpy.ops.object.delete()


def _default_debug_bundle_directory():
    base = bpy.path.abspath("//") if bpy.data.filepath else ""
    if not base or base == "//":
        base = getattr(bpy.app, "tempdir", "") or os.path.expanduser("~")
    return os.path.join(base, "stablegen_retarget_debug")


def _rig_export_objects(armature):
    objects = [armature]
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if _mesh_armature(obj) == armature and obj not in objects:
            objects.append(obj)
    return objects


def _export_rig_fbx(context, armature, filepath):
    objects = _rig_export_objects(armature)
    original_active = context.view_layer.objects.active
    original_selection = list(context.selected_objects)

    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        context.view_layer.objects.active = armature
        context.view_layer.update()
        bpy.ops.export_scene.fbx(
            filepath=filepath,
            use_selection=True,
            object_types={"ARMATURE", "MESH"},
            add_leaf_bones=False,
            bake_anim=False,
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in original_selection:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if original_active and original_active.name in bpy.data.objects:
            context.view_layer.objects.active = original_active

    return objects


def _bone_diagnostics(armature):
    result = {}
    for bone in armature.data.bones:
        result[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "children": [child.name for child in bone.children],
            "head_local": _vector_to_list(bone.head_local),
            "tail_local": _vector_to_list(bone.tail_local),
            "matrix_local": _matrix_to_rows(bone.matrix_local),
        }
    return result


def _pose_bone_diagnostics(armature):
    result = {}
    for pose_bone in armature.pose.bones:
        result[pose_bone.name] = {
            "head": _vector_to_list(pose_bone.head),
            "tail": _vector_to_list(pose_bone.tail),
            "matrix": _matrix_to_rows(pose_bone.matrix),
            "matrix_basis": _matrix_to_rows(pose_bone.matrix_basis),
            "rotation_mode": pose_bone.rotation_mode,
        }
    return result


def _chain_point_diagnostics(armature, profile):
    result = {}
    for chain_name, chain in profile.get("chains", {}).items():
        points = []
        for bone_name in chain:
            bone = armature.data.bones.get(bone_name)
            if not bone:
                continue
            points.append({
                "bone": bone.name,
                "head_local": _vector_to_list(bone.head_local),
                "tail_local": _vector_to_list(bone.tail_local),
            })
        result[chain_name] = points
    return result


def _alignment_diagnostics(source_armature, source_profile, target_armature, target_profile, flip_target_forward=False):
    source_basis = _profile_rotation_basis(source_armature, source_profile)
    target_basis = _profile_rotation_basis(
        target_armature,
        target_profile,
        flip_forward=flip_target_forward,
    )
    alignment = target_basis @ source_basis.inverted()
    return {
        "source_basis": _basis_diagnostics(source_armature, source_profile),
        "target_basis": _basis_diagnostics(
            target_armature,
            target_profile,
            flip_forward=flip_target_forward,
        ),
        "alignment_matrix": _matrix3_to_rows(alignment),
        "alignment_determinant": float(alignment.to_3x3().determinant()),
    }


def _mapping_option_diagnostics(source_armature, source_profile, target_armature, target_profile):
    options = {}
    for name, swap_target_sides, flip_target_forward in (
            ("normal", False, False),
            ("swap_target_sides", True, False),
            ("flip_target_forward", False, True),
            ("swap_sides_and_flip_forward", True, True)):
        pairs = _common_chain_pairs(
            source_profile,
            target_profile,
            swap_target_sides=swap_target_sides,
            source_armature=source_armature,
            target_armature=target_armature,
            trim_feet=True,
        )
        options[name] = {
            "swap_target_sides": swap_target_sides,
            "flip_target_forward": flip_target_forward,
            "pair_count": len(pairs),
            "summary": _pair_summary(pairs),
            "pairs": [
                {"source": source, "target": target, "role": role}
                for source, target, role in pairs
            ],
            "alignment": _alignment_diagnostics(
                source_armature,
                source_profile,
                target_armature,
                target_profile,
                flip_target_forward=flip_target_forward,
            ),
        }
    return options


def _animation_source_diagnostics(context, filepath, target_armature, target_profile):
    if not filepath:
        return None
    filepath = bpy.path.abspath(filepath)
    if not os.path.exists(filepath):
        return {"error": f"Animation file does not exist: {filepath}"}

    original_frame = context.scene.frame_current
    source_armature = None
    imported_objects = []
    try:
        source_armature, imported_objects = _import_animation_source(filepath)
        source_profile = characterize_armature(source_armature)
        if not source_profile:
            return {"error": "Could not infer a humanoid profile for the source animation rig"}
        start, end = _action_frame_range(source_armature, context.scene)
        context.scene.frame_set(start)
        context.view_layer.update()
        action = source_armature.animation_data.action if source_armature.animation_data else None
        return {
            "filepath": filepath,
            "armature": source_armature.name,
            "action": action.name if action else None,
            "frame_range": [start, end],
            "profile": source_profile,
            "summary": _profile_summary(source_profile),
            "bones": _bone_diagnostics(source_armature),
            "pose_bones_at_start": _pose_bone_diagnostics(source_armature),
            "basis": _basis_diagnostics(source_armature, source_profile),
            "mapping_options": _mapping_option_diagnostics(
                source_armature,
                source_profile,
                target_armature,
                target_profile,
            ),
        }
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}
    finally:
        if imported_objects:
            _delete_objects(imported_objects)
        context.scene.frame_set(original_frame)
        context.view_layer.update()


def _retarget_bundle_diagnostics(context, armature, profile, target_fbx_path, animation_filepath):
    target_names = sorted({bone.name for bone in armature.pose.bones})
    stored_reference = _load_reference_pose(armature)
    current_reference = _capture_reference_pose(
        armature,
        bone_names=target_names,
        label="current_pose_at_export",
    )
    return {
        "schema": "stablegen_retarget_debug_bundle",
        "schema_version": 1,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "blender_version": bpy.app.version_string,
        "stablegen_profile_version": _PROFILE_VERSION,
        "files": {
            "target_fbx": target_fbx_path,
            "animation_filepath": bpy.path.abspath(animation_filepath) if animation_filepath else None,
        },
        "target": {
            "object": armature.name,
            "profile": profile,
            "summary": _profile_summary(profile),
            "matrix_world": _matrix_to_rows(armature.matrix_world),
            "basis": _basis_diagnostics(armature, profile),
            "basis_flipped_forward": _basis_diagnostics(
                armature,
                profile,
                flip_forward=True,
            ),
            "bones": _bone_diagnostics(armature),
            "pose_bones": _pose_bone_diagnostics(armature),
            "chains": _chain_point_diagnostics(armature, profile),
            "stored_reference_pose": stored_reference,
            "current_reference_pose": current_reference,
        },
        "source_animation": _animation_source_diagnostics(
            context,
            animation_filepath,
            armature,
            profile,
        ),
    }


class StableGenExportRetargetDebugBundle(bpy.types.Operator):
    """Export the selected rig and retarget diagnostics for automated testing."""

    bl_idname = "stablegen.export_retarget_debug_bundle"
    bl_label = "Export Retarget Test Bundle"
    bl_options = {"REGISTER"}

    directory: bpy.props.StringProperty(
        name="Output Folder",
        description="Folder where the target FBX and diagnostics JSON will be written",
        subtype="DIR_PATH",
    )  # type: ignore

    animation_filepath: bpy.props.StringProperty(
        name="Animation File",
        description="Optional FBX/BVH animation to include in the diagnostics",
        subtype="FILE_PATH",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def invoke(self, context, _event):
        if not self.directory:
            self.directory = _default_debug_bundle_directory()
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "directory")
        layout.prop(self, "animation_filepath")

    def execute(self, context):
        armature = _active_or_selected_armature(context)
        profile = _profile_for_armature(armature)
        if not _profile_is_usable(armature, profile):
            self.report({"ERROR"}, "Analyze or edit the rig mapping before exporting a test bundle.")
            return {"CANCELLED"}

        output_dir = bpy.path.abspath(self.directory) if self.directory else _default_debug_bundle_directory()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{_safe_file_stem(armature.name)}_{timestamp}"
        target_fbx_path = os.path.join(output_dir, f"{stem}_target.fbx")
        diagnostics_path = os.path.join(output_dir, f"{stem}_diagnostics.json")

        animation_filepath = self.animation_filepath.strip()
        if animation_filepath and not os.path.exists(bpy.path.abspath(animation_filepath)):
            self.report({"ERROR"}, "Selected animation file does not exist.")
            return {"CANCELLED"}

        original_active = context.view_layer.objects.active
        original_selection = list(context.selected_objects)
        original_frame = context.scene.frame_current
        try:
            exported_objects = _export_rig_fbx(context, armature, target_fbx_path)
            diagnostics = _retarget_bundle_diagnostics(
                context,
                armature,
                profile,
                target_fbx_path,
                animation_filepath,
            )
            diagnostics["files"]["diagnostics_json"] = diagnostics_path
            diagnostics["target"]["exported_objects"] = [obj.name for obj in exported_objects]
            with open(diagnostics_path, "w", encoding="utf-8") as handle:
                json.dump(diagnostics, handle, indent=2, sort_keys=True)

            print(f"[StableGen] Retarget debug target FBX: {target_fbx_path}")
            print(f"[StableGen] Retarget debug diagnostics: {diagnostics_path}")
            self.report({"INFO"}, f"Retarget test bundle exported to {output_dir}.")
        except Exception as exc:
            self.report({"ERROR"}, f"Retarget bundle export failed: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}
        finally:
            try:
                if bpy.ops.object.mode_set.poll():
                    bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
            bpy.ops.object.select_all(action="DESELECT")
            for obj in original_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            if original_active and original_active.name in bpy.data.objects:
                context.view_layer.objects.active = original_active
            context.scene.frame_set(original_frame)

        return {"FINISHED"}


class StableGenCharacterizeRig(bpy.types.Operator):
    """Infer and store a semantic humanoid profile for the active rig."""

    bl_idname = "stablegen.characterize_rig"
    bl_label = "Analyze Rig"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def execute(self, context):
        armature = _active_or_selected_armature(context)
        profile = characterize_armature(armature)
        if not profile:
            self.report({"ERROR"}, "Could not infer a usable humanoid rig profile.")
            return {"CANCELLED"}

        _save_profile(armature, profile)
        if _rig_mapping_armature_name == armature.name:
            _ensure_rig_mapping_overlay(armature)
        print(f"[StableGen] Rig profile for {armature.name}: {_profile_summary(profile)}")
        self.report({"INFO"}, f"Rig profile stored on {armature.name}.")
        return {"FINISHED"}


class StableGenApplyRigPosePreset(bpy.types.Operator):
    """Pose the selected rig into a simple humanoid reference pose."""

    bl_idname = "stablegen.apply_rig_pose_preset"
    bl_label = "Pose Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(
        name="Pose",
        items=(
            ("REST", "Rest", "Reset all pose bones to the rig rest pose"),
            ("A_POSE", "A-Pose", "Pose arms down at an A-pose angle"),
            ("T_POSE", "T-Pose", "Pose arms horizontally into a T-pose"),
        ),
        default="A_POSE",
    )  # type: ignore

    strength: bpy.props.FloatProperty(
        name="Strength",
        min=0.0,
        max=1.0,
        default=1.0,
        description="How strongly to rotate mapped chains toward the preset",
    )  # type: ignore

    adjust_arms: bpy.props.BoolProperty(
        name="Arms",
        default=True,
        description="Adjust mapped arm chains",
    )  # type: ignore

    adjust_legs: bpy.props.BoolProperty(
        name="Legs",
        default=True,
        description="Adjust mapped leg chains using ankle direction",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def invoke(self, context, _event):
        if self.preset in {"A_POSE", "T_POSE"}:
            return context.window_manager.invoke_props_dialog(self, width=260)
        return self.execute(context)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "adjust_arms", toggle=True)
        layout.prop(self, "adjust_legs", toggle=True)

    def execute(self, context):
        armature = _active_or_selected_armature(context)
        profile = _profile_for_armature(armature)
        if not _profile_is_usable(armature, profile):
            self.report({"ERROR"}, "Analyze or edit the rig mapping before applying a pose preset.")
            return {"CANCELLED"}
        if self.preset in {"A_POSE", "T_POSE"} and not (self.adjust_arms or self.adjust_legs):
            self.report({"ERROR"}, "Enable arms, legs, or both.")
            return {"CANCELLED"}

        try:
            if bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        context.view_layer.objects.active = armature

        applied = _apply_rig_pose_preset(
            context,
            armature,
            profile,
            self.preset,
            strength=self.strength,
            adjust_arms=self.adjust_arms,
            adjust_legs=self.adjust_legs,
        )
        preset_name = _POSE_PRESET_LABELS.get(self.preset, self.preset)
        print(f"[StableGen] Applied {preset_name} to {armature.name}; adjusted {applied} chain(s).")
        if self.preset != "REST" and applied == 0:
            self.report({"WARNING"}, "Pose reset, but no mapped limb chains could be adjusted.")
            return {"FINISHED"}

        self.report({"INFO"}, f"{preset_name} applied to {armature.name}.")
        return {"FINISHED"}


class StableGenSetRetargetReferencePose(bpy.types.Operator):
    """Store the selected rig's current pose as the retarget reference pose."""

    bl_idname = "stablegen.set_retarget_reference_pose"
    bl_label = "Set Ref Pose"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def execute(self, context):
        armature = _active_or_selected_armature(context)
        profile = _profile_for_armature(armature)
        if not _profile_is_usable(armature, profile):
            self.report({"ERROR"}, "Analyze or edit the rig mapping before setting a reference pose.")
            return {"CANCELLED"}

        reference_pose = _capture_reference_pose(armature, label="stored_reference")
        if not reference_pose or not reference_pose.get("bones"):
            self.report({"ERROR"}, "Could not capture the current rig pose.")
            return {"CANCELLED"}

        _save_reference_pose(armature, reference_pose)
        print(
            f"[StableGen] Stored retarget reference pose for {armature.name} "
            f"({len(reference_pose['bones'])} bone(s))."
        )
        self.report({"INFO"}, f"Retarget reference pose stored on {armature.name}.")
        return {"FINISHED"}


class StableGenMatchSourceReferencePose(bpy.types.Operator):
    """Pose the target rig to match an animation source's reference frame."""

    bl_idname = "stablegen.match_source_reference_pose"
    bl_label = "Match Source Ref"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="Animation File",
        description="FBX or BVH file containing the source animation/reference pose",
        subtype="FILE_PATH",
    )  # type: ignore

    filter_glob: bpy.props.StringProperty(
        default="*.fbx;*.bvh",
        options={"HIDDEN"},
    )  # type: ignore

    match_strength: bpy.props.FloatProperty(
        name="Auto Match Strength",
        description="How strongly limb directions are matched to the source reference frame",
        default=1.0,
        min=0.0,
        max=1.0,
    )  # type: ignore

    store_reference: bpy.props.BoolProperty(
        name="Store As Ref Pose",
        description="Save the matched target pose as the retarget reference pose immediately",
        default=True,
    )  # type: ignore

    swap_target_sides: bpy.props.BoolProperty(
        name="Swap Target Left/Right",
        description="Use this if the detected target left/right side is inverted",
        default=True,
    )  # type: ignore

    keep_source_rig: bpy.props.BoolProperty(
        name="Keep Imported Source Rig",
        description="Leave the imported animation source armature in the scene for debugging",
        default=False,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "match_strength")
        layout.prop(self, "store_reference")
        layout.prop(self, "swap_target_sides")
        layout.prop(self, "keep_source_rig")

    def execute(self, context):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({"ERROR"}, "Select an existing FBX or BVH animation file.")
            return {"CANCELLED"}

        target_armature = _active_or_selected_armature(context)
        target_profile = _profile_for_armature(target_armature)
        if not _profile_is_usable(target_armature, target_profile):
            self.report({"ERROR"}, "Analyze or edit the target rig mapping first.")
            return {"CANCELLED"}

        source_armature = None
        imported_objects = []
        try:
            source_armature, imported_objects = _import_animation_source(self.filepath)
            source_profile = characterize_armature(source_armature)
            if not source_profile:
                raise RuntimeError("Could not infer a humanoid profile for the source animation rig")

            pairs = _common_chain_pairs(
                source_profile,
                target_profile,
                swap_target_sides=self.swap_target_sides,
                source_armature=source_armature,
                target_armature=target_armature,
                trim_feet=True,
            )
            if len(pairs) < 6:
                raise RuntimeError(
                    "Not enough source/target bone pairs were found. "
                    "Try analyzing/editing the target rig mapping first."
                )

            start, _end = _action_frame_range(source_armature, context.scene)
            context.scene.frame_set(start)
            context.view_layer.update()

            print(f"[StableGen] Source reference rig profile for {source_armature.name}: {_profile_summary(source_profile)}")
            print(f"[StableGen] Target rig profile for {target_armature.name}: {_profile_summary(target_profile)}")
            print(f"[StableGen] Reference match mapping: {_pair_summary(pairs)}")
            matched_chains = _match_source_reference_pose(
                context,
                source_armature,
                target_armature,
                pairs,
                source_profile,
                target_profile,
                strength=self.match_strength,
                swap_target_sides=self.swap_target_sides,
            )

            if self.store_reference:
                reference_pose = _capture_reference_pose(
                    target_armature,
                    label="matched_source_reference",
                )
                if not reference_pose or not reference_pose.get("bones"):
                    raise RuntimeError("Could not capture the matched target reference pose")
                _save_reference_pose(target_armature, reference_pose)

            print(
                f"[StableGen] Matched {matched_chains} limb chain(s) from source "
                f"reference frame {start} onto {target_armature.name}."
            )
            if self.store_reference:
                self.report({
                    "INFO"
                }, "Matched source reference pose and stored it. Adjust manually and click Set Ref Pose if needed.")
            else:
                self.report({
                    "INFO"
                }, "Matched source reference pose. Adjust manually, then click Set Ref Pose.")
        except Exception as exc:
            self.report({"ERROR"}, f"Reference match failed: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}
        finally:
            try:
                if bpy.ops.object.mode_set.poll():
                    bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
            if imported_objects and not self.keep_source_rig:
                _delete_objects(imported_objects)
            bpy.ops.object.select_all(action="DESELECT")
            if target_armature and target_armature.name in bpy.data.objects:
                target_armature.select_set(True)
                context.view_layer.objects.active = target_armature

        return {"FINISHED"}


class StableGenToggleRigMappingOverlay(bpy.types.Operator):
    """Show or hide a viewport overlay for the selected rig's chain mapping."""

    bl_idname = "stablegen.toggle_rig_mapping_overlay"
    bl_label = "Show Mapping"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if _rig_mapping_draw_handle is not None:
            return True
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def execute(self, context):
        if _rig_mapping_draw_handle is not None:
            _remove_rig_mapping_overlay()
            self.report({"INFO"}, "Rig mapping overlay hidden")
            return {"FINISHED"}

        armature = _active_or_selected_armature(context)
        profile = _profile_for_armature(armature)
        if not _profile_is_usable(armature, profile):
            self.report({"ERROR"}, "Could not infer a usable rig mapping.")
            return {"CANCELLED"}

        _ensure_rig_mapping_overlay(armature)
        print(f"[StableGen] Showing rig mapping overlay for {armature.name}: {_profile_summary(profile)}")
        self.report({"INFO"}, "Rig mapping overlay visible")
        return {"FINISHED"}


class StableGenEditRigMapping(bpy.types.Operator):
    """Manually edit and save the semantic bone-chain mapping for a rig."""

    bl_idname = "stablegen.edit_rig_mapping"
    bl_label = "Edit Rig Mapping"
    bl_options = {"REGISTER", "UNDO"}

    pelvis: bpy.props.EnumProperty(
        name="Pelvis",
        description="Root/pelvis bone used as the retarget root",
        items=_bone_enum_items,
    )  # type: ignore

    head: bpy.props.EnumProperty(
        name="Head",
        description="Head or top spine bone used for body-frame alignment",
        items=_bone_enum_items,
    )  # type: ignore

    spine_start: bpy.props.EnumProperty(
        name="Start",
        description="First spine bone",
        items=_bone_enum_items,
    )  # type: ignore

    spine_end: bpy.props.EnumProperty(
        name="End",
        description="Last spine/head bone",
        items=_bone_enum_items,
    )  # type: ignore

    left_arm_start: bpy.props.EnumProperty(
        name="Start",
        description="First left arm bone",
        items=_bone_enum_items,
    )  # type: ignore

    left_arm_end: bpy.props.EnumProperty(
        name="End",
        description="Last left arm/hand bone",
        items=_bone_enum_items,
    )  # type: ignore

    right_arm_start: bpy.props.EnumProperty(
        name="Start",
        description="First right arm bone",
        items=_bone_enum_items,
    )  # type: ignore

    right_arm_end: bpy.props.EnumProperty(
        name="End",
        description="Last right arm/hand bone",
        items=_bone_enum_items,
    )  # type: ignore

    left_leg_start: bpy.props.EnumProperty(
        name="Start",
        description="First left leg bone",
        items=_bone_enum_items,
    )  # type: ignore

    left_leg_end: bpy.props.EnumProperty(
        name="End",
        description="Last left leg/foot bone",
        items=_bone_enum_items,
    )  # type: ignore

    right_leg_start: bpy.props.EnumProperty(
        name="Start",
        description="First right leg bone",
        items=_bone_enum_items,
    )  # type: ignore

    right_leg_end: bpy.props.EnumProperty(
        name="End",
        description="Last right leg/foot bone",
        items=_bone_enum_items,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def invoke(self, context, event):
        armature = _active_or_selected_armature(context)
        profile = _load_profile(armature)
        if not _profile_is_usable(armature, profile):
            profile = characterize_armature(armature)
        _initialize_mapping_operator(self, armature, profile)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "pelvis")
        layout.prop(self, "head")
        layout.separator()

        for label, start_prop, end_prop in (
                ("Spine", "spine_start", "spine_end"),
                ("Left Arm", "left_arm_start", "left_arm_end"),
                ("Right Arm", "right_arm_start", "right_arm_end"),
                ("Left Leg", "left_leg_start", "left_leg_end"),
                ("Right Leg", "right_leg_start", "right_leg_end")):
            box = layout.box()
            box.label(text=label)
            row = box.row(align=True)
            row.prop(self, start_prop)
            row.prop(self, end_prop)

    def execute(self, context):
        armature = _active_or_selected_armature(context)
        if not armature:
            self.report({"ERROR"}, "Select a rigged mesh or armature.")
            return {"CANCELLED"}

        chains = {
            "spine": _bone_path_between(armature, self.spine_start, self.spine_end),
            "left_arm": _bone_path_between(armature, self.left_arm_start, self.left_arm_end),
            "right_arm": _bone_path_between(armature, self.right_arm_start, self.right_arm_end),
            "left_leg": _bone_path_between(armature, self.left_leg_start, self.left_leg_end),
            "right_leg": _bone_path_between(armature, self.right_leg_start, self.right_leg_end),
        }
        missing = [name for name, chain in chains.items() if len(chain) < 2]
        if missing:
            self.report({"ERROR"}, f"Invalid chain mapping: {', '.join(missing)}")
            return {"CANCELLED"}

        profile = {
            "version": _PROFILE_VERSION,
            "method": "manual",
            "armature": armature.name,
            "pelvis": self.pelvis,
            "head": self.head,
            "chains": chains,
        }
        if not _profile_is_usable(armature, profile):
            self.report({"ERROR"}, "Manual mapping references missing bones.")
            return {"CANCELLED"}

        _save_profile(armature, profile)
        if _rig_mapping_armature_name == armature.name:
            _ensure_rig_mapping_overlay(armature)
        print(f"[StableGen] Manual rig profile for {armature.name}: {_profile_summary(profile)}")
        self.report({"INFO"}, f"Manual rig mapping saved on {armature.name}.")
        return {"FINISHED"}


class StableGenApplyAnimationToRig(bpy.types.Operator):
    """Import an FBX/BVH animation source and bake it onto the selected rig."""

    bl_idname = "stablegen.apply_animation_to_rig"
    bl_label = "Apply Animation"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="Animation File",
        description="FBX or BVH file containing the source animation",
        subtype="FILE_PATH",
    )  # type: ignore

    filter_glob: bpy.props.StringProperty(
        default="*.fbx;*.bvh",
        options={"HIDDEN"},
    )  # type: ignore

    frame_step: bpy.props.IntProperty(
        name="Key Every N Frames",
        description="Bake keyframes at this interval",
        default=1,
        min=1,
        max=10,
    )  # type: ignore

    reference_mode: bpy.props.EnumProperty(
        name="Reference Pose",
        description="Target pose used as the neutral pose before animation deltas are applied",
        items=(
            ("SAVED", "Stored Ref Pose", "Use the explicit retarget reference pose stored on the target rig"),
            ("CURRENT", "Current Pose", "Use the target rig's current pose for this bake only"),
            ("REST", "Rig Rest Pose", "Use the armature rest pose/identity pose as the target reference"),
        ),
        default="SAVED",
    )  # type: ignore

    source_reference_mode: bpy.props.EnumProperty(
        name="Source Neutral",
        description="Source pose used as the animation's neutral baseline",
        items=(
            ("START_FRAME", "First Frame", "Treat the first animation frame as neutral and apply only motion deltas"),
            ("REST", "Source Rest Pose", "Use the source rig rest pose as neutral; useful for some clips but can over-rotate generated rigs"),
        ),
        default="START_FRAME",
    )  # type: ignore

    replace_action: bpy.props.BoolProperty(
        name="Replace Existing Action",
        description="Create a fresh action on the target armature before baking",
        default=True,
    )  # type: ignore

    copy_root_motion: bpy.props.BoolProperty(
        name="Copy Root Motion",
        description="Also transfer pelvis/root translation. Disable if the character teleports or drifts",
        default=False,
    )  # type: ignore

    swap_target_sides: bpy.props.BoolProperty(
        name="Swap Target Left/Right",
        description="Use this if the first preview has left and right limbs crossed",
        default=True,
    )  # type: ignore

    flip_target_forward: bpy.props.BoolProperty(
        name="Flip Target Forward",
        description="Use this if forward/back motion is inverted on the target rig",
        default=True,
    )  # type: ignore

    keep_source_rig: bpy.props.BoolProperty(
        name="Keep Imported Source Rig",
        description="Leave the imported animation armature in the scene for debugging",
        default=False,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "reference_mode")
        layout.prop(self, "source_reference_mode")
        layout.prop(self, "frame_step")
        layout.prop(self, "replace_action")
        layout.prop(self, "copy_root_motion")
        layout.prop(self, "swap_target_sides")
        layout.prop(self, "flip_target_forward")
        layout.prop(self, "keep_source_rig")

    def execute(self, context):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({"ERROR"}, "Select an existing FBX or BVH animation file.")
            return {"CANCELLED"}

        target_armature = _active_or_selected_armature(context)
        target_profile = _profile_for_armature(target_armature)
        if not target_profile:
            self.report({"ERROR"}, "Could not infer the target rig profile.")
            return {"CANCELLED"}

        if self.reference_mode == "SAVED" and not _load_reference_pose(target_armature):
            self.report({
                "ERROR"
            }, "No retarget reference pose is stored. Pose the rig, then use Set Ref Pose.")
            return {"CANCELLED"}

        current_reference_pose = None
        if self.reference_mode == "CURRENT":
            context.view_layer.update()
            current_reference_pose = _capture_reference_pose(target_armature, label="current_pose")

        original_active = context.view_layer.objects.active
        original_selection = list(context.selected_objects)
        original_frame = context.scene.frame_current

        source_armature = None
        imported_objects = []
        try:
            source_armature, imported_objects = _import_animation_source(self.filepath)
            source_profile = characterize_armature(source_armature)
            if not source_profile:
                raise RuntimeError("Could not infer a humanoid profile for the source animation rig")

            pairs = _common_chain_pairs(
                source_profile,
                target_profile,
                swap_target_sides=self.swap_target_sides,
                source_armature=source_armature,
                target_armature=target_armature,
                trim_feet=True,
            )
            if len(pairs) < 6:
                raise RuntimeError(
                    "Not enough source/target bone pairs were found. "
                    "Try analyzing the rig or using a different source animation."
                )
            print(f"[StableGen] Source rig profile for {source_armature.name}: {_profile_summary(source_profile)}")
            print(f"[StableGen] Target rig profile for {target_armature.name}: {_profile_summary(target_profile)}")
            print(f"[StableGen] Retarget mapping: {_pair_summary(pairs)}")
            print(f"[StableGen] {_axis_summary('Source', source_armature, source_profile)}")
            print(
                f"[StableGen] {_axis_summary('Target', target_armature, target_profile, self.flip_target_forward)}"
            )
            if self.flip_target_forward:
                print("[StableGen] Retarget option: target forward axis flipped.")

            target_names = _mapped_target_bones(pairs)
            if self.reference_mode == "SAVED":
                target_reference_pose = _load_reference_pose(target_armature)
                reference_label = "stored reference pose"
            elif self.reference_mode == "CURRENT":
                target_reference_pose = current_reference_pose or _capture_reference_pose(
                    target_armature,
                    label="current_pose",
                )
                reference_label = "current target pose"
            else:
                target_reference_pose = _rest_reference_pose(target_armature, target_names)
                reference_label = "rig rest pose"

            if not _reference_pose_is_usable(target_armature, target_reference_pose, target_names):
                raise RuntimeError(
                    "The selected target rig does not have a usable retarget reference pose. "
                    "Pose the rig in the desired neutral pose and click Set Ref Pose, "
                    "or choose Current Pose / Rig Rest Pose in the Apply Animation options."
                )
            print(f"[StableGen] Target retarget reference: {reference_label}")
            source_reference_label = (
                "source first animation frame"
                if self.source_reference_mode == "START_FRAME"
                else "source rig rest pose"
            )
            print(f"[StableGen] Source retarget neutral: {source_reference_label}")

            start, end = _action_frame_range(source_armature, context.scene)
            action_name = (
                f"{target_armature.name}_"
                f"{os.path.splitext(os.path.basename(self.filepath))[0]}"
            )
            target_armature.animation_data_create()
            if self.replace_action or not target_armature.animation_data.action:
                target_armature.animation_data.action = bpy.data.actions.new(action_name)

            bpy.ops.object.select_all(action="DESELECT")
            target_armature.select_set(True)
            context.view_layer.objects.active = target_armature
            context.view_layer.update()
            _clear_retarget_constraints(target_armature)
            _bake_pose_delta_retarget(
                context,
                source_armature,
                target_armature,
                pairs,
                source_profile,
                target_profile,
                start,
                end,
                self.frame_step,
                copy_root_motion=self.copy_root_motion,
                target_reference_pose=target_reference_pose,
                flip_target_forward=self.flip_target_forward,
                source_reference_mode=self.source_reference_mode,
            )

            context.scene.frame_set(start)
            print(
                f"[StableGen] Retargeted {len(pairs)} bone pair(s) "
                f"from {source_armature.name} to {target_armature.name}; "
                f"baked pose-delta frames {start}-{end}."
            )
            self.report({"INFO"}, f"Animation baked to {target_armature.name}.")
        except Exception as exc:
            _clear_retarget_constraints(target_armature)
            self.report({"ERROR"}, f"Animation retarget failed: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}
        finally:
            try:
                if bpy.ops.object.mode_set.poll():
                    bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
            if imported_objects and not self.keep_source_rig:
                _delete_objects(imported_objects)
            bpy.ops.object.select_all(action="DESELECT")
            for obj in original_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            if original_active and original_active.name in bpy.data.objects:
                context.view_layer.objects.active = original_active
            context.scene.frame_set(original_frame)

        return {"FINISHED"}
