"""Humanoid animation retargeting for generated SkinTokens rigs.

This module intentionally treats generated armatures as arbitrary skeletons:
bone names may be generic (``bone_0``), so the first step is to infer a
semantic profile from the rest pose and store it on the armature.
"""

import hashlib
import json
import math
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
_SOURCE_PREVIEW_PROP = "sg_retarget_source_preview"
_COLLISION_PROXY_PROP = "sg_retarget_collision_proxy"
_COLLISION_SEGMENT_MODE_PROP = "sg_retarget_collision_segment_mode"
_UPPER_ARM_COLLISION_PROP = "sg_retarget_upper_arm_collision"
_ARM_CORRECTION_DRIVER_PROP = "sg_retarget_arm_correction_driver"
_HAND_MATCH_ENABLED_PROP = "sg_retarget_match_hand_orientation"
_HAND_MATCH_STRENGTH_PROP = "sg_retarget_hand_match_strength"
_HAND_ROLL_MODE_PROP = "sg_retarget_hand_roll_mode"
_HAND_ROLL_LIMIT_PROP = "sg_retarget_hand_roll_limit"
_TWIST_CHAIN_VERSION_PROP = "sg_retarget_twist_chain_version"
_PROFILE_VERSION = 14
_REFERENCE_POSE_VERSION = 2
_COLLISION_PROXY_VERSION = 3
_HAND_MATCH_ALGORITHM_VERSION = 4
# 2: AUTO corrects from the upper arm when the arm chain starts at a clavicle. Reference poses
# matched by version 1 carry a shrug in the clavicles, so they are re-matched.
_ARM_CORRECTION_ALGORITHM_VERSION = 2
_TWIST_CHAIN_VERSION = 1
_LEGACY_REFERENCE_POSE_VERSION = 1
_CONSTRAINT_PREFIX = "SG_Retarget_"
_BONE_ENUM_ITEMS_CACHE = {}
_rig_mapping_draw_handle = None
_rig_mapping_label_draw_handle = None
_rig_mapping_armature_name = None
_collision_proxy_draw_handle = None
_collision_proxy_armature_name = None

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

_HAND_LABELS = {
    "left_arm": "L Hand",
    "right_arm": "R Hand",
}

_BASIS_COLORS = {
    "forward": (1.0, 0.25, 0.95, 0.98),
    "left": (0.15, 0.55, 1.0, 0.98),
}

_BASIS_LABELS = {
    "forward": "Forward",
    "left": "Left",
}

_COLLISION_PROXY_COLORS = {
    "body": (1.0, 0.2, 0.35, 0.82),
    "left_arm": (0.2, 0.95, 0.55, 0.82),
    "right_arm": (1.0, 0.55, 0.16, 0.82),
    "inactive": (0.55, 0.55, 0.6, 0.38),
    "driver": (0.3, 0.65, 1.0, 0.62),
}

_ARM_CORRECTION_DRIVER_ITEMS = (
    (
        "AUTO",
        "Automatic (Recommended)",
        "Use the upper arm when the first arm bone is a clavicle or a disconnected mechanical "
        "shoulder, so corrections do not shrug the shoulder; otherwise the first arm bone",
    ),
    (
        "SHOULDER",
        "Shoulder / Clavicle",
        "Apply direction and collision correction from the first mapped arm bone",
    ),
    (
        "UPPER_ARM",
        "Upper Arm",
        "Preserve the clavicle and apply direction and collision correction from the upper arm",
    ),
)

_COLLISION_SEGMENT_MODE_ITEMS = (
    (
        "BONE_TAILS",
        "Bone Heads to Tails",
        "Build capsules from each Blender bone head to its raw tail",
    ),
    (
        "JOINT_HEADS",
        "Joint to Joint",
        "Build capsules between consecutive mapped joint heads; recommended for disconnected rigs",
    ),
)

_UPPER_ARM_COLLISION_ITEMS = (
    (
        "DISABLED",
        "Disabled",
        "Do not use the upper-arm capsule for collision detection",
    ),
    (
        "DISTAL",
        "Distal 75% (Recommended)",
        "Check the upper arm while excluding its proximal shoulder-attachment region",
    ),
    (
        "FULL",
        "Full Upper Arm",
        "Check the complete upper-arm capsule and preserve reference-pose intersections",
    ),
)

_HAND_ROLL_MODE_ITEMS = (
    (
        "FOREARM",
        "Single Forearm",
        "Apply palm roll to the mapped forearm and leave only flexion at the wrist",
    ),
    (
        "TWIST_CHAIN",
        "Generated Twist Chain",
        "Create deform helpers and distribute palm roll gradually along the forearm",
    ),
    (
        "LIMITED",
        "Limited Roll",
        "Limit transferred palm roll to protect meshes that cannot tolerate full torsion",
    ),
    (
        "WRIST",
        "Wrist Only",
        "Apply the complete palm correction directly to the hand bone",
    ),
)


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
    chains = _profile_chains(profile)
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


def _profile_forward_flipped(profile):
    return (profile or {}).get("orientation_correction") == "ROTATE_180"


def _profile_chains(profile, oriented=True):
    chains = dict((profile or {}).get("chains", {}))
    if oriented and _profile_forward_flipped(profile):
        for suffix in ("arm", "leg"):
            left_key = f"left_{suffix}"
            right_key = f"right_{suffix}"
            chains[left_key], chains[right_key] = (
                chains.get(right_key, []),
                chains.get(left_key, []),
            )
    return chains


def _profile_signature(profile):
    if not profile:
        return None
    payload = {
        "version": profile.get("version"),
        "pelvis": profile.get("pelvis"),
        "head": profile.get("head"),
        "chains": profile.get("chains", {}),
        "orientation_correction": profile.get("orientation_correction", "NORMAL"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _animation_file_identity(filepath):
    path = os.path.normcase(os.path.abspath(bpy.path.abspath(filepath))) if filepath else None
    result = {"path": path}
    if path and os.path.exists(path):
        stat = os.stat(path)
        result["size"] = int(stat.st_size)
        result["mtime_ns"] = int(stat.st_mtime_ns)
    return result


def _reference_provenance(
        filepath, source_profile, target_profile, source_frame,
        source_reference_mode, swap_target_sides=False,
        match_hand_orientation=False, hand_match_strength=1.0,
        hand_roll_mode="FOREARM", hand_roll_limit=60.0,
        arm_correction_driver="AUTO"):
    return {
        "animation": _animation_file_identity(filepath),
        "arm_correction_driver": arm_correction_driver,
        "arm_correction_algorithm_version": _ARM_CORRECTION_ALGORITHM_VERSION,
        "source_frame": int(source_frame),
        "source_reference_mode": source_reference_mode,
        "source_profile_signature": _profile_signature(source_profile),
        "target_profile_signature": _profile_signature(target_profile),
        "swap_target_sides": bool(swap_target_sides),
        "match_hand_orientation": bool(match_hand_orientation),
        "hand_match_strength": (
            round(float(hand_match_strength), 6)
            if match_hand_orientation else None
        ),
        "hand_match_algorithm_version": (
            _HAND_MATCH_ALGORITHM_VERSION
            if match_hand_orientation else None
        ),
        "hand_roll_mode": hand_roll_mode if match_hand_orientation else None,
        "hand_roll_limit": (
            round(float(hand_roll_limit), 4)
            if match_hand_orientation and hand_roll_mode == "LIMITED"
            else None
        ),
    }


def _reference_provenance_matches(reference_pose, expected):
    actual = (reference_pose or {}).get("provenance")
    if not actual or not expected:
        return False, "target reference has no source provenance"
    for key in (
            "source_frame", "source_reference_mode", "source_profile_signature",
            "target_profile_signature", "swap_target_sides",
            "match_hand_orientation", "hand_match_strength",
            "hand_match_algorithm_version", "hand_roll_mode",
            "hand_roll_limit", "arm_correction_driver",
            "arm_correction_algorithm_version"):
        default = False if key == "match_hand_orientation" else None
        if actual.get(key, default) != expected.get(key):
            return False, f"target reference {key} does not match"
    actual_file = actual.get("animation", {})
    expected_file = expected.get("animation", {})
    if actual_file.get("path") != expected_file.get("path"):
        return False, "target reference was made from another animation file"
    for key in ("size", "mtime_ns"):
        if key in actual_file and key in expected_file and actual_file[key] != expected_file[key]:
            return False, "source animation file changed after the target reference was stored"
    return True, "reference provenance matches"


def _rows_to_matrix(rows):
    return mathutils.Matrix(rows)


def _capture_reference_pose(
        armature, bone_names=None, label="current_pose", provenance=None):
    if not armature or armature.type != "ARMATURE":
        return None
    names = list(bone_names) if bone_names else [bone.name for bone in armature.pose.bones]
    matrices = {}
    for name in names:
        pose_bone = armature.pose.bones.get(name)
        if not pose_bone:
            continue
        matrices[name] = _matrix_to_rows(pose_bone.matrix_basis.copy())
    result = {
        "version": _REFERENCE_POSE_VERSION,
        "armature": armature.name,
        "label": label,
        "bones": matrices,
    }
    if provenance:
        result["provenance"] = provenance
    return result


def _rest_reference_pose(armature, bone_names):
    identity = _matrix_to_rows(mathutils.Matrix.Identity(4))
    return {
        "version": _REFERENCE_POSE_VERSION,
        "armature": armature.name,
        "label": "rest_pose",
        "bones": {name: identity for name in bone_names},
        "provenance": {"source_reference_mode": "REST"},
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
    if reference_pose.get("version") not in {
            _LEGACY_REFERENCE_POSE_VERSION, _REFERENCE_POSE_VERSION}:
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


def _active_collision_proxy_armature():
    if not _collision_proxy_armature_name:
        return None
    obj = bpy.data.objects.get(_collision_proxy_armature_name)
    if obj and obj.type == "ARMATURE":
        return obj
    return None


def _chain_world_points(armature, chain):
    points = []
    for bone_name in chain:
        pose_bone = armature.pose.bones.get(bone_name)
        if not pose_bone:
            continue
        points.append(armature.matrix_world @ pose_bone.head)
    return points


def _hand_mapping_overlay_data(armature, profile):
    """Return mapped palm/digit segments without assuming exactly five digits."""
    result = []
    world_matrix = armature.matrix_world
    for role in ("left_arm", "right_arm"):
        chain = _profile_chains(profile).get(role, [])
        if not chain:
            continue
        hand_name = chain[-1]
        hand_pose = armature.pose.bones.get(hand_name)
        if not hand_pose:
            continue

        segments = [
            world_matrix @ hand_pose.head,
            world_matrix @ hand_pose.tail,
        ]
        label_points = [world_matrix @ hand_pose.tail]
        digit_roots = _hand_digit_root_names(armature, hand_name)
        visited = set()
        stack = []
        for root_name in digit_roots:
            cursor = armature.data.bones.get(root_name)
            while cursor and cursor.name != hand_name:
                stack.append(cursor.name)
                cursor = cursor.parent
        while stack:
            bone_name = stack.pop()
            if bone_name in visited:
                continue
            visited.add(bone_name)
            pose_bone = armature.pose.bones.get(bone_name)
            data_bone = armature.data.bones.get(bone_name)
            if not pose_bone or not data_bone:
                continue
            head = world_matrix @ pose_bone.head
            tail = world_matrix @ pose_bone.tail
            if pose_bone.parent:
                parent_tail = world_matrix @ pose_bone.parent.tail
                if (head - parent_tail).length > 1e-6:
                    segments.extend((parent_tail, head))
            segments.extend((head, tail))
            label_points.extend((head, tail))
            stack.extend(child.name for child in data_bone.children)

        center = sum(
            label_points,
            mathutils.Vector((0.0, 0.0, 0.0)),
        ) / len(label_points)
        result.append({
            "role": role,
            "hand": hand_name,
            "segments": segments,
            "center": center,
            "digit_count": len(digit_roots),
        })
    return result


def _profile_world_points(armature, profile):
    points = []
    chains = _profile_chains(profile) if profile else {}
    names = {profile.get("pelvis"), profile.get("head")} if profile else set()
    for chain in chains.values():
        names.update(chain)

    for name in names:
        pose_bone = armature.pose.bones.get(name) if name else None
        if not pose_bone:
            continue
        points.append(armature.matrix_world @ pose_bone.head)
        points.append(armature.matrix_world @ pose_bone.tail)
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
    pelvis = _pose_point(armature, profile.get("pelvis"), use_tail=False)
    if pelvis is None:
        return {}

    basis = _pose_body_basis(armature, profile)
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


def _rig_mapping_frame_change(_scene, _depsgraph=None):
    if (
            _rig_mapping_draw_handle is not None
            or _collision_proxy_draw_handle is not None):
        _tag_view3d_redraw()


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
        for chain_name, chain in _profile_chains(profile).items():
            points = _chain_world_points(armature, chain)
            if len(points) < 2:
                continue
            shader.uniform_float(
                "color",
                _CHAIN_COLORS.get(chain_name, (1.0, 1.0, 1.0, 0.9)),
            )
            batch = batch_for_shader(shader, "LINE_STRIP", {"pos": points})
            batch.draw(shader)

        gpu.state.line_width_set(2.2)
        for hand_data in _hand_mapping_overlay_data(armature, profile):
            segments = hand_data["segments"]
            if len(segments) < 2:
                continue
            shader.uniform_float(
                "color",
                _CHAIN_COLORS.get(
                    hand_data["role"], (1.0, 1.0, 1.0, 0.9)),
            )
            batch = batch_for_shader(shader, "LINES", {"pos": segments})
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
        for chain_name, chain in _profile_chains(profile).items():
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

        for hand_data in _hand_mapping_overlay_data(armature, profile):
            co_2d = view3d_utils.location_3d_to_region_2d(
                region, rv3d, hand_data["center"])
            if co_2d is None:
                continue
            label = _HAND_LABELS.get(hand_data["role"], "Hand")
            color = _CHAIN_COLORS.get(
                hand_data["role"], (1.0, 1.0, 1.0, 0.9))
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
    if _rig_mapping_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_rig_mapping_frame_change)
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
    if (
            _collision_proxy_draw_handle is None
            and _rig_mapping_frame_change in bpy.app.handlers.frame_change_post):
        bpy.app.handlers.frame_change_post.remove(_rig_mapping_frame_change)
    _rig_mapping_armature_name = None
    _tag_view3d_redraw()


def _cached_collision_proxy(armature):
    raw = armature.get(_COLLISION_PROXY_PROP) if armature else None
    if not raw:
        return None
    try:
        proxy = json.loads(raw)
    except Exception:
        return None
    if proxy.get("version") != _COLLISION_PROXY_VERSION:
        return None
    return proxy


def _capsule_wire_segments(
        start, end, radius, world_matrix, ring_sides=12, cap_steps=3):
    axis = end - start
    if radius <= 1e-6:
        return []
    if axis.length < 1e-6:
        segments = []
        circle_axes = (
            (mathutils.Vector((1.0, 0.0, 0.0)), mathutils.Vector((0.0, 1.0, 0.0))),
            (mathutils.Vector((1.0, 0.0, 0.0)), mathutils.Vector((0.0, 0.0, 1.0))),
            (mathutils.Vector((0.0, 1.0, 0.0)), mathutils.Vector((0.0, 0.0, 1.0))),
        )
        for side_a, side_b in circle_axes:
            points = []
            for side_index in range(ring_sides):
                angle = (math.pi * 2.0) * side_index / ring_sides
                point = start + (
                    side_a * (math.cos(angle) * radius)
                    + side_b * (math.sin(angle) * radius)
                )
                points.append(world_matrix @ point)
            for side_index, point in enumerate(points):
                segments.extend([point, points[(side_index + 1) % ring_sides]])
        return segments
    axis.normalize()
    helper = mathutils.Vector((0.0, 0.0, 1.0))
    if abs(axis.dot(helper)) > 0.9:
        helper = mathutils.Vector((1.0, 0.0, 0.0))
    side_a = axis.cross(helper).normalized()
    side_b = axis.cross(side_a).normalized()

    rings = []
    cap_steps = max(2, int(cap_steps))
    for step in range(cap_steps + 1):
        angle = (-math.pi * 0.5) + ((math.pi * 0.5) * step / cap_steps)
        rings.append((start + (axis * (math.sin(angle) * radius)), math.cos(angle) * radius))
    for step in range(cap_steps + 1):
        angle = (math.pi * 0.5) * step / cap_steps
        rings.append((end + (axis * (math.sin(angle) * radius)), math.cos(angle) * radius))

    ring_points = []
    for center, ring_radius in rings:
        points = []
        for side_index in range(ring_sides):
            angle = (math.pi * 2.0) * side_index / ring_sides
            point = center + (
                side_a * (math.cos(angle) * ring_radius)
                + side_b * (math.sin(angle) * ring_radius)
            )
            points.append(world_matrix @ point)
        ring_points.append(points)

    segments = []
    for ring_index, points in enumerate(ring_points):
        if rings[ring_index][1] > 1e-6:
            for side_index, point in enumerate(points):
                segments.extend([point, points[(side_index + 1) % ring_sides]])
        if ring_index:
            previous = ring_points[ring_index - 1]
            for side_index, point in enumerate(points):
                segments.extend([previous[side_index], point])
    segments.extend([world_matrix @ start, world_matrix @ end])
    return segments


def _collision_proxy_bone_groups(proxy, armature=None, profile=None):
    groups = []
    seen = set()
    for name in proxy.get("body_bones", []):
        if name not in seen:
            seen.add(name)
            groups.append((name, "body", 0.0))
    upper_arm_collision = (
        armature.get(_UPPER_ARM_COLLISION_PROP, "DISABLED")
        if armature else "DISABLED"
    )
    arm_correction_driver = (
        armature.get(_ARM_CORRECTION_DRIVER_PROP, "AUTO")
        if armature else "AUTO"
    )
    for role in ("left_arm", "right_arm"):
        chain = proxy.get("arm_chains", {}).get(role, [])
        active_bone_list, start_fractions = _arm_collision_probe_spec(
            chain, upper_arm_collision)
        active_bones = set(active_bone_list)
        correction_driver_bone = None
        if armature and profile:
            correction_chain = _arm_correction_chain(
                armature,
                profile,
                role,
                chain,
                driver_mode=arm_correction_driver,
            )
            if correction_chain:
                correction_driver_bone = correction_chain[0]
        for name in chain:
            if name not in seen:
                seen.add(name)
                if name in active_bones:
                    display_role = role
                elif name == correction_driver_bone:
                    display_role = "driver"
                else:
                    display_role = "inactive"
                groups.append((
                    name,
                    display_role,
                    float(start_fractions.get(name, 0.0)),
                ))
    return groups


def _draw_collision_proxy_overlay():
    armature = _active_collision_proxy_armature()
    proxy = _cached_collision_proxy(armature)
    if not armature or not proxy:
        return
    profile = _load_profile(armature)

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(1.6)
    try:
        for bone_name, role, start_fraction in _collision_proxy_bone_groups(
                proxy, armature=armature, profile=profile):
            capsule = _proxy_pose_capsule(armature, proxy, bone_name)
            if not capsule:
                continue
            if start_fraction > 0.0:
                capsule["start"] = capsule["start"].lerp(
                    capsule["end"], start_fraction)
            segments = _capsule_wire_segments(
                capsule["start"],
                capsule["end"],
                capsule["radius"],
                armature.matrix_world,
            )
            if not segments:
                continue
            shader.uniform_float(
                "color",
                _COLLISION_PROXY_COLORS.get(role, (0.8, 0.8, 0.8, 0.8)),
            )
            batch = batch_for_shader(shader, "LINES", {"pos": segments})
            batch.draw(shader)
    finally:
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")
        gpu.state.line_width_set(1.0)


def _ensure_collision_proxy_overlay(armature):
    global _collision_proxy_draw_handle, _collision_proxy_armature_name

    _collision_proxy_armature_name = armature.name
    if _collision_proxy_draw_handle is None:
        _collision_proxy_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_collision_proxy_overlay, (), "WINDOW", "POST_VIEW")
    if _rig_mapping_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_rig_mapping_frame_change)
    _tag_view3d_redraw()


def _remove_collision_proxy_overlay():
    global _collision_proxy_draw_handle, _collision_proxy_armature_name

    if _collision_proxy_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(
            _collision_proxy_draw_handle, "WINDOW")
        _collision_proxy_draw_handle = None
    if (
            _rig_mapping_draw_handle is None
            and _rig_mapping_frame_change in bpy.app.handlers.frame_change_post):
        bpy.app.handlers.frame_change_post.remove(_rig_mapping_frame_change)
    _collision_proxy_armature_name = None
    _tag_view3d_redraw()


def cleanup_retarget_overlays():
    """Remove viewport handlers before the add-on is unregistered or reloaded."""
    _remove_collision_proxy_overlay()
    _remove_rig_mapping_overlay()


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
    if hasattr(operator, "rotate_180"):
        operator.rotate_180 = _profile_forward_flipped(profile)

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


def _initialize_collision_operator_settings(operator, context):
    armature = _active_or_selected_armature(context)
    if not armature:
        return
    stored_values = (
        (
            "collision_segment_mode",
            _COLLISION_SEGMENT_MODE_PROP,
            {"BONE_TAILS", "JOINT_HEADS"},
        ),
        (
            "upper_arm_collision",
            _UPPER_ARM_COLLISION_PROP,
            {"DISABLED", "DISTAL", "FULL"},
        ),
        (
            "arm_correction_driver",
            _ARM_CORRECTION_DRIVER_PROP,
            {"AUTO", "SHOULDER", "UPPER_ARM"},
        ),
    )
    for attribute, property_name, valid_values in stored_values:
        value = armature.get(property_name)
        if hasattr(operator, attribute) and value in valid_values:
            setattr(operator, attribute, value)
    if (
            hasattr(operator, "match_hand_orientation")
            and _HAND_MATCH_ENABLED_PROP in armature):
        operator.match_hand_orientation = bool(
            armature[_HAND_MATCH_ENABLED_PROP])
    if (
            hasattr(operator, "hand_match_strength")
            and _HAND_MATCH_STRENGTH_PROP in armature):
        operator.hand_match_strength = max(
            0.0,
            min(1.0, float(armature[_HAND_MATCH_STRENGTH_PROP])),
        )
    if (
            hasattr(operator, "hand_roll_mode")
            and armature.get(_HAND_ROLL_MODE_PROP) in {
                "WRIST", "FOREARM", "TWIST_CHAIN", "LIMITED"}):
        operator.hand_roll_mode = armature[_HAND_ROLL_MODE_PROP]
    if (
            hasattr(operator, "hand_roll_limit")
            and _HAND_ROLL_LIMIT_PROP in armature):
        operator.hand_roll_limit = max(
            0.0,
            min(180.0, float(armature[_HAND_ROLL_LIMIT_PROP])),
        )


def _common_chain_pairs(
        source_profile, target_profile, swap_target_sides=False,
        source_armature=None, target_armature=None, trim_feet=False):
    source_chains = _profile_chains(source_profile)
    target_chains = _profile_chains(target_profile)
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
    chains = _profile_chains(profile)
    names = set([profile.get("pelvis"), profile.get("head")])
    for chain in chains.values():
        names.update(chain)

    terminal_arm_bones = {
        chain[-1]
        for role, chain in chains.items()
        if role in {"left_arm", "right_arm"} and chain
    }
    up_axis = _profile_up_axis(armature, profile)

    heights = []
    for name in names:
        bone = armature.data.bones.get(name)
        if not bone:
            continue
        heights.append(bone.head_local.dot(up_axis))
        if name not in terminal_arm_bones:
            heights.append(bone.tail_local.dot(up_axis))
    if not heights:
        return 1.0

    return max(max(heights) - min(heights), 1e-5)


def _target_skin_meshes(armature):
    return [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and _mesh_armature(obj) == armature
    ]


def _collision_mesh_signature(armature):
    payload = []
    for obj in sorted(_target_skin_meshes(armature), key=lambda item: item.name):
        payload.append({
            "name": obj.name,
            "data": obj.data.name,
            "vertices": len(obj.data.vertices),
            "groups": sorted(group.name for group in obj.vertex_groups),
            "matrix_world": [
                [round(float(value), 6) for value in row]
                for row in obj.matrix_world
            ],
            "bounds": [
                [round(float(value), 6) for value in corner]
                for corner in obj.bound_box
            ],
        })
    signature_payload = {
        "armature_matrix_world": [
            [round(float(value), 6) for value in row]
            for row in armature.matrix_world
        ],
        "meshes": payload,
    }
    encoded = json.dumps(
        signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _point_segment_distance(point, start, end):
    segment = end - start
    length_squared = segment.length_squared
    if length_squared < 1e-12:
        return (point - start).length, 0.0
    parameter = (point - start).dot(segment) / length_squared
    closest = start + (segment * max(0.0, min(1.0, parameter)))
    return (point - closest).length, parameter


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * max(0.0, min(1.0, fraction))))
    return ordered[index]


def _proxy_bone_role(bone_name, target_profile, chains):
    if bone_name == target_profile.get("head"):
        return "head"
    if bone_name == target_profile.get("pelvis") or bone_name in chains.get("spine", []):
        return "body"
    for role in ("left_arm", "right_arm"):
        if bone_name in chains.get(role, []):
            return "arm"
    return "limb"


def _default_proxy_radius(bone, role, target_height):
    bone_length = max(float(bone.length), target_height * 0.01)
    if role == "head":
        return max(target_height * 0.055, bone_length * 0.45)
    if role == "body":
        return max(target_height * 0.065, bone_length * 0.32)
    return max(target_height * 0.016, bone_length * 0.16)


def _proxy_radius_limits(bone, role, target_height):
    bone_length = max(float(bone.length), target_height * 0.01)
    if role == "head":
        return target_height * 0.025, max(target_height * 0.12, bone_length * 0.8)
    if role == "body":
        return target_height * 0.035, max(target_height * 0.20, bone_length * 0.75)
    return target_height * 0.010, max(target_height * 0.075, bone_length * 0.42)


def _build_collision_proxy(
        armature, target_profile, segment_mode="BONE_TAILS"):
    """Estimate a compact rest-space capsule rig from dominant skin weights."""
    chains = _profile_chains(target_profile)
    body_bones = []
    for name in [
            target_profile.get("pelvis"),
            *chains.get("spine", []),
            target_profile.get("head")]:
        if name and name not in body_bones and armature.data.bones.get(name):
            body_bones.append(name)

    arm_chains = {}
    for role in ("left_arm", "right_arm"):
        arm_chains[role] = [
            name for name in chains.get(role, [])
            if armature.data.bones.get(name)
        ]
    terminal_arm_bones = [
        chain[-1] for chain in arm_chains.values() if chain
    ]
    segment_end_bones = {}
    for chain in [body_bones, *arm_chains.values()]:
        for name, next_name in zip(chain, chain[1:]):
            segment_end_bones[name] = next_name

    proxy_bones = list(body_bones)
    for chain in arm_chains.values():
        for name in chain:
            if name not in proxy_bones:
                proxy_bones.append(name)
    if not body_bones or any(len(chain) < 2 for chain in arm_chains.values()):
        return None

    radial_samples = {name: [] for name in proxy_bones}
    terminal_head_samples = {name: [] for name in terminal_arm_bones}
    target_height = _height_from_profile(armature, target_profile)
    armature_inverse = armature.matrix_world.inverted_safe()
    sampled_vertices = 0
    meshes = _target_skin_meshes(armature)

    for obj in meshes:
        if not obj.vertex_groups or not obj.data.vertices:
            continue
        group_names = {
            group.index: group.name
            for group in obj.vertex_groups
            if group.name in radial_samples
        }
        if not group_names:
            continue
        to_armature = armature_inverse @ obj.matrix_world
        stride = max(1, len(obj.data.vertices) // 25000)
        for vertex_index, vertex in enumerate(obj.data.vertices):
            if vertex_index % stride:
                continue
            dominant_name = None
            dominant_weight = 0.0
            for membership in vertex.groups:
                name = group_names.get(membership.group)
                if name and membership.weight > dominant_weight:
                    dominant_name = name
                    dominant_weight = float(membership.weight)
            if dominant_name is None or dominant_weight < 0.05:
                continue
            bone = armature.data.bones.get(dominant_name)
            if not bone:
                continue
            point = to_armature @ vertex.co
            segment_end = bone.tail_local
            if (
                    segment_mode == "JOINT_HEADS"
                    and dominant_name not in terminal_arm_bones):
                next_name = segment_end_bones.get(dominant_name)
                next_bone = (
                    armature.data.bones.get(next_name)
                    if next_name else None
                )
                if next_bone:
                    segment_end = next_bone.head_local
            distance, parameter = _point_segment_distance(
                point, bone.head_local, segment_end)
            if parameter < -0.35 or parameter > 1.35:
                continue
            radial_samples[dominant_name].append(float(distance))
            if (
                    dominant_name in terminal_head_samples
                    and -0.35 <= parameter <= 0.45):
                terminal_head_samples[dominant_name].append(
                    float((point - bone.head_local).length))
            sampled_vertices += 1

    radii = {}
    sample_counts = {}
    for name in proxy_bones:
        bone = armature.data.bones.get(name)
        role = _proxy_bone_role(name, target_profile, chains)
        estimate = _percentile(radial_samples[name], 0.88)
        if estimate is None:
            estimate = _default_proxy_radius(bone, role, target_height)
        minimum, maximum = _proxy_radius_limits(bone, role, target_height)
        if name in terminal_head_samples:
            hand_estimate = _percentile(terminal_head_samples[name], 0.88)
            if hand_estimate is not None:
                estimate = hand_estimate
            minimum = target_height * 0.012
            maximum = target_height * 0.055
        radii[name] = float(max(minimum, min(maximum, estimate)))
        sample_counts[name] = len(radial_samples[name])

    proxy = {
        "version": _COLLISION_PROXY_VERSION,
        "profile_signature": _profile_signature(target_profile),
        "mesh_signature": _collision_mesh_signature(armature),
        "segment_mode": segment_mode,
        "segment_end_bones": segment_end_bones,
        "height": float(target_height),
        "body_bones": body_bones,
        "arm_chains": arm_chains,
        "terminal_arm_bones": terminal_arm_bones,
        "radii": radii,
        "sample_counts": sample_counts,
    }
    armature[_COLLISION_PROXY_PROP] = json.dumps(proxy, sort_keys=True)
    print(
        f"[StableGen] Built skeletal collision proxy for {armature.name}: "
        f"{len(proxy_bones)} capsule bone(s), {sampled_vertices} weighted vertex sample(s)."
    )
    return proxy


def _collision_proxy_for_armature(
        armature, target_profile, segment_mode=None, rebuild=False):
    if segment_mode not in {"BONE_TAILS", "JOINT_HEADS"}:
        segment_mode = armature.get(
            _COLLISION_SEGMENT_MODE_PROP, "BONE_TAILS")
    if segment_mode not in {"BONE_TAILS", "JOINT_HEADS"}:
        segment_mode = "BONE_TAILS"
    expected_profile = _profile_signature(target_profile)
    expected_mesh = _collision_mesh_signature(armature)
    if not rebuild:
        raw = armature.get(_COLLISION_PROXY_PROP)
        if raw:
            try:
                proxy = json.loads(raw)
                if (
                        proxy.get("version") == _COLLISION_PROXY_VERSION
                        and proxy.get("profile_signature") == expected_profile
                        and proxy.get("mesh_signature") == expected_mesh
                        and proxy.get("segment_mode") == segment_mode):
                    return proxy
            except Exception:
                pass
    return _build_collision_proxy(
        armature, target_profile, segment_mode=segment_mode)


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


def _chain_end_point(armature, chain, use_terminal_head=False):
    if not chain:
        return None
    return _profile_point(
        armature, chain[-1], use_tail=not use_terminal_head)


def _pose_chain_end_point(armature, chain, use_terminal_head=False):
    if not chain:
        return None
    return _pose_point(
        armature, chain[-1], use_tail=not use_terminal_head)


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
    chains = _profile_chains(profile)
    left_point = _average_points([
        _chain_end_point(
            armature, chains.get("left_arm", []), use_terminal_head=True),
        _chain_end_point(armature, chains.get("left_leg", [])),
    ])
    right_point = _average_points([
        _chain_end_point(
            armature, chains.get("right_arm", []), use_terminal_head=True),
        _chain_end_point(armature, chains.get("right_leg", [])),
    ])
    side = (left_point - right_point) if left_point is not None and right_point is not None else None
    projected = _horizontal_projection(side, up)
    return _safe_normalized(projected, mathutils.Vector((1.0, 0.0, 0.0)))


def _pose_side_axis(armature, profile, up):
    chains = _profile_chains(profile)
    left_point = _average_points([
        _pose_chain_end_point(
            armature, chains.get("left_arm", []), use_terminal_head=True),
        _pose_chain_end_point(armature, chains.get("left_leg", [])),
    ])
    right_point = _average_points([
        _pose_chain_end_point(
            armature, chains.get("right_arm", []), use_terminal_head=True),
        _pose_chain_end_point(armature, chains.get("right_leg", [])),
    ])
    side = (left_point - right_point) if left_point is not None and right_point is not None else None
    projected = _horizontal_projection(side, up)
    return _safe_normalized(projected, mathutils.Vector((1.0, 0.0, 0.0)))


def _rest_foot_front_hint(armature, profile, up):
    chains = _profile_chains(profile)
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
    if bool(flip_forward) ^ _profile_forward_flipped(profile):
        forward = -forward
        left = _safe_normalized(up.cross(forward), -left)
    return mathutils.Matrix((left, forward, up)).transposed()


def _pose_body_basis(armature, profile, flip_forward=False):
    up = _pose_up_axis(armature, profile)
    left = _pose_side_axis(armature, profile, up)
    left, forward = _body_axes_from_side_up(armature, profile, left, up)
    if bool(flip_forward) ^ _profile_forward_flipped(profile):
        forward = -forward
        left = _safe_normalized(up.cross(forward), -left)
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
        key_locations=False, key_scale=False,
        previous_rotations=None):
    for name in target_names:
        pose_bone = target_armature.pose.bones.get(name)
        if not pose_bone:
            continue
        if previous_rotations is not None:
            rotation = pose_bone.rotation_quaternion.copy()
            previous = previous_rotations.get(name)
            if previous is not None:
                rotation.make_compatible(previous)
                pose_bone.rotation_quaternion = rotation
            previous_rotations[name] = rotation.copy()
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


def _armature_reference_matrices(armature, use_rest=False):
    if use_rest:
        return {
            bone.name: bone.matrix_local.copy()
            for bone in armature.data.bones
        }
    return {
        pose_bone.name: pose_bone.matrix.copy()
        for pose_bone in armature.pose.bones
    }


def _reference_local_matrix(reference_matrices, pose_bone):
    matrix = reference_matrices.get(pose_bone.name)
    if matrix is None:
        return None
    if not pose_bone.parent:
        return matrix.copy()
    parent_matrix = reference_matrices.get(pose_bone.parent.name)
    if parent_matrix is None:
        return matrix.copy()
    return parent_matrix.inverted_safe() @ matrix


def _current_local_matrix(pose_bone):
    if not pose_bone.parent:
        return pose_bone.matrix.copy()
    return pose_bone.parent.matrix.inverted_safe() @ pose_bone.matrix


def _pose_bone_depth(pose_bone):
    depth = 0
    parent = pose_bone.parent
    while parent:
        depth += 1
        parent = parent.parent
    return depth


def _rest_space_pair_records(
        source_armature, target_armature, pairs, source_reference_matrices,
        target_reference_matrices, body_alignment):
    body_alignment_q = body_alignment.to_quaternion()
    records = []
    for source_name, target_name, role in pairs:
        source_pose = source_armature.pose.bones.get(source_name)
        target_pose = target_armature.pose.bones.get(target_name)
        if not source_pose or not target_pose:
            continue
        source_reference_local = _reference_local_matrix(
            source_reference_matrices, source_pose)
        target_reference_local = _reference_local_matrix(
            target_reference_matrices, target_pose)
        if source_reference_local is None or target_reference_local is None:
            continue

        if source_pose.parent:
            source_parent_reference = source_reference_matrices.get(
                source_pose.parent.name, mathutils.Matrix.Identity(4))
        else:
            source_parent_reference = mathutils.Matrix.Identity(4)
        if target_pose.parent:
            target_parent_reference = target_reference_matrices.get(
                target_pose.parent.name, mathutils.Matrix.Identity(4))
        else:
            target_parent_reference = mathutils.Matrix.Identity(4)

        source_parent_rotation = source_parent_reference.to_quaternion()
        target_parent_rotation = target_parent_reference.to_quaternion()
        parent_alignment = (
            target_parent_rotation.inverted()
            @ body_alignment_q
            @ source_parent_rotation
        )
        source_reference_rotation = source_reference_matrices[source_name].to_quaternion()
        target_reference_rotation = target_reference_matrices[target_name].to_quaternion()
        bone_alignment = (
            target_reference_rotation.inverted()
            @ body_alignment_q
            @ source_reference_rotation
        )
        records.append({
            "source": source_pose,
            "target": target_pose,
            "role": role,
            "source_reference_local": source_reference_local,
            "target_reference_local": target_reference_local,
            "parent_alignment": parent_alignment,
            "bone_alignment": bone_alignment,
        })
    records.sort(key=lambda record: _pose_bone_depth(record["target"]))
    return records


def _apply_rest_space_pose(
        records, copy_root_motion, location_scale, context):
    for record in records:
        source_pose = record["source"]
        target_pose = record["target"]
        source_reference_local = record["source_reference_local"]
        target_reference_local = record["target_reference_local"]
        parent_alignment = record["parent_alignment"]
        bone_alignment = record["bone_alignment"]

        source_pose_local = _current_local_matrix(source_pose)
        source_loc, source_rot, _source_scale = source_pose_local.decompose()
        source_ref_loc, source_ref_rot, _source_ref_scale = source_reference_local.decompose()
        target_ref_loc, target_ref_rot, target_ref_scale = target_reference_local.decompose()

        source_delta = source_ref_rot.inverted() @ source_rot
        target_delta = bone_alignment @ source_delta @ bone_alignment.inverted()
        desired_local_rotation = target_ref_rot @ target_delta
        desired_local_location = target_ref_loc.copy()
        if record["role"] == "pelvis" and copy_root_motion:
            source_location_delta = source_loc - source_ref_loc
            desired_local_location += (
                parent_alignment @ source_location_delta
            ) * location_scale

        desired_local_matrix = mathutils.Matrix.LocRotScale(
            desired_local_location,
            desired_local_rotation,
            target_ref_scale,
        )
        if target_pose.parent:
            desired_matrix = target_pose.parent.matrix @ desired_local_matrix
        else:
            desired_matrix = desired_local_matrix
        target_pose.rotation_mode = "QUATERNION"
        target_pose.matrix = desired_matrix
        context.view_layer.update()


def _bake_rest_space_retarget(
        context, source_armature, target_armature, pairs,
        source_profile, target_profile, start, end, frame_step,
        copy_root_motion, target_reference_pose, source_reference_mode="START_FRAME",
        swap_target_sides=False, limb_match_strength=0.65,
        collision_proxy=None, collision_adaptation=True,
        collision_strength=0.75, collision_margin=0.008,
        collision_iterations=2, collision_max_deviation=0.08,
        arm_correction_driver="AUTO",
        upper_arm_collision="DISABLED",
        match_hand_orientation=False, hand_match_strength=1.0,
        hand_roll_mode="FOREARM", hand_roll_limit=60.0):
    """Bake parent-local motion deltas through corresponding reference frames."""
    twist_helper_names = []
    if match_hand_orientation:
        twist_helper_names = _ensure_hand_roll_setup(
            context,
            target_armature,
            target_profile,
            hand_roll_mode,
        )
    else:
        _configure_twist_helpers(
            target_armature, target_profile, use_twist_chain=False)
    target_names = _mapped_target_bones(pairs)
    target_names.extend(
        name for name in twist_helper_names if name not in target_names)
    all_target_names = [pose_bone.name for pose_bone in target_armature.pose.bones]
    source_height = _height_from_profile(source_armature, source_profile)
    target_height = _height_from_profile(target_armature, target_profile)
    location_scale = target_height / source_height if source_height > 1e-5 else 1.0

    context.scene.frame_set(start)
    context.view_layer.update()
    source_reference_matrices = _armature_reference_matrices(
        source_armature,
        use_rest=source_reference_mode == "REST",
    )
    source_basis = _profile_rotation_basis(source_armature, source_profile)
    target_basis = _profile_rotation_basis(target_armature, target_profile)
    body_alignment = target_basis @ source_basis.inverted()
    source_reference_body_directions = _limb_reference_body_directions(
        source_armature,
        source_profile,
        _profile_body_basis(source_armature, source_profile),
        use_rest=source_reference_mode == "REST",
    )
    source_reference_hand_frames = {}
    if match_hand_orientation:
        source_reference_hand_frames = _hand_reference_body_frames(
            source_armature,
            source_profile,
            _profile_rotation_basis(source_armature, source_profile),
            use_rest=source_reference_mode == "REST",
        )

    if target_reference_pose.get("label") == "rest_pose":
        _reset_target_pose(target_armature, all_target_names)
    else:
        _apply_target_reference_pose(
            target_armature, all_target_names, target_reference_pose)
    context.view_layer.update()
    target_reference_body_directions = _limb_reference_body_directions(
        target_armature,
        target_profile,
        _profile_body_basis(target_armature, target_profile),
        use_rest=False,
    )
    target_reference_hand_frames = {}
    if match_hand_orientation:
        target_reference_hand_frames = _hand_reference_body_frames(
            target_armature,
            target_profile,
            _profile_rotation_basis(target_armature, target_profile),
            use_rest=False,
        )
    collision_baseline = {}
    if collision_adaptation and collision_proxy:
        collision_baseline = _capture_proxy_collision_baseline(
            target_armature,
            collision_proxy,
            margin=collision_margin,
            upper_arm_collision=upper_arm_collision,
        )
    target_reference_matrices = _armature_reference_matrices(target_armature)
    records = _rest_space_pair_records(
        source_armature,
        target_armature,
        pairs,
        source_reference_matrices,
        target_reference_matrices,
        body_alignment,
    )
    if len(records) < 6:
        raise RuntimeError("Not enough parent-local retarget records could be constructed")

    target_chains = _profile_chains(target_profile)
    resolved_arm_drivers = {
        role: _resolved_arm_correction_driver(
            target_armature,
            target_profile,
            role,
            target_chains.get(role, []),
            driver_mode=arm_correction_driver,
        )
        for role in ("left_arm", "right_arm")
    }
    print(
        "[StableGen] Arm correction drivers: "
        + ", ".join(
            f"{role}={driver.lower()}"
            for role, driver in resolved_arm_drivers.items()
        )
    )
    if collision_proxy:
        print(
            f"[StableGen] Collision proxy mode: "
            f"segments={collision_proxy.get('segment_mode', 'BONE_TAILS').lower()}, "
            f"upper_arm={upper_arm_collision.lower()}"
        )

    collision_adjusted_frames = 0
    hand_matched_frames = 0
    previous_keyed_rotations = {}
    for frame in range(start, end + 1, frame_step):
        context.scene.frame_set(frame)
        context.view_layer.update()
        if target_reference_pose.get("label") == "rest_pose":
            _reset_target_pose(target_armature, all_target_names)
        else:
            _apply_target_reference_pose(
                target_armature, all_target_names, target_reference_pose)
        context.view_layer.update()
        _apply_rest_space_pose(
            records,
            copy_root_motion=copy_root_motion,
            location_scale=location_scale,
            context=context,
        )
        if limb_match_strength > 0.0:
            _match_limb_chain_directions(
                context,
                source_armature,
                target_armature,
                source_profile,
                target_profile,
                strength=limb_match_strength,
                swap_target_sides=swap_target_sides,
                use_reference_basis=True,
                source_reference_body_directions=source_reference_body_directions,
                target_reference_body_directions=target_reference_body_directions,
                arm_correction_driver=arm_correction_driver,
            )
        if collision_adaptation and collision_proxy:
            adjusted = _resolve_skeletal_proxy_collisions(
                context,
                target_armature,
                target_profile,
                collision_proxy,
                strength=collision_strength,
                margin=collision_margin,
                iterations=collision_iterations,
                max_pose_deviation=collision_max_deviation,
                accepted_penetrations=collision_baseline,
                arm_correction_driver=arm_correction_driver,
                upper_arm_collision=upper_arm_collision,
            )
            if adjusted:
                collision_adjusted_frames += 1
        if match_hand_orientation and hand_roll_mode == "TWIST_CHAIN":
            _sync_twist_helpers_from_reference(
                target_armature,
                target_profile,
                target_reference_matrices,
            )
            context.view_layer.update()
        if match_hand_orientation and hand_match_strength > 0.0:
            matched_hands = _match_hand_orientations(
                context,
                source_armature,
                target_armature,
                source_profile,
                target_profile,
                strength=hand_match_strength,
                swap_target_sides=swap_target_sides,
                use_reference_basis=True,
                source_reference_body_frames=source_reference_hand_frames,
                target_reference_body_frames=target_reference_hand_frames,
                hand_roll_mode=hand_roll_mode,
                hand_roll_limit=hand_roll_limit,
            )
            if matched_hands:
                hand_matched_frames += 1
        _keyframe_retarget_pose(
            target_armature,
            target_names,
            frame,
            copy_root_motion=copy_root_motion,
            key_locations=target_reference_pose.get("label") != "rest_pose",
            key_scale=target_reference_pose.get("label") != "rest_pose",
            previous_rotations=previous_keyed_rotations,
        )
    if collision_adaptation and collision_proxy:
        print(
            f"[StableGen] Skeletal proxy collision pass adjusted "
            f"{collision_adjusted_frames} baked frame(s)."
        )
    if match_hand_orientation:
        print(
            f"[StableGen] Palm-frame matching adjusted "
            f"{hand_matched_frames} baked frame(s)."
        )


def _target_chain_key(source_chain_key, swap_target_sides=False):
    if not swap_target_sides:
        return source_chain_key
    if source_chain_key.startswith("left_"):
        return "right_" + source_chain_key[5:]
    if source_chain_key.startswith("right_"):
        return "left_" + source_chain_key[6:]
    return source_chain_key


def _resolved_arm_correction_driver(
        armature, profile, role, chain, driver_mode="AUTO"):
    if role not in {"left_arm", "right_arm"} or len(chain) < 3:
        return "SHOULDER"
    if driver_mode in {"SHOULDER", "UPPER_ARM"}:
        return driver_mode

    shoulder = armature.data.bones.get(chain[0])
    upper_arm = armature.data.bones.get(chain[1])
    if not shoulder or not upper_arm:
        return "SHOULDER"
    # A clavicle ahead of the upper arm (shoulder, upper arm, forearm, hand): aiming the wrist
    # by rotating the short clavicle turns every direction or clearance correction into a
    # shrug. Correct from the upper arm and let the clavicle keep the source's own motion.
    if len(chain) >= 4 and float(shoulder.length) <= float(upper_arm.length) * 0.75:
        return "UPPER_ARM"
    gap = (upper_arm.head_local - shoulder.tail_local).length
    target_height = _height_from_profile(armature, profile)
    disconnected_threshold = max(
        target_height * 0.035,
        max(float(shoulder.length), float(upper_arm.length)) * 0.45,
    )
    if not upper_arm.use_connect and gap > disconnected_threshold:
        return "UPPER_ARM"
    return "SHOULDER"


def _arm_correction_chain(
        armature, profile, role, chain, driver_mode="AUTO"):
    driver = _resolved_arm_correction_driver(
        armature, profile, role, chain, driver_mode=driver_mode)
    if driver == "UPPER_ARM" and len(chain) >= 3:
        return chain[1:]
    return chain


def _pose_chain_direction(armature, chain, use_terminal_head=False):
    if len(chain) < 2:
        return None
    start = _pose_point(armature, chain[0], use_tail=False)
    end = _pose_point(
        armature, chain[-1], use_tail=not use_terminal_head)
    if start is None or end is None:
        return None
    direction = end - start
    if direction.length < 1e-6:
        return None
    return direction.normalized()


def _rest_chain_direction(armature, chain, use_terminal_head=False):
    if len(chain) < 2:
        return None
    start = _profile_point(armature, chain[0], use_tail=False)
    end = _profile_point(
        armature, chain[-1], use_tail=not use_terminal_head)
    if start is None or end is None:
        return None
    direction = end - start
    if direction.length < 1e-6:
        return None
    return direction.normalized()


def _limb_reference_body_directions(
        armature, profile, basis, use_rest=False):
    directions = {}
    for role in ("left_arm", "right_arm", "left_leg", "right_leg"):
        chain = _profile_chains(profile).get(role, [])
        chain = _retarget_chain_for_role(armature, profile, role, chain)
        use_terminal_head = role in {"left_arm", "right_arm"}
        if use_rest:
            direction = _rest_chain_direction(
                armature, chain, use_terminal_head=use_terminal_head)
        else:
            direction = _pose_chain_direction(
                armature, chain, use_terminal_head=use_terminal_head)
        if direction is None:
            continue
        body_direction = basis.inverted_safe() @ direction
        if body_direction.length > 1e-6:
            directions[role] = body_direction.normalized()
        if role in {"left_arm", "right_arm"} and len(chain) >= 3:
            upper_chain = chain[1:]
            if use_rest:
                upper_direction = _rest_chain_direction(
                    armature, upper_chain, use_terminal_head=True)
            else:
                upper_direction = _pose_chain_direction(
                    armature, upper_chain, use_terminal_head=True)
            if upper_direction is not None:
                upper_body_direction = basis.inverted_safe() @ upper_direction
                if upper_body_direction.length > 1e-6:
                    directions[f"{role}:upper_arm"] = (
                        upper_body_direction.normalized())
    return directions


def _hand_digit_root_names(armature, hand_name):
    """Find a palm branch with three to ten digit roots, allowing extra digits."""
    hand_bone = armature.data.bones.get(hand_name) if hand_name else None
    if not hand_bone:
        return []
    candidates = [(hand_bone, 0)]
    cursor = 0
    branching = []
    while cursor < len(candidates):
        bone, depth = candidates[cursor]
        cursor += 1
        children = list(bone.children)
        if 3 <= len(children) <= 10:
            branching.append((depth, -len(children), bone.name, children))
        if depth < 3:
            candidates.extend((child, depth + 1) for child in children)
    if not branching:
        return []
    _depth, _count, _name, roots = min(branching)
    return sorted(root.name for root in roots)


def _hand_landmark_point(armature, bone_name, use_rest=False, use_tail=False):
    if use_rest:
        return _profile_point(armature, bone_name, use_tail=use_tail)
    return _pose_point(armature, bone_name, use_tail=use_tail)


def _hand_thumb_root_name(armature, hand_name, digit_roots=None):
    """Identify the thumb by semantics or its rest-pose offset toward the wrist."""
    digit_roots = list(
        digit_roots or _hand_digit_root_names(armature, hand_name))
    if len(digit_roots) < 3:
        return None
    for root_name in digit_roots:
        normalized = _norm_name(root_name)
        if "thumb" in normalized or "pollex" in normalized:
            return root_name

    wrist = _profile_point(armature, hand_name, use_tail=False)
    points = {
        name: _profile_point(armature, name, use_tail=False)
        for name in digit_roots
    }
    if wrist is None or any(point is None for point in points.values()):
        return None
    center = sum(
        points.values(), mathutils.Vector((0.0, 0.0, 0.0))) / len(points)
    forward = center - wrist
    if forward.length < 1e-6:
        return None
    forward.normalize()
    return min(
        digit_roots,
        key=lambda name: (points[name] - wrist).dot(forward),
    )


def _hand_frame(armature, hand_name, use_rest=False):
    digit_roots = _hand_digit_root_names(armature, hand_name)
    if len(digit_roots) < 3:
        return None
    thumb_name = _hand_thumb_root_name(
        armature, hand_name, digit_roots=digit_roots)
    if not thumb_name:
        return None
    wrist = _hand_landmark_point(armature, hand_name, use_rest=use_rest)
    root_points = [
        _hand_landmark_point(armature, name, use_rest=use_rest)
        for name in digit_roots
    ]
    if wrist is None or any(point is None for point in root_points):
        return None

    def frame_from_points(points):
        center = sum(
            points, mathutils.Vector((0.0, 0.0, 0.0))) / len(points)
        forward = center - wrist
        if forward.length < 1e-6:
            return None
        forward.normalize()
        thumb_index = digit_roots.index(thumb_name)
        thumb_point = points[thumb_index]
        other_points = [
            point for index, point in enumerate(points)
            if index != thumb_index
        ]
        other_center = sum(
            other_points,
            mathutils.Vector((0.0, 0.0, 0.0)),
        ) / len(other_points)
        across = thumb_point - other_center
        across -= forward * across.dot(forward)
        if across.length < 1e-6:
            return None
        across.normalize()
        normal = across.cross(forward)
        if normal.length < 1e-6:
            return None
        normal.normalize()
        forward = normal.cross(across).normalized()
        return mathutils.Matrix((across, forward, normal)).transposed()

    frame = frame_from_points(root_points)
    if frame is None:
        root_points = [
            _hand_landmark_point(
                armature, name, use_rest=use_rest, use_tail=True)
            for name in digit_roots
        ]
        if any(point is None for point in root_points):
            return None
        frame = frame_from_points(root_points)
    return frame


def _hand_reference_body_frames(armature, profile, basis, use_rest=False):
    frames = {}
    chains = _profile_chains(profile)
    for role in ("left_arm", "right_arm"):
        chain = chains.get(role, [])
        if not chain:
            continue
        frame = _hand_frame(armature, chain[-1], use_rest=use_rest)
        if frame is not None:
            frames[role] = basis.inverted_safe() @ frame
    return frames


def _quaternion_twist_about_axis(rotation, axis):
    """Extract the shortest quaternion component around an armature-space axis."""
    if axis is None or axis.length < 1e-6:
        return None
    axis = axis.normalized()
    rotation = rotation.normalized()
    if rotation.w < 0.0:
        rotation = mathutils.Quaternion(
            (-rotation.w, -rotation.x, -rotation.y, -rotation.z))
    vector = mathutils.Vector((rotation.x, rotation.y, rotation.z))
    projected = axis * vector.dot(axis)
    twist = mathutils.Quaternion(
        (rotation.w, projected.x, projected.y, projected.z))
    if twist.magnitude < 1e-8:
        return None
    return twist.normalized()


def _twist_helper_specs(role):
    side = "L" if role == "left_arm" else "R"
    return (
        (f"SG_{side}_ForearmTwist_00", 0.0),
        (f"SG_{side}_ForearmTwist_33", 1.0 / 3.0),
        (f"SG_{side}_ForearmTwist_67", 2.0 / 3.0),
    )


def _twist_helper_names(armature, profile):
    names = []
    chains = _profile_chains(profile)
    for role in ("left_arm", "right_arm"):
        if len(chains.get(role, [])) < 2:
            continue
        names.extend(
            name for name, _fraction in _twist_helper_specs(role)
            if armature.data.bones.get(name)
        )
    return names


def _twist_follow_constraint(pose_bone):
    if not pose_bone:
        return None
    constraint = pose_bone.constraints.get("SG_Follow_Forearm")
    if constraint and constraint.type == "COPY_TRANSFORMS":
        return constraint
    constraint = pose_bone.constraints.new("COPY_TRANSFORMS")
    constraint.name = "SG_Follow_Forearm"
    return constraint


def _ensure_twist_helper_bones(context, armature, profile):
    chains = _profile_chains(profile)
    required = []
    for role in ("left_arm", "right_arm"):
        chain = chains.get(role, [])
        if len(chain) < 2:
            continue
        required.extend(
            (role, chain[-2], name, fraction)
            for name, fraction in _twist_helper_specs(role)
        )
    if not required:
        return []

    missing = [
        item for item in required
        if not armature.data.bones.get(item[2])
    ]
    if missing:
        original_active = context.view_layer.objects.active
        original_selection = list(context.selected_objects)
        try:
            if bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            armature.select_set(True)
            context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode="EDIT")
            edit_bones = armature.data.edit_bones
            for _role, forearm_name, helper_name, _fraction in missing:
                forearm = edit_bones.get(forearm_name)
                if not forearm or edit_bones.get(helper_name):
                    continue
                helper = edit_bones.new(helper_name)
                helper.head = forearm.head
                helper.tail = forearm.tail
                helper.roll = forearm.roll
                helper.parent = None
                helper.use_connect = False
                helper.use_deform = True
                helper.head_radius = forearm.head_radius
                helper.tail_radius = forearm.tail_radius
                helper.envelope_distance = forearm.envelope_distance
                helper.envelope_weight = forearm.envelope_weight
            bpy.ops.object.mode_set(mode="OBJECT")
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

    helper_names = []
    for role, forearm_name, helper_name, fraction in required:
        data_bone = armature.data.bones.get(helper_name)
        pose_bone = armature.pose.bones.get(helper_name)
        if not data_bone or not pose_bone:
            continue
        data_bone["sg_twist_source"] = forearm_name
        data_bone["sg_twist_fraction"] = float(fraction)
        data_bone["sg_twist_role"] = role
        pose_bone.rotation_mode = "QUATERNION"
        constraint = _twist_follow_constraint(pose_bone)
        constraint.target = armature
        constraint.subtarget = forearm_name
        constraint.target_space = "POSE"
        constraint.owner_space = "POSE"
        constraint.influence = 1.0
        helper_names.append(helper_name)
    context.view_layer.update()
    return helper_names


def _armature_deformed_meshes(armature):
    result = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(
                modifier.type == "ARMATURE" and modifier.object == armature
                for modifier in obj.modifiers):
            result.append(obj)
    return result


def _redistribute_forearm_twist_weights(armature, profile):
    chains = _profile_chains(profile)
    total_vertices = 0
    for mesh_obj in _armature_deformed_meshes(armature):
        mesh_to_armature = armature.matrix_world.inverted_safe() @ mesh_obj.matrix_world
        for modifier in mesh_obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == armature:
                modifier.use_deform_preserve_volume = True

        for role in ("left_arm", "right_arm"):
            chain = chains.get(role, [])
            if len(chain) < 2:
                continue
            forearm_name = chain[-2]
            hand_name = chain[-1]
            marker = f"sg_twist_weights_{armature.name}_{role}"
            if mesh_obj.get(marker) == _TWIST_CHAIN_VERSION:
                continue
            source_group = mesh_obj.vertex_groups.get(forearm_name)
            forearm = armature.data.bones.get(forearm_name)
            hand = armature.data.bones.get(hand_name)
            if not source_group or not forearm or not hand:
                continue
            helper_groups = [
                mesh_obj.vertex_groups.get(name)
                or mesh_obj.vertex_groups.new(name=name)
                for name, _fraction in _twist_helper_specs(role)
            ]
            elbow = forearm.head_local.copy()
            wrist = hand.head_local.copy()
            axis = wrist - elbow
            length_squared = axis.length_squared
            if length_squared < 1e-12:
                continue

            assignments = []
            source_index = source_group.index
            for vertex in mesh_obj.data.vertices:
                source_weight = next(
                    (
                        item.weight for item in vertex.groups
                        if item.group == source_index
                    ),
                    0.0,
                )
                if source_weight <= 1e-8:
                    continue
                point = mesh_to_armature @ vertex.co
                parameter = max(
                    0.0,
                    min(1.0, (point - elbow).dot(axis) / length_squared),
                )
                scaled = parameter * 3.0
                lower = min(2, int(math.floor(scaled)))
                blend = scaled - lower
                weights = [0.0, 0.0, 0.0, 0.0]
                weights[lower] = 1.0 - blend
                weights[lower + 1] = blend
                assignments.append((vertex.index, source_weight, weights))

            for vertex_index, source_weight, weights in assignments:
                for group, weight in zip(helper_groups, weights[:3]):
                    if weight > 1e-8:
                        group.add(
                            [vertex_index], source_weight * weight, "REPLACE")
                    else:
                        try:
                            group.remove([vertex_index])
                        except RuntimeError:
                            pass
                if weights[3] > 1e-8:
                    source_group.add(
                        [vertex_index], source_weight * weights[3], "REPLACE")
                else:
                    source_group.remove([vertex_index])
            mesh_obj[marker] = _TWIST_CHAIN_VERSION
            total_vertices += len(assignments)
    if total_vertices:
        print(
            f"[StableGen] Distributed {total_vertices} forearm-weighted "
            "vertices across generated twist chains."
        )
    return total_vertices


def _configure_twist_helpers(armature, profile, use_twist_chain):
    chains = _profile_chains(profile)
    for role in ("left_arm", "right_arm"):
        chain = chains.get(role, [])
        if len(chain) < 2:
            continue
        forearm_name = chain[-2]
        forearm_pose = armature.pose.bones.get(forearm_name)
        for helper_name, _fraction in _twist_helper_specs(role):
            pose_bone = armature.pose.bones.get(helper_name)
            if not pose_bone:
                continue
            constraint = _twist_follow_constraint(pose_bone)
            constraint.target = armature
            constraint.subtarget = forearm_name
            constraint.target_space = "POSE"
            constraint.owner_space = "POSE"
            if use_twist_chain:
                followed_matrix = (
                    forearm_pose.matrix.copy() if forearm_pose else None)
                constraint.influence = 0.0
                if followed_matrix is not None:
                    pose_bone.matrix = followed_matrix
            else:
                constraint.influence = 1.0


def _ensure_hand_roll_setup(context, armature, profile, hand_roll_mode):
    use_twist_chain = hand_roll_mode == "TWIST_CHAIN"
    if use_twist_chain:
        _ensure_twist_helper_bones(context, armature, profile)
        _redistribute_forearm_twist_weights(armature, profile)
        armature[_TWIST_CHAIN_VERSION_PROP] = _TWIST_CHAIN_VERSION
    _configure_twist_helpers(
        armature,
        profile,
        use_twist_chain=use_twist_chain,
    )
    context.view_layer.update()
    return _twist_helper_names(armature, profile) if use_twist_chain else []


def _apply_twist_to_helper_pose(armature, role, twist):
    for helper_name, fraction in _twist_helper_specs(role):
        pose_bone = armature.pose.bones.get(helper_name)
        if not pose_bone:
            continue
        partial_twist = mathutils.Quaternion(
            (1.0, 0.0, 0.0, 0.0)).slerp(twist, fraction)
        _rotate_pose_bone_in_armature_space(pose_bone, partial_twist)


def _sync_twist_helpers_from_reference(
        armature, profile, reference_matrices):
    chains = _profile_chains(profile)
    for role in ("left_arm", "right_arm"):
        chain = chains.get(role, [])
        if len(chain) < 2:
            continue
        forearm_pose = armature.pose.bones.get(chain[-2])
        reference_forearm = reference_matrices.get(chain[-2])
        if not forearm_pose or reference_forearm is None:
            continue
        motion_delta = forearm_pose.matrix @ reference_forearm.inverted_safe()
        for helper_name, _fraction in _twist_helper_specs(role):
            helper_pose = armature.pose.bones.get(helper_name)
            reference_helper = reference_matrices.get(helper_name)
            if helper_pose and reference_helper is not None:
                helper_pose.matrix = motion_delta @ reference_helper


def _sync_twist_helpers_to_forearms(armature, profile):
    chains = _profile_chains(profile)
    for role in ("left_arm", "right_arm"):
        chain = chains.get(role, [])
        if len(chain) < 2:
            continue
        forearm_pose = armature.pose.bones.get(chain[-2])
        if not forearm_pose:
            continue
        for helper_name, _fraction in _twist_helper_specs(role):
            helper_pose = armature.pose.bones.get(helper_name)
            if helper_pose:
                helper_pose.matrix = forearm_pose.matrix.copy()


def _match_hand_orientations(
        context, source_armature, target_armature,
        source_profile, target_profile, strength=1.0,
        swap_target_sides=False, use_reference_basis=True,
        source_reference_body_frames=None,
        target_reference_body_frames=None,
        absolute_alignment=False,
        hand_roll_mode="FOREARM", hand_roll_limit=60.0):
    if use_reference_basis:
        source_basis = _profile_rotation_basis(source_armature, source_profile)
        target_basis = _profile_rotation_basis(target_armature, target_profile)
    else:
        source_basis = _pose_rotation_basis(source_armature, source_profile)
        target_basis = _pose_rotation_basis(target_armature, target_profile)
    source_chains = _profile_chains(source_profile)
    target_chains = _profile_chains(target_profile)
    strength = max(0.0, min(1.0, float(strength)))
    hand_roll_mode = (
        hand_roll_mode
        if hand_roll_mode in {"WRIST", "FOREARM", "TWIST_CHAIN", "LIMITED"}
        else "FOREARM"
    )
    hand_roll_limit = max(0.0, min(180.0, float(hand_roll_limit)))
    matched = 0

    for source_role in ("left_arm", "right_arm"):
        target_role = _target_chain_key(
            source_role, swap_target_sides=swap_target_sides)
        source_chain = source_chains.get(source_role, [])
        target_chain = target_chains.get(target_role, [])
        if not source_chain or not target_chain:
            continue
        source_frame = _hand_frame(source_armature, source_chain[-1])
        target_frame = _hand_frame(target_armature, target_chain[-1])
        if source_frame is None or target_frame is None:
            continue
        source_body_frame = source_basis.inverted_safe() @ source_frame
        source_reference = (source_reference_body_frames or {}).get(source_role)
        target_reference = (target_reference_body_frames or {}).get(target_role)
        if absolute_alignment:
            desired_target_body_frame = source_body_frame
        elif use_reference_basis and (
                source_reference is None or target_reference is None):
            continue
        elif source_reference is not None and target_reference is not None:
            # Transfer the reference-to-current palm rotation in body space.
            source_delta = source_body_frame @ source_reference.inverted_safe()
            desired_target_body_frame = source_delta @ target_reference
        else:
            desired_target_body_frame = source_body_frame
        desired_target_frame = target_basis @ desired_target_body_frame
        try:
            # The pose-bone helper left-multiplies an armature-space rotation,
            # so construct that delta explicitly in armature space. Quaternion
            # rotation_difference() instead returns a local/right-side delta.
            current_rotation = target_frame.to_quaternion()
            desired_rotation = desired_target_frame.to_quaternion()
            delta = desired_rotation @ current_rotation.inverted()
        except ValueError:
            continue
        if strength < 1.0:
            delta = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(
                delta, strength)
        hand_pose = target_armature.pose.bones.get(target_chain[-1])
        if not hand_pose:
            continue
        forearm_pose = (
            target_armature.pose.bones.get(target_chain[-2])
            if len(target_chain) >= 2 else None
        )
        forearm_axis = (
            hand_pose.head - forearm_pose.head if forearm_pose else None)
        twist = _quaternion_twist_about_axis(delta, forearm_axis)

        if hand_roll_mode == "LIMITED" and twist is not None:
            twist_angle = min(twist.angle, (2.0 * math.pi) - twist.angle)
            limit_radians = math.radians(hand_roll_limit)
            if twist_angle > limit_radians and twist_angle > 1e-8:
                limited_twist = mathutils.Quaternion(
                    (1.0, 0.0, 0.0, 0.0)).slerp(
                        twist, limit_radians / twist_angle)
                swing = delta @ twist.inverted()
                delta = swing @ limited_twist
                twist = limited_twist

        applied_desired_rotation = delta @ current_rotation

        if (
                hand_roll_mode != "WRIST"
                and forearm_pose is not None
                and twist is not None):
            _rotate_pose_bone_in_armature_space(forearm_pose, twist)
            context.view_layer.update()
        if hand_roll_mode == "TWIST_CHAIN" and twist is not None:
            _apply_twist_to_helper_pose(
                target_armature, target_role, twist)
            context.view_layer.update()

        updated_target_frame = _hand_frame(
            target_armature, target_chain[-1])
        if updated_target_frame is None:
            continue
        updated_rotation = updated_target_frame.to_quaternion()
        residual = applied_desired_rotation @ updated_rotation.inverted()
        _rotate_pose_bone_in_armature_space(hand_pose, residual)
        context.view_layer.update()
        matched += 1
    return matched


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
        swap_target_sides=False, use_reference_basis=False,
        source_reference_body_directions=None,
        target_reference_body_directions=None,
        arm_correction_driver="AUTO"):
    if use_reference_basis:
        source_basis = _profile_body_basis(source_armature, source_profile)
        target_basis = _profile_body_basis(target_armature, target_profile)
    else:
        source_basis = _pose_body_basis(source_armature, source_profile)
        target_basis = _pose_body_basis(target_armature, target_profile)
    source_chains = _profile_chains(source_profile)
    target_chains = _profile_chains(target_profile)
    strength = max(0.0, min(float(strength), 1.0))
    matched = 0

    for source_key in ("left_arm", "right_arm", "left_leg", "right_leg"):
        target_key = _target_chain_key(source_key, swap_target_sides=swap_target_sides)
        source_chain = source_chains.get(source_key, [])
        target_chain = target_chains.get(target_key, [])
        source_chain = _retarget_chain_for_role(
            source_armature, source_profile, source_key, source_chain)
        target_chain = _retarget_chain_for_role(
            target_armature, target_profile, target_key, target_chain)
        use_terminal_head = source_key in {"left_arm", "right_arm"}
        reference_key = source_key
        target_reference_key = target_key
        if use_terminal_head:
            target_driver = _resolved_arm_correction_driver(
                target_armature,
                target_profile,
                target_key,
                target_chain,
                driver_mode=arm_correction_driver,
            )
            if target_driver == "UPPER_ARM" and len(source_chain) >= 3:
                source_chain = source_chain[1:]
                target_chain = target_chain[1:]
                reference_key = f"{source_key}:upper_arm"
                target_reference_key = f"{target_key}:upper_arm"
        source_direction = _pose_chain_direction(
            source_armature,
            source_chain,
            use_terminal_head=use_terminal_head,
        )
        target_direction = _pose_chain_direction(
            target_armature,
            target_chain,
            use_terminal_head=use_terminal_head,
        )
        if source_direction is None or target_direction is None:
            continue

        source_body_direction = source_basis.inverted() @ source_direction
        source_reference_direction = (
            source_reference_body_directions or {}).get(reference_key)
        target_reference_direction = (
            target_reference_body_directions or {}).get(target_reference_key)
        if source_reference_direction is not None and target_reference_direction is not None:
            try:
                source_direction_delta = source_reference_direction.rotation_difference(
                    source_body_direction)
            except ValueError:
                continue
            desired_target_body_direction = (
                source_direction_delta @ target_reference_direction)
        else:
            desired_target_body_direction = source_body_direction
        desired_target_direction = target_basis @ desired_target_body_direction
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


def _rotate_chain_endpoint_toward(
        context, armature, chain, desired_endpoint, strength=1.0,
        use_terminal_head=False):
    if len(chain) < 2:
        return False
    root_point = _pose_point(armature, chain[0], use_tail=False)
    current_endpoint = _pose_point(
        armature, chain[-1], use_tail=not use_terminal_head)
    if root_point is None or current_endpoint is None:
        return False
    current_direction = current_endpoint - root_point
    desired_direction = desired_endpoint - root_point
    if current_direction.length < 1e-6 or desired_direction.length < 1e-6:
        return False
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


def _closest_points_on_segments(start_a, end_a, start_b, end_b):
    """Return the closest points on two finite 3D line segments."""
    direction_a = end_a - start_a
    direction_b = end_b - start_b
    offset = start_a - start_b
    length_a = direction_a.dot(direction_a)
    length_b = direction_b.dot(direction_b)
    epsilon = 1e-12

    if length_a <= epsilon and length_b <= epsilon:
        return start_a.copy(), start_b.copy()
    if length_a <= epsilon:
        parameter_a = 0.0
        parameter_b = max(0.0, min(1.0, direction_b.dot(offset) / length_b))
    else:
        cross_ab = direction_a.dot(direction_b)
        cross_ao = direction_a.dot(offset)
        if length_b <= epsilon:
            parameter_b = 0.0
            parameter_a = max(0.0, min(1.0, -cross_ao / length_a))
        else:
            cross_bo = direction_b.dot(offset)
            denominator = (length_a * length_b) - (cross_ab * cross_ab)
            if abs(denominator) > epsilon:
                parameter_a = max(
                    0.0,
                    min(1.0, ((cross_ab * cross_bo) - (cross_ao * length_b)) / denominator),
                )
            else:
                parameter_a = 0.0
            parameter_b = ((cross_ab * parameter_a) + cross_bo) / length_b
            if parameter_b < 0.0:
                parameter_b = 0.0
                parameter_a = max(0.0, min(1.0, -cross_ao / length_a))
            elif parameter_b > 1.0:
                parameter_b = 1.0
                parameter_a = max(
                    0.0,
                    min(1.0, (cross_ab - cross_ao) / length_a),
                )

    return (
        start_a + (direction_a * parameter_a),
        start_b + (direction_b * parameter_b),
    )


def _proxy_pose_capsule(armature, proxy, bone_name, radius_scale=1.0):
    pose_bone = armature.pose.bones.get(bone_name)
    if not pose_bone:
        return None
    radius = float(proxy.get("radii", {}).get(bone_name, 0.0)) * radius_scale
    if radius <= 1e-6:
        return None
    start = pose_bone.head.copy()
    if bone_name in set(proxy.get("terminal_arm_bones", [])):
        end = start.copy()
    elif proxy.get("segment_mode") == "JOINT_HEADS":
        end_name = proxy.get("segment_end_bones", {}).get(bone_name)
        end_pose_bone = (
            armature.pose.bones.get(end_name) if end_name else None)
        end = end_pose_bone.head.copy() if end_pose_bone else pose_bone.tail.copy()
    else:
        end = pose_bone.tail.copy()
    return {
        "bone": bone_name,
        "start": start,
        "end": end,
        "radius": radius,
    }


def _proxy_capsules_for_bones(
        armature, proxy, bone_names, radius_scale=1.0,
        start_fractions=None):
    result = []
    seen = set()
    for name in bone_names:
        if name in seen:
            continue
        seen.add(name)
        capsule = _proxy_pose_capsule(
            armature, proxy, name, radius_scale=radius_scale)
        if capsule:
            start_fraction = float((start_fractions or {}).get(name, 0.0))
            if start_fraction > 0.0:
                start_fraction = max(0.0, min(0.95, start_fraction))
                capsule["start"] = capsule["start"].lerp(
                    capsule["end"], start_fraction)
            result.append(capsule)
    return result


def _distal_arm_proxy_bones(chain):
    # Shoulder and upper-arm capsules normally enter the torso volume at their
    # attachment. Only forearm/hand geometry should drive endpoint correction.
    return chain[-2:] if len(chain) >= 2 else chain


def _arm_collision_probe_spec(chain, upper_arm_collision="DISABLED"):
    bones = list(_distal_arm_proxy_bones(chain))
    start_fractions = {}
    if upper_arm_collision in {"DISTAL", "FULL"} and len(chain) >= 3:
        upper_arm = chain[-3]
        if upper_arm not in bones:
            bones.insert(0, upper_arm)
        if upper_arm_collision == "DISTAL":
            start_fractions[upper_arm] = 0.25
    return bones, start_fractions


def _capsule_pair_key(moving, obstacle):
    return f"{moving['bone']}|{obstacle['bone']}"


def _capsule_pair_penetration(moving, obstacle, margin):
    point_a, point_b = _closest_points_on_segments(
        moving["start"], moving["end"], obstacle["start"], obstacle["end"])
    distance = (point_a - point_b).length
    return max(
        0.0,
        moving["radius"] + obstacle["radius"] + margin - distance,
    )


def _capture_proxy_collision_baseline(
        target_armature, collision_proxy, margin=0.0, radius_scale=1.0,
        upper_arm_collision="DISABLED"):
    """Record intersections that already exist in an accepted reference pose."""
    if not collision_proxy:
        return {}
    target_height = max(float(collision_proxy.get("height", 0.0)), 1e-5)
    margin_distance = max(0.0, float(margin)) * target_height
    radius_scale = max(0.25, min(2.0, float(radius_scale)))
    arm_chains = collision_proxy.get("arm_chains", {})
    body_capsules = _proxy_capsules_for_bones(
        target_armature,
        collision_proxy,
        collision_proxy.get("body_bones", []),
        radius_scale=radius_scale,
    )
    baseline = {}
    for role in ("left_arm", "right_arm"):
        moving_bones, moving_start_fractions = _arm_collision_probe_spec(
            arm_chains.get(role, []), upper_arm_collision)
        moving_capsules = _proxy_capsules_for_bones(
            target_armature,
            collision_proxy,
            moving_bones,
            radius_scale=radius_scale,
            start_fractions=moving_start_fractions,
        )
        other_role = "right_arm" if role == "left_arm" else "left_arm"
        other_bones, other_start_fractions = _arm_collision_probe_spec(
            arm_chains.get(other_role, []), upper_arm_collision)
        other_capsules = _proxy_capsules_for_bones(
            target_armature,
            collision_proxy,
            other_bones,
            radius_scale=radius_scale,
            start_fractions=other_start_fractions,
        )
        for moving in moving_capsules:
            for obstacle in body_capsules + other_capsules:
                baseline[_capsule_pair_key(moving, obstacle)] = (
                    _capsule_pair_penetration(
                        moving, obstacle, margin_distance)
                )
    return baseline


def _strongest_capsule_push(
        moving_capsules, obstacle_capsules, margin, fallback,
        accepted_penetrations=None):
    strongest = None
    strongest_depth = 0.0
    accepted_penetrations = accepted_penetrations or {}
    for moving in moving_capsules:
        for obstacle in obstacle_capsules:
            point_a, point_b = _closest_points_on_segments(
                moving["start"], moving["end"], obstacle["start"], obstacle["end"])
            separation = point_a - point_b
            distance = separation.length
            penetration = (
                moving["radius"] + obstacle["radius"] + margin - distance
                - float(accepted_penetrations.get(
                    _capsule_pair_key(moving, obstacle), 0.0))
            )
            if penetration <= strongest_depth:
                continue
            if distance > 1e-6:
                normal = separation / distance
            else:
                normal = fallback.copy()
            if normal.length < 1e-6:
                normal = mathutils.Vector((1.0, 0.0, 0.0))
            strongest = normal.normalized() * penetration
            strongest_depth = penetration
    return strongest


def _resolve_skeletal_proxy_collisions(
        context, target_armature, target_profile, collision_proxy,
        strength=0.75, margin=0.008, iterations=2,
        max_pose_deviation=0.08, radius_scale=1.0,
        accepted_penetrations=None, arm_correction_driver="AUTO",
        upper_arm_collision="DISABLED"):
    """Push distal arm capsules out of body/opposite-arm proxy capsules."""
    if not collision_proxy:
        return 0
    arm_chains = collision_proxy.get("arm_chains", {})
    if any(len(arm_chains.get(role, [])) < 2 for role in ("left_arm", "right_arm")):
        return 0

    target_height = max(
        float(collision_proxy.get("height", 0.0)),
        _height_from_profile(target_armature, target_profile),
    )
    margin_distance = max(0.0, float(margin)) * target_height
    maximum_deviation = max(0.0, float(max_pose_deviation)) * target_height
    strength = max(0.0, min(1.0, float(strength)))
    iterations = max(1, min(8, int(iterations)))
    radius_scale = max(0.25, min(2.0, float(radius_scale)))

    original_endpoints = {}
    for role, chain in arm_chains.items():
        endpoint = _pose_point(target_armature, chain[-1], use_tail=False)
        if endpoint is not None:
            original_endpoints[role] = endpoint.copy()
    body_capsules = _proxy_capsules_for_bones(
        target_armature,
        collision_proxy,
        collision_proxy.get("body_bones", []),
        radius_scale=radius_scale,
    )
    basis = _pose_body_basis(target_armature, target_profile)
    left_axis = basis @ mathutils.Vector((1.0, 0.0, 0.0))
    adjusted_roles = set()

    for _iteration in range(iterations):
        changed = False
        for role in ("left_arm", "right_arm"):
            chain = arm_chains.get(role, [])
            correction_chain = _arm_correction_chain(
                target_armature,
                target_profile,
                role,
                chain,
                driver_mode=arm_correction_driver,
            )
            endpoint = _pose_point(target_armature, chain[-1], use_tail=False)
            original_endpoint = original_endpoints.get(role)
            if endpoint is None or original_endpoint is None:
                continue
            moving_bones, moving_start_fractions = _arm_collision_probe_spec(
                chain, upper_arm_collision)
            moving_capsules = _proxy_capsules_for_bones(
                target_armature,
                collision_proxy,
                moving_bones,
                radius_scale=radius_scale,
                start_fractions=moving_start_fractions,
            )
            other_role = "right_arm" if role == "left_arm" else "left_arm"
            other_bones, other_start_fractions = _arm_collision_probe_spec(
                arm_chains.get(other_role, []), upper_arm_collision)
            other_capsules = _proxy_capsules_for_bones(
                target_armature,
                collision_proxy,
                other_bones,
                radius_scale=radius_scale,
                start_fractions=other_start_fractions,
            )
            fallback = left_axis if role == "left_arm" else -left_axis
            push = _strongest_capsule_push(
                moving_capsules,
                body_capsules + other_capsules,
                margin_distance,
                fallback,
                accepted_penetrations=accepted_penetrations,
            )
            if push is None or push.length < 1e-6:
                continue

            desired_endpoint = endpoint + (push * strength)
            correction = desired_endpoint - original_endpoint
            if maximum_deviation > 0.0 and correction.length > maximum_deviation:
                desired_endpoint = (
                    original_endpoint + (correction.normalized() * maximum_deviation)
                )
            if _rotate_chain_endpoint_toward(
                    context,
                    target_armature,
                    correction_chain,
                    desired_endpoint,
                    strength=1.0,
                    use_terminal_head=True):
                adjusted_roles.add(role)
                changed = True
        if not changed:
            break
    return len(adjusted_roles)


def _protect_reference_pose_clearance(
        context, target_armature, target_profile, clearance=1.0,
        collision_proxy=None, accepted_penetrations=None,
        arm_correction_driver="AUTO",
        upper_arm_collision="DISABLED"):
    """Keep matched hands outside conservative torso and hand proxy colliders."""
    if collision_proxy:
        return _resolve_skeletal_proxy_collisions(
            context,
            target_armature,
            target_profile,
            collision_proxy,
            strength=1.0,
            margin=0.01,
            iterations=4,
            max_pose_deviation=0.08,
            radius_scale=clearance,
            accepted_penetrations=accepted_penetrations,
            arm_correction_driver=arm_correction_driver,
            upper_arm_collision=upper_arm_collision,
        )

    chains = _profile_chains(target_profile)
    arm_chains = {
        role: chains.get(role, [])
        for role in ("left_arm", "right_arm")
    }
    if any(len(chain) < 2 for chain in arm_chains.values()):
        return 0

    pelvis = _pose_point(target_armature, target_profile.get("pelvis"), use_tail=False)
    if pelvis is None:
        return 0
    basis = _pose_body_basis(target_armature, target_profile)
    basis_inv = basis.inverted_safe()
    target_height = _height_from_profile(target_armature, target_profile)

    roots = {}
    hands = {}
    for role, chain in arm_chains.items():
        root = _pose_point(target_armature, chain[0], use_tail=False)
        hand = _pose_point(target_armature, chain[-1], use_tail=False)
        if root is None or hand is None:
            return 0
        roots[role] = basis_inv @ (root - pelvis)
        hands[role] = basis_inv @ (hand - pelvis)

    shoulder_half_width = abs(
        roots["left_arm"].x - roots["right_arm"].x
    ) * 0.5
    clearance = max(0.25, min(float(clearance), 2.0))
    torso_radius = max(shoulder_half_width * 0.9, target_height * 0.11) * clearance
    torso_bottom = -target_height * 0.25
    torso_top = max(roots["left_arm"].z, roots["right_arm"].z) + target_height * 0.08

    corrected = {role: point.copy() for role, point in hands.items()}
    for role, point in corrected.items():
        if point.z < torso_bottom or point.z > torso_top:
            continue
        forward_component = point.y * 0.65
        if abs(forward_component) >= torso_radius:
            continue
        required_lateral = max(
            0.0,
            (torso_radius * torso_radius - forward_component * forward_component) ** 0.5,
        )
        side_sign = 1.0 if role == "left_arm" else -1.0
        if point.x * side_sign < required_lateral:
            point.x = side_sign * required_lateral

    separation = corrected["left_arm"] - corrected["right_arm"]
    minimum_hand_separation = torso_radius * 1.1
    if separation.length < minimum_hand_separation:
        deficit = (minimum_hand_separation - separation.length) * 0.5
        corrected["left_arm"].x += deficit
        corrected["right_arm"].x -= deficit

    adjusted = 0
    for role, body_point in corrected.items():
        if (body_point - hands[role]).length < 1e-6:
            continue
        desired_endpoint = pelvis + (basis @ body_point)
        correction_chain = _arm_correction_chain(
            target_armature,
            target_profile,
            role,
            arm_chains[role],
            driver_mode=arm_correction_driver,
        )
        if _rotate_chain_endpoint_toward(
                context,
                target_armature,
                correction_chain,
                desired_endpoint,
                strength=1.0,
                use_terminal_head=True):
            adjusted += 1
    return adjusted


def _match_source_reference_pose(
        context, source_armature, target_armature, pairs,
        source_profile, target_profile, strength=1.0,
        swap_target_sides=False, protect_mesh=True, clearance=1.0,
        collision_proxy=None, arm_correction_driver="AUTO",
        upper_arm_collision="DISABLED",
        match_hand_orientation=False, hand_match_strength=1.0,
        hand_roll_mode="FOREARM", hand_roll_limit=60.0):
    if match_hand_orientation:
        _ensure_hand_roll_setup(
            context,
            target_armature,
            target_profile,
            hand_roll_mode,
        )
    else:
        _configure_twist_helpers(
            target_armature, target_profile, use_twist_chain=False)
    target_names = _mapped_target_bones(pairs)
    all_target_names = [pose_bone.name for pose_bone in target_armature.pose.bones]
    source_height = _height_from_profile(source_armature, source_profile)
    target_height = _height_from_profile(target_armature, target_profile)
    location_scale = target_height / source_height if source_height > 1e-5 else 1.0

    _reset_target_pose(target_armature, all_target_names)
    context.view_layer.update()
    source_reference_hand_frames = {}
    target_reference_hand_frames = {}
    if match_hand_orientation:
        source_reference_hand_frames = _hand_reference_body_frames(
            source_armature,
            source_profile,
            _profile_rotation_basis(source_armature, source_profile),
            use_rest=True,
        )
        target_reference_hand_frames = _hand_reference_body_frames(
            target_armature,
            target_profile,
            _profile_rotation_basis(target_armature, target_profile),
            use_rest=False,
        )
    collision_baseline = {}
    if protect_mesh and collision_proxy:
        collision_baseline = _capture_proxy_collision_baseline(
            target_armature,
            collision_proxy,
            margin=0.01,
            radius_scale=clearance,
            upper_arm_collision=upper_arm_collision,
        )
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
        arm_correction_driver=arm_correction_driver,
    )
    if protect_mesh:
        adjusted = _protect_reference_pose_clearance(
            context,
            target_armature,
            target_profile,
            clearance=clearance,
            collision_proxy=collision_proxy,
            accepted_penetrations=collision_baseline,
            arm_correction_driver=arm_correction_driver,
            upper_arm_collision=upper_arm_collision,
        )
        if adjusted:
            print(
                f"[StableGen] Reference clearance adjusted {adjusted} arm chain(s) "
                "away from skeletal proxy colliders."
            )
    if match_hand_orientation and hand_match_strength > 0.0:
        if hand_roll_mode == "TWIST_CHAIN":
            _sync_twist_helpers_to_forearms(
                target_armature, target_profile)
            context.view_layer.update()
        matched_hands = _match_hand_orientations(
            context,
            source_armature,
            target_armature,
            source_profile,
            target_profile,
            strength=hand_match_strength,
            swap_target_sides=swap_target_sides,
            use_reference_basis=True,
            source_reference_body_frames=source_reference_hand_frames,
            target_reference_body_frames=target_reference_hand_frames,
            absolute_alignment=True,
            hand_roll_mode=hand_roll_mode,
            hand_roll_limit=hand_roll_limit,
        )
        print(
            f"[StableGen] Reference palm-frame matching adjusted "
            f"{matched_hands} hand(s)."
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


def _apply_pose_chain_direction(
        context, armature, chain, desired_direction, strength=1.0,
        use_terminal_head=False):
    if len(chain) < 2:
        return False
    current_direction = _pose_chain_direction(
        armature, chain, use_terminal_head=use_terminal_head)
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

    chains = _profile_chains(profile)
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
        use_terminal_head = chain_key.endswith("_arm")
        current_direction = _pose_chain_direction(
            armature,
            solve_chain,
            use_terminal_head=use_terminal_head,
        )
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
        if _apply_pose_chain_direction(
                context,
                armature,
                solve_chain,
                desired,
                strength=strength,
                use_terminal_head=use_terminal_head):
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


def _source_preview_objects():
    return [obj for obj in bpy.data.objects if obj.get(_SOURCE_PREVIEW_PROP)]


def _clear_source_preview():
    preview_objects = _source_preview_objects()
    if preview_objects:
        _delete_objects(preview_objects)


def _objects_world_bound_points(objects):
    points = []
    for obj in objects:
        if obj.name not in bpy.data.objects:
            continue
        if obj.type == "MESH":
            points.extend(
                obj.matrix_world @ mathutils.Vector(corner)
                for corner in obj.bound_box
            )
        elif obj.type == "ARMATURE":
            for pose_bone in obj.pose.bones:
                points.append(obj.matrix_world @ pose_bone.head)
                points.append(obj.matrix_world @ pose_bone.tail)
    return points


def _preview_root_objects(objects):
    object_set = set(objects)
    return [obj for obj in objects if obj.parent not in object_set]


def _prepare_source_preview(
        context, imported_objects, target_armature, target_profile):
    imported_objects = [
        obj for obj in imported_objects if obj.name in bpy.data.objects
    ]
    if not imported_objects:
        return 0
    for obj in imported_objects:
        obj[_SOURCE_PREVIEW_PROP] = True
        obj.hide_viewport = False
        obj.hide_render = True
        if obj.type == "ARMATURE":
            obj.show_in_front = True

    roots = _preview_root_objects(imported_objects)
    target_objects = _rig_export_objects(target_armature)
    target_points = _objects_world_bound_points(target_objects)
    source_points = _objects_world_bound_points(imported_objects)
    if not target_points or not source_points:
        return len(imported_objects)

    target_height = max(point.z for point in target_points) - min(
        point.z for point in target_points)
    source_height = max(point.z for point in source_points) - min(
        point.z for point in source_points)
    if target_height > 1e-5 and source_height > 1e-5:
        scale_factor = max(0.25, min(target_height / source_height, 4.0))
        for root in roots:
            root.scale *= scale_factor
        context.view_layer.update()
        source_points = _objects_world_bound_points(imported_objects)

    target_basis = _profile_body_basis(target_armature, target_profile)
    left_local = target_basis @ mathutils.Vector((1.0, 0.0, 0.0))
    left_world = target_armature.matrix_world.to_3x3() @ left_local
    if left_world.length < 1e-6:
        left_world = mathutils.Vector((1.0, 0.0, 0.0))
    left_world.normalize()

    target_projection = [point.dot(left_world) for point in target_points]
    source_projection = [point.dot(left_world) for point in source_points]
    target_center = (min(target_projection) + max(target_projection)) * 0.5
    source_center = (min(source_projection) + max(source_projection)) * 0.5
    target_half = (max(target_projection) - min(target_projection)) * 0.5
    source_half = (max(source_projection) - min(source_projection)) * 0.5
    margin = max(target_height * 0.2, 0.15)
    desired_source_center = target_center - target_half - source_half - margin
    translation = left_world * (desired_source_center - source_center)
    for root in roots:
        root.matrix_world.translation += translation
    context.view_layer.update()
    return len(imported_objects)


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
    for chain_name, chain in _profile_chains(profile).items():
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


def _hand_mapping_diagnostics(armature, profile):
    result = {}
    for role in ("left_arm", "right_arm"):
        chain = _profile_chains(profile).get(role, [])
        if not chain:
            continue
        hand_name = chain[-1]
        rest_frame = _hand_frame(armature, hand_name, use_rest=True)
        pose_frame = _hand_frame(armature, hand_name, use_rest=False)
        result[role] = {
            "hand": hand_name,
            "digit_roots": _hand_digit_root_names(armature, hand_name),
            "thumb_root": _hand_thumb_root_name(armature, hand_name),
            "rest_palm_frame": (
                _matrix3_to_rows(rest_frame) if rest_frame is not None else None
            ),
            "pose_palm_frame": (
                _matrix3_to_rows(pose_frame) if pose_frame is not None else None
            ),
        }
    return result


def _retarget_setting_diagnostics(armature):
    return {
        "match_hand_orientation": bool(
            armature.get(_HAND_MATCH_ENABLED_PROP, False)),
        "hand_match_strength": float(
            armature.get(_HAND_MATCH_STRENGTH_PROP, 1.0)),
        "hand_match_algorithm_version": _HAND_MATCH_ALGORITHM_VERSION,
        "hand_roll_mode": armature.get(_HAND_ROLL_MODE_PROP, "FOREARM"),
        "hand_roll_limit": float(
            armature.get(_HAND_ROLL_LIMIT_PROP, 60.0)),
        "twist_chain_version": int(
            armature.get(_TWIST_CHAIN_VERSION_PROP, 0)),
        "collision_segment_mode": armature.get(
            _COLLISION_SEGMENT_MODE_PROP, "BONE_TAILS"),
        "upper_arm_collision": armature.get(
            _UPPER_ARM_COLLISION_PROP, "DISABLED"),
        "arm_correction_driver": armature.get(
            _ARM_CORRECTION_DRIVER_PROP, "AUTO"),
    }


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
            "hands": _hand_mapping_diagnostics(source_armature, source_profile),
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
        "schema_version": 2,
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
            "hands": _hand_mapping_diagnostics(armature, profile),
            "retarget_settings": _retarget_setting_diagnostics(armature),
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

        previous_reference = _load_reference_pose(armature)
        provenance = (previous_reference or {}).get("provenance")
        reference_pose = _capture_reference_pose(
            armature,
            label="stored_reference",
            provenance=provenance,
        )
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

    match_engine: bpy.props.EnumProperty(
        name="Reference Matching",
        items=(
            (
                "REFERENCE_AWARE",
                "Reference-Aware (Recommended)",
                "Use the analyzed target orientation and same-side semantic chains",
            ),
            (
                "LEGACY",
                "Legacy Compatibility",
                "Keep the previous manual side-swap behavior",
            ),
        ),
        default="REFERENCE_AWARE",
    )  # type: ignore

    match_strength: bpy.props.FloatProperty(
        name="Auto Match Strength",
        description="How strongly limb directions are matched to the source reference frame",
        default=1.0,
        min=0.0,
        max=1.0,
    )  # type: ignore

    match_hand_orientation: bpy.props.BoolProperty(
        name="Match Hand Orientation",
        description=(
            "Align palm frames when both rigs expose at least three digit branches; "
            "supports generated hands with additional digits"
        ),
        default=False,
    )  # type: ignore

    hand_match_strength: bpy.props.FloatProperty(
        name="Hand Match Strength",
        description="How strongly explicit palm-frame orientation is applied",
        default=1.0,
        min=0.0,
        max=1.0,
    )  # type: ignore

    hand_roll_mode: bpy.props.EnumProperty(
        name="Hand Roll Handling",
        description="Choose where palm roll is applied and how the mesh is protected",
        items=_HAND_ROLL_MODE_ITEMS,
        default="FOREARM",
    )  # type: ignore

    hand_roll_limit: bpy.props.FloatProperty(
        name="Maximum Hand Roll (Degrees)",
        description="Maximum axial palm correction transferred in Limited Roll mode",
        default=60.0,
        min=0.0,
        max=180.0,
    )  # type: ignore

    arm_correction_driver: bpy.props.EnumProperty(
        name="Arm Correction Driver",
        description="Choose which joint post-processing may rotate to position each wrist",
        items=_ARM_CORRECTION_DRIVER_ITEMS,
        default="AUTO",
    )  # type: ignore

    collision_segment_mode: bpy.props.EnumProperty(
        name="Proxy Segments",
        description="Choose how skeletal collision capsule endpoints are constructed",
        items=_COLLISION_SEGMENT_MODE_ITEMS,
        default="BONE_TAILS",
    )  # type: ignore

    upper_arm_collision: bpy.props.EnumProperty(
        name="Upper Arm Collision",
        description="Choose how much of the upper-arm capsule participates in collision tests",
        items=_UPPER_ARM_COLLISION_ITEMS,
        default="DISABLED",
    )  # type: ignore

    store_reference: bpy.props.BoolProperty(
        name="Store As Ref Pose",
        description="Save the matched target pose as the retarget reference pose immediately",
        default=True,
    )  # type: ignore

    protect_mesh: bpy.props.BoolProperty(
        name="Protect Body Clearance",
        description="Keep matched hands outside conservative torso and hand proxy colliders",
        default=True,
    )  # type: ignore

    clearance: bpy.props.FloatProperty(
        name="Body Clearance",
        description="Scale the conservative torso and hand separation margin",
        default=1.0,
        min=0.25,
        max=2.0,
    )  # type: ignore

    swap_target_sides: bpy.props.BoolProperty(
        name="Swap Target Left/Right",
        description="Manual override in addition to the target profile orientation correction",
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
        _initialize_collision_operator_settings(self, context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "match_engine")
        layout.prop(self, "match_strength")
        if self.match_engine == "REFERENCE_AWARE":
            layout.prop(self, "arm_correction_driver")
            layout.prop(self, "match_hand_orientation")
            if self.match_hand_orientation:
                layout.prop(self, "hand_match_strength")
                layout.prop(self, "hand_roll_mode")
                if self.hand_roll_mode == "LIMITED":
                    layout.prop(self, "hand_roll_limit")
                elif self.hand_roll_mode == "TWIST_CHAIN":
                    layout.label(
                        text="Creates deform bones and redistributes forearm weights",
                        icon="INFO",
                    )
        layout.prop(self, "store_reference")
        layout.prop(self, "protect_mesh")
        if self.protect_mesh:
            layout.prop(self, "clearance")
            if self.match_engine == "REFERENCE_AWARE":
                layout.prop(self, "collision_segment_mode")
                layout.prop(self, "upper_arm_collision")
        if self.match_engine == "LEGACY":
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

            effective_swap = (
                bool(self.swap_target_sides)
                if self.match_engine == "LEGACY"
                else False
            )
            pairs = _common_chain_pairs(
                source_profile,
                target_profile,
                swap_target_sides=effective_swap,
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

            effective_hand_match = (
                self.match_engine == "REFERENCE_AWARE"
                and self.match_hand_orientation
            )
            target_armature[_HAND_MATCH_ENABLED_PROP] = effective_hand_match
            target_armature[_HAND_MATCH_STRENGTH_PROP] = self.hand_match_strength
            target_armature[_HAND_ROLL_MODE_PROP] = self.hand_roll_mode
            target_armature[_HAND_ROLL_LIMIT_PROP] = self.hand_roll_limit

            collision_proxy = None
            if self.protect_mesh and self.match_engine == "REFERENCE_AWARE":
                target_armature[_COLLISION_SEGMENT_MODE_PROP] = (
                    self.collision_segment_mode)
                target_armature[_UPPER_ARM_COLLISION_PROP] = (
                    self.upper_arm_collision)
                target_armature[_ARM_CORRECTION_DRIVER_PROP] = (
                    self.arm_correction_driver)
                collision_proxy = _collision_proxy_for_armature(
                    target_armature,
                    target_profile,
                    segment_mode=self.collision_segment_mode,
                )

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
                swap_target_sides=effective_swap,
                protect_mesh=self.protect_mesh,
                clearance=self.clearance,
                collision_proxy=collision_proxy,
                arm_correction_driver=(
                    self.arm_correction_driver
                    if self.match_engine == "REFERENCE_AWARE"
                    else "SHOULDER"
                ),
                upper_arm_collision=self.upper_arm_collision,
                match_hand_orientation=effective_hand_match,
                hand_match_strength=self.hand_match_strength,
                hand_roll_mode=self.hand_roll_mode,
                hand_roll_limit=self.hand_roll_limit,
            )

            if self.store_reference:
                reference_pose = _capture_reference_pose(
                    target_armature,
                    label="matched_source_reference",
                    provenance=_reference_provenance(
                        self.filepath,
                        source_profile,
                        target_profile,
                        start,
                        "START_FRAME",
                        swap_target_sides=effective_swap,
                        match_hand_orientation=effective_hand_match,
                        hand_match_strength=self.hand_match_strength,
                        hand_roll_mode=self.hand_roll_mode,
                        hand_roll_limit=self.hand_roll_limit,
                        arm_correction_driver=self.arm_correction_driver,
                    ),
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


class StableGenToggleCollisionProxyOverlay(bpy.types.Operator):
    """Show collision capsules; blue is driver-only and gray is inactive."""

    bl_idname = "stablegen.toggle_collision_proxy_overlay"
    bl_label = "Show Collision Proxy"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if _collision_proxy_draw_handle is not None:
            return True
        if not _active_or_selected_armature(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def execute(self, context):
        if _collision_proxy_draw_handle is not None:
            _remove_collision_proxy_overlay()
            self.report({"INFO"}, "Collision proxy hidden")
            return {"FINISHED"}

        armature = _active_or_selected_armature(context)
        profile = _profile_for_armature(armature)
        if not _profile_is_usable(armature, profile):
            self.report({"ERROR"}, "Analyze or edit the rig mapping first.")
            return {"CANCELLED"}
        proxy = _collision_proxy_for_armature(armature, profile)
        if not proxy:
            self.report({"ERROR"}, "Could not generate a collision proxy for this rig.")
            return {"CANCELLED"}

        _ensure_collision_proxy_overlay(armature)
        capsule_count = len(_collision_proxy_bone_groups(proxy))
        print(
            f"[StableGen] Showing {capsule_count} collision capsule(s) "
            f"for {armature.name}."
        )
        self.report({"INFO"}, f"Showing {capsule_count} collision capsules")
        return {"FINISHED"}


class StableGenToggleSourcePreview(bpy.types.Operator):
    """Show or hide the retained source animation preview objects."""

    bl_idname = "stablegen.toggle_retarget_source_preview"
    bl_label = "Toggle Source Preview"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, _context):
        if not _source_preview_objects():
            cls.poll_message_set("Apply an animation with Show Source Preview enabled first")
            return False
        return True

    def execute(self, _context):
        objects = _source_preview_objects()
        hide = any(not obj.hide_viewport for obj in objects)
        for obj in objects:
            obj.hide_viewport = hide
        _tag_view3d_redraw()
        self.report({"INFO"}, "Source preview hidden" if hide else "Source preview shown")
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

    rotate_180: bpy.props.BoolProperty(
        name="Rotate Target Interpretation 180°",
        description=(
            "Treat the character as facing the opposite direction and update "
            "left/right pairing as one profile-level correction"
        ),
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
        layout.prop(self, "rotate_180")
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
            "orientation_correction": "ROTATE_180" if self.rotate_180 else "NORMAL",
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

    retarget_engine: bpy.props.EnumProperty(
        name="Retarget Engine",
        description="Choose the reference-aware solver or the previous compatibility behavior",
        items=(
            (
                "REST_SPACE",
                "Reference-Aware (Recommended)",
                "Transfer parent-local motion through matching source and target reference frames",
            ),
            (
                "LEGACY",
                "Legacy Compatibility",
                "Use the previous armature-space pose-delta behavior for comparison",
            ),
        ),
        default="REST_SPACE",
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

    auto_match_reference: bpy.props.BoolProperty(
        name="Auto-Match First Frame",
        description=(
            "Temporarily reshape the target reference to resemble the source first frame; "
            "disable this to preserve the selected target rest/stored reference pose"
        ),
        default=True,
    )  # type: ignore

    protect_auto_match: bpy.props.BoolProperty(
        name="Protect Body Clearance",
        description="Keep auto-matched hands outside conservative torso and hand proxy colliders",
        default=True,
    )  # type: ignore

    auto_match_clearance: bpy.props.FloatProperty(
        name="Body Clearance",
        description="Scale the conservative torso and hand separation margin",
        default=1.0,
        min=0.25,
        max=2.0,
    )  # type: ignore

    limb_match_strength: bpy.props.FloatProperty(
        name="Limb Direction Match",
        description=(
            "Stabilize arm and leg chain directions after rest-space rotation transfer; "
            "set to zero to test pure per-bone transfer"
        ),
        default=0.65,
        min=0.0,
        max=1.0,
    )  # type: ignore

    match_hand_orientation: bpy.props.BoolProperty(
        name="Match Hand Orientation",
        description=(
            "Align palm frames when both rigs expose at least three digit branches; "
            "supports generated hands with additional digits"
        ),
        default=False,
    )  # type: ignore

    hand_match_strength: bpy.props.FloatProperty(
        name="Hand Match Strength",
        description="How strongly explicit palm-frame orientation is applied",
        default=1.0,
        min=0.0,
        max=1.0,
    )  # type: ignore

    hand_roll_mode: bpy.props.EnumProperty(
        name="Hand Roll Handling",
        description="Choose where palm roll is applied and how the mesh is protected",
        items=_HAND_ROLL_MODE_ITEMS,
        default="FOREARM",
    )  # type: ignore

    hand_roll_limit: bpy.props.FloatProperty(
        name="Maximum Hand Roll (Degrees)",
        description="Maximum axial palm correction transferred in Limited Roll mode",
        default=60.0,
        min=0.0,
        max=180.0,
    )  # type: ignore

    arm_correction_driver: bpy.props.EnumProperty(
        name="Arm Correction Driver",
        description="Choose which joint post-processing may rotate to position each wrist",
        items=_ARM_CORRECTION_DRIVER_ITEMS,
        default="AUTO",
    )  # type: ignore

    collision_segment_mode: bpy.props.EnumProperty(
        name="Proxy Segments",
        description="Choose how skeletal collision capsule endpoints are constructed",
        items=_COLLISION_SEGMENT_MODE_ITEMS,
        default="BONE_TAILS",
    )  # type: ignore

    upper_arm_collision: bpy.props.EnumProperty(
        name="Upper Arm Collision",
        description="Choose how much of the upper-arm capsule participates in collision tests",
        items=_UPPER_ARM_COLLISION_ITEMS,
        default="DISABLED",
    )  # type: ignore

    collision_adaptation: bpy.props.BoolProperty(
        name="Skeletal Proxy Collisions",
        description=(
            "Generate lightweight capsules from the target mesh's skin weights and "
            "keep forearms/hands out of the torso and opposite arm while baking"
        ),
        default=True,
    )  # type: ignore

    collision_strength: bpy.props.FloatProperty(
        name="Collision Strength",
        description="How strongly each collision pass pushes intersecting arm capsules apart",
        default=0.75,
        min=0.0,
        max=1.0,
    )  # type: ignore

    collision_margin: bpy.props.FloatProperty(
        name="Collision Margin",
        description="Extra capsule spacing as a fraction of the character height",
        default=0.008,
        min=0.0,
        max=0.05,
        precision=3,
    )  # type: ignore

    collision_iterations: bpy.props.IntProperty(
        name="Collision Passes",
        description="Maximum lightweight collision correction passes per baked frame",
        default=2,
        min=1,
        max=8,
    )  # type: ignore

    collision_max_deviation: bpy.props.FloatProperty(
        name="Maximum Correction",
        description="Maximum hand displacement from the transferred pose as a fraction of height",
        default=0.08,
        min=0.01,
        max=0.35,
        precision=3,
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

    show_source_preview: bpy.props.BoolProperty(
        name="Show Source Preview",
        description=(
            "Keep the imported source animation beside the target; both update from the same scene frame"
        ),
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
        _initialize_collision_operator_settings(self, context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "retarget_engine")
        layout.prop(self, "reference_mode")
        layout.prop(self, "source_reference_mode")
        if self.retarget_engine == "REST_SPACE" and self.source_reference_mode == "START_FRAME":
            layout.prop(self, "auto_match_reference")
            if self.auto_match_reference:
                layout.label(
                    text="First frame replaces the target neutral pose",
                    icon="INFO",
                )
                layout.prop(self, "protect_auto_match")
                if self.protect_auto_match:
                    layout.prop(self, "auto_match_clearance")
        if self.retarget_engine == "REST_SPACE":
            layout.prop(self, "limb_match_strength")
            layout.prop(self, "match_hand_orientation")
            if self.match_hand_orientation:
                layout.prop(self, "hand_match_strength")
                layout.prop(self, "hand_roll_mode")
                if self.hand_roll_mode == "LIMITED":
                    layout.prop(self, "hand_roll_limit")
                elif self.hand_roll_mode == "TWIST_CHAIN":
                    layout.label(
                        text="Creates deform bones and redistributes forearm weights",
                        icon="INFO",
                    )
            layout.prop(self, "arm_correction_driver")
            layout.prop(self, "collision_segment_mode")
            layout.prop(self, "upper_arm_collision")
            layout.prop(self, "collision_adaptation")
            if self.collision_adaptation:
                collision_box = layout.box()
                collision_box.prop(self, "collision_strength")
                collision_box.prop(self, "collision_margin")
                collision_box.prop(self, "collision_iterations")
                collision_box.prop(self, "collision_max_deviation")
        layout.prop(self, "frame_step")
        layout.prop(self, "replace_action")
        layout.prop(self, "copy_root_motion")
        if self.retarget_engine == "LEGACY":
            box = layout.box()
            box.label(text="Legacy compatibility overrides")
            box.prop(self, "swap_target_sides")
            box.prop(self, "flip_target_forward")
        layout.prop(self, "show_source_preview")
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

        can_auto_match_missing_reference = (
            self.retarget_engine == "REST_SPACE"
            and self.source_reference_mode == "START_FRAME"
            and self.auto_match_reference
        )
        if (
                self.reference_mode == "SAVED"
                and not _load_reference_pose(target_armature)
                and not can_auto_match_missing_reference):
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
        operation_succeeded = False
        try:
            if self.show_source_preview:
                _clear_source_preview()
            source_armature, imported_objects = _import_animation_source(self.filepath)
            source_profile = characterize_armature(source_armature)
            if not source_profile:
                raise RuntimeError("Could not infer a humanoid profile for the source animation rig")

            if self.retarget_engine == "LEGACY":
                effective_swap = bool(self.swap_target_sides)
                effective_flip = bool(self.flip_target_forward)
            else:
                effective_swap = False
                effective_flip = False
            pairs = _common_chain_pairs(
                source_profile,
                target_profile,
                swap_target_sides=effective_swap,
                source_armature=source_armature,
                target_armature=target_armature,
                trim_feet=True,
            )
            if len(pairs) < 6:
                raise RuntimeError(
                    "Not enough source/target bone pairs were found. "
                    "Try analyzing the rig or using a different source animation."
                )
            target_armature[_HAND_MATCH_ENABLED_PROP] = (
                self.retarget_engine == "REST_SPACE"
                and self.match_hand_orientation
            )
            target_armature[_HAND_MATCH_STRENGTH_PROP] = (
                self.hand_match_strength)
            target_armature[_HAND_ROLL_MODE_PROP] = self.hand_roll_mode
            target_armature[_HAND_ROLL_LIMIT_PROP] = self.hand_roll_limit
            collision_proxy = None
            if self.retarget_engine == "REST_SPACE":
                target_armature[_COLLISION_SEGMENT_MODE_PROP] = (
                    self.collision_segment_mode)
                target_armature[_UPPER_ARM_COLLISION_PROP] = (
                    self.upper_arm_collision)
                target_armature[_ARM_CORRECTION_DRIVER_PROP] = (
                    self.arm_correction_driver)
            needs_reference_proxy = (
                self.source_reference_mode == "START_FRAME"
                and self.auto_match_reference
                and self.protect_auto_match
            )
            if (
                    self.retarget_engine == "REST_SPACE"
                    and (self.collision_adaptation or needs_reference_proxy)):
                collision_proxy = _collision_proxy_for_armature(
                    target_armature,
                    target_profile,
                    segment_mode=self.collision_segment_mode,
                )
                if collision_proxy is None:
                    print(
                        "[StableGen] WARNING: Could not build a skeletal collision proxy; "
                        "continuing without per-frame proxy correction."
                    )
            print(f"[StableGen] Source rig profile for {source_armature.name}: {_profile_summary(source_profile)}")
            print(f"[StableGen] Target rig profile for {target_armature.name}: {_profile_summary(target_profile)}")
            print(f"[StableGen] Retarget mapping: {_pair_summary(pairs)}")
            print(f"[StableGen] Retarget engine: {self.retarget_engine}")
            print(f"[StableGen] {_axis_summary('Source', source_armature, source_profile)}")
            print(
                f"[StableGen] {_axis_summary('Target', target_armature, target_profile, effective_flip)}"
            )
            if effective_flip:
                print("[StableGen] Legacy retarget option: target forward axis flipped.")

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

            start, end = _action_frame_range(source_armature, context.scene)
            context.scene.frame_set(start)
            context.view_layer.update()
            if (
                    self.retarget_engine == "REST_SPACE"
                    and self.source_reference_mode == "START_FRAME"):
                expected_provenance = _reference_provenance(
                    self.filepath,
                    source_profile,
                    target_profile,
                    start,
                    self.source_reference_mode,
                    swap_target_sides=effective_swap,
                    match_hand_orientation=self.match_hand_orientation,
                    hand_match_strength=self.hand_match_strength,
                    hand_roll_mode=self.hand_roll_mode,
                    hand_roll_limit=self.hand_roll_limit,
                    arm_correction_driver=self.arm_correction_driver,
                )
                reference_matches, mismatch_reason = _reference_provenance_matches(
                    target_reference_pose,
                    expected_provenance,
                )
                if not reference_matches and self.auto_match_reference:
                    matched_chains = _match_source_reference_pose(
                        context,
                        source_armature,
                        target_armature,
                        pairs,
                        source_profile,
                        target_profile,
                        strength=1.0,
                        swap_target_sides=effective_swap,
                        protect_mesh=self.protect_auto_match,
                        clearance=self.auto_match_clearance,
                        collision_proxy=collision_proxy,
                        arm_correction_driver=self.arm_correction_driver,
                        upper_arm_collision=self.upper_arm_collision,
                        match_hand_orientation=self.match_hand_orientation,
                        hand_match_strength=self.hand_match_strength,
                        hand_roll_mode=self.hand_roll_mode,
                        hand_roll_limit=self.hand_roll_limit,
                    )
                    target_reference_pose = _capture_reference_pose(
                        target_armature,
                        label="auto_matched_source_reference",
                        provenance=expected_provenance,
                    )
                    reference_label = (
                        f"temporary source-matched reference ({matched_chains} limb chains)"
                    )
                    print(
                        f"[StableGen] {mismatch_reason}; automatically matched source frame {start}."
                    )
                elif not reference_matches:
                    warning = (
                        f"Reference mismatch: {mismatch_reason}. "
                        "Enable Auto-Match First Frame or run Match Source Ref."
                    )
                    print(f"[StableGen] WARNING: {warning}")
                    self.report({"WARNING"}, warning)

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

            action_name = (
                f"{target_armature.name}_"
                f"{os.path.splitext(os.path.basename(self.filepath))[0]}_"
                f"{self.retarget_engine.lower()}"
            )
            target_armature.animation_data_create()
            if self.replace_action or not target_armature.animation_data.action:
                target_armature.animation_data.action = bpy.data.actions.new(action_name)

            bpy.ops.object.select_all(action="DESELECT")
            target_armature.select_set(True)
            context.view_layer.objects.active = target_armature
            context.view_layer.update()
            _clear_retarget_constraints(target_armature)
            if self.retarget_engine == "LEGACY":
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
                    flip_target_forward=effective_flip,
                    source_reference_mode=self.source_reference_mode,
                )
            else:
                _bake_rest_space_retarget(
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
                    source_reference_mode=self.source_reference_mode,
                    swap_target_sides=effective_swap,
                    limb_match_strength=self.limb_match_strength,
                    collision_proxy=collision_proxy,
                    collision_adaptation=self.collision_adaptation,
                    collision_strength=self.collision_strength,
                    collision_margin=self.collision_margin,
                    collision_iterations=self.collision_iterations,
                    collision_max_deviation=self.collision_max_deviation,
                    arm_correction_driver=self.arm_correction_driver,
                    upper_arm_collision=self.upper_arm_collision,
                    match_hand_orientation=self.match_hand_orientation,
                    hand_match_strength=self.hand_match_strength,
                    hand_roll_mode=self.hand_roll_mode,
                    hand_roll_limit=self.hand_roll_limit,
                )

            context.scene.frame_set(start)
            print(
                f"[StableGen] Retargeted {len(pairs)} bone pair(s) "
                f"from {source_armature.name} to {target_armature.name}; "
                f"baked {self.retarget_engine.lower()} frames {start}-{end}."
            )
            self.report({"INFO"}, f"Animation baked to {target_armature.name}.")
            operation_succeeded = True
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
            if imported_objects:
                if self.show_source_preview and operation_succeeded:
                    preview_count = _prepare_source_preview(
                        context,
                        imported_objects,
                        target_armature,
                        target_profile,
                    )
                    print(
                        f"[StableGen] Source preview retained beside target "
                        f"({preview_count} object(s)); scene frame is synchronized."
                    )
                elif not self.keep_source_rig:
                    _delete_objects(imported_objects)
            bpy.ops.object.select_all(action="DESELECT")
            for obj in original_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            if original_active and original_active.name in bpy.data.objects:
                context.view_layer.objects.active = original_active
            context.scene.frame_set(original_frame)

        return {"FINISHED"}
