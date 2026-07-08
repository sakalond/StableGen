"""Compare source and retargeted leg motion numerically in background Blender."""

import argparse
import importlib
import json
import math
import os
import sys
import traceback
import types
from pathlib import Path

import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_retargeting_module():
    stablegen_pkg = types.ModuleType("stablegen")
    stablegen_pkg.__path__ = [str(REPO_ROOT / "stablegen")]
    animation_pkg = types.ModuleType("stablegen.animation")
    animation_pkg.__path__ = [str(REPO_ROOT / "stablegen" / "animation")]
    utils_module = types.ModuleType("stablegen.utils")
    utils_module.sg_modal_active = lambda _context: False
    sys.modules.setdefault("stablegen", stablegen_pkg)
    sys.modules.setdefault("stablegen.animation", animation_pkg)
    sys.modules.setdefault("stablegen.utils", utils_module)
    return importlib.import_module("stablegen.animation.retargeting")


rt = _load_retargeting_module()


VARIANTS = {
    "normal": (False, False),
    "swap": (True, False),
    "flip": (False, True),
    "swap_flip": (True, True),
}


def _args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Probe retargeted leg motion.")
    parser.add_argument("--bundle-json", required=True)
    parser.add_argument("--animation", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument(
        "--source-reference",
        default="start",
        choices=("rest", "start"),
        help="Source neutral baseline: source rig rest pose or first animation frame",
    )
    return parser.parse_args(argv)


def _clear_scene():
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_fbx_armature(filepath):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=filepath)
    new_objects = [obj for obj in bpy.data.objects if obj.name not in before]
    armatures = [obj for obj in new_objects if obj.type == "ARMATURE"]
    if not armatures:
        armatures = [obj for obj in bpy.context.selected_objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature imported from {filepath}")
    return armatures[0]


def _basis_vectors(armature, profile, flip_forward=False):
    basis = rt._profile_rotation_basis(  # pylint: disable=protected-access
        armature,
        profile,
        flip_forward=flip_forward,
    )
    lateral = basis @ mathutils.Vector((1.0, 0.0, 0.0))
    forward = basis @ mathutils.Vector((0.0, 1.0, 0.0))
    up = basis @ mathutils.Vector((0.0, 0.0, 1.0))
    return lateral.normalized(), forward.normalized(), up.normalized()


def _chain_endpoint(armature, profile, role):
    chain = profile.get("chains", {}).get(role, [])
    chain = rt._retarget_chain_for_role(  # pylint: disable=protected-access
        armature,
        profile,
        role,
        chain,
    )
    if not chain:
        return None
    return chain[-1]


def _pose_tail_or_head(armature, bone_name, use_tail=True):
    pose_bone = armature.pose.bones.get(bone_name)
    if not pose_bone:
        return None
    return pose_bone.tail.copy() if use_tail else pose_bone.head.copy()


def _sample_leg_motion(armature, profile, frames, flip_measure_forward=False):
    lateral, forward, up = _basis_vectors(
        armature,
        profile,
        flip_forward=flip_measure_forward,
    )
    pelvis_name = profile.get("pelvis")
    endpoints = {
        "left_leg": _chain_endpoint(armature, profile, "left_leg"),
        "right_leg": _chain_endpoint(armature, profile, "right_leg"),
    }
    samples = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        pelvis = _pose_tail_or_head(armature, pelvis_name, use_tail=False)
        if pelvis is None:
            continue
        row = {"frame": frame}
        for role, endpoint in endpoints.items():
            point = _pose_tail_or_head(armature, endpoint, use_tail=True)
            if point is None:
                continue
            offset = point - pelvis
            row[role] = {
                "forward": float(offset.dot(forward)),
                "lateral": float(offset.dot(lateral)),
                "up": float(offset.dot(up)),
                "endpoint": endpoint,
            }
        samples.append(row)
    return samples


def _series(samples, role, axis):
    result = []
    for row in samples:
        values = row.get(role)
        if values is not None:
            result.append(values[axis])
    return result


def _corr(a_values, b_values):
    n = min(len(a_values), len(b_values))
    if n < 2:
        return None
    a_values = a_values[:n]
    b_values = b_values[:n]
    mean_a = sum(a_values) / n
    mean_b = sum(b_values) / n
    da = [value - mean_a for value in a_values]
    db = [value - mean_b for value in b_values]
    denom_a = math.sqrt(sum(value * value for value in da))
    denom_b = math.sqrt(sum(value * value for value in db))
    if denom_a < 1e-8 or denom_b < 1e-8:
        return None
    return sum(a * b for a, b in zip(da, db)) / (denom_a * denom_b)


def _motion_scores(source_samples, target_samples):
    source_left = _series(source_samples, "left_leg", "forward")
    source_right = _series(source_samples, "right_leg", "forward")
    target_left = _series(target_samples, "left_leg", "forward")
    target_right = _series(target_samples, "right_leg", "forward")
    same_left = _corr(source_left, target_left)
    same_right = _corr(source_right, target_right)
    cross_left = _corr(source_left, target_right)
    cross_right = _corr(source_right, target_left)
    return {
        "same_left_corr": same_left,
        "same_right_corr": same_right,
        "cross_left_corr": cross_left,
        "cross_right_corr": cross_right,
        "same_avg": None if same_left is None or same_right is None else (same_left + same_right) / 2.0,
        "cross_avg": None if cross_left is None or cross_right is None else (cross_left + cross_right) / 2.0,
    }


def _frame_leg_order(samples):
    result = []
    for row in samples:
        left = row.get("left_leg", {}).get("forward")
        right = row.get("right_leg", {}).get("forward")
        if left is None or right is None:
            continue
        result.append({
            "frame": row["frame"],
            "left_forward": left,
            "right_forward": right,
            "leading": "left" if left > right else "right",
        })
    return result


def _reference_from_bundle(bundle, target_armature, target_names):
    stored = bundle.get("target", {}).get("stored_reference_pose")
    if rt._reference_pose_is_usable(target_armature, stored, target_names):  # pylint: disable=protected-access
        return stored, "stored"
    current = bundle.get("target", {}).get("current_reference_pose")
    if rt._reference_pose_is_usable(target_armature, current, target_names):  # pylint: disable=protected-access
        return current, "current"
    return rt._rest_reference_pose(target_armature, target_names), "rest"  # pylint: disable=protected-access


def _sample_baked_variant(bundle, target_fbx, animation_path, variant, source_reference_mode):
    swap_target_sides, flip_target_forward = VARIANTS[variant]
    _clear_scene()
    target_armature = _import_fbx_armature(target_fbx)
    target_profile = bundle["target"]["profile"]
    if not rt._profile_is_usable(target_armature, target_profile):  # pylint: disable=protected-access
        target_profile = rt.characterize_armature(target_armature)
    if not rt._profile_is_usable(target_armature, target_profile):  # pylint: disable=protected-access
        raise RuntimeError("Could not characterize imported target FBX")

    source_armature, imported_source = rt._import_animation_source(animation_path)  # pylint: disable=protected-access
    source_profile = rt.characterize_armature(source_armature)
    if not source_profile:
        raise RuntimeError("Could not infer source animation profile")

    pairs = rt._common_chain_pairs(  # pylint: disable=protected-access
        source_profile,
        target_profile,
        swap_target_sides=swap_target_sides,
        source_armature=source_armature,
        target_armature=target_armature,
        trim_feet=True,
    )
    start, end = rt._action_frame_range(source_armature, bpy.context.scene)  # pylint: disable=protected-access
    target_names = rt._mapped_target_bones(pairs)  # pylint: disable=protected-access
    reference_pose, reference_label = _reference_from_bundle(bundle, target_armature, target_names)

    target_armature.animation_data_create()
    target_armature.animation_data.action = bpy.data.actions.new(f"{variant}_probe")
    rt._bake_pose_delta_retarget(  # pylint: disable=protected-access
        bpy.context,
        source_armature,
        target_armature,
        pairs,
        source_profile,
        target_profile,
        start,
        end,
        1,
        copy_root_motion=False,
        target_reference_pose=reference_pose,
        flip_target_forward=flip_target_forward,
        source_reference_mode=source_reference_mode,
    )
    frames = list(range(start, end + 1))
    unflipped_samples = _sample_leg_motion(
        target_armature,
        target_profile,
        frames,
        flip_measure_forward=False,
    )
    flipped_samples = _sample_leg_motion(
        target_armature,
        target_profile,
        frames,
        flip_measure_forward=True,
    )
    return {
        "variant": variant,
        "swap_target_sides": swap_target_sides,
        "flip_target_forward": flip_target_forward,
        "reference": reference_label,
        "source_reference": source_reference_mode,
        "mapping": rt._pair_summary(pairs),  # pylint: disable=protected-access
        "unflipped_measure": {
            "scores": _motion_scores(bundle["_source_samples"], unflipped_samples),
            "forward_order": _frame_leg_order(unflipped_samples),
        },
        "flipped_measure": {
            "scores": _motion_scores(bundle["_source_samples"], flipped_samples),
            "forward_order": _frame_leg_order(flipped_samples),
        },
        "source_objects_removed": [obj.name for obj in imported_source],
    }


def main():
    args = _args()
    with open(args.bundle_json, "r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    runs_dir = os.path.abspath(args.runs_dir)
    source_reference_mode = "START_FRAME" if args.source_reference == "start" else "REST"
    target_fbx = bundle.get("files", {}).get("target_fbx")
    if not target_fbx or not os.path.exists(target_fbx):
        raise RuntimeError(f"Target FBX from bundle does not exist: {target_fbx}")

    _clear_scene()
    source_armature, imported = rt._import_animation_source(args.animation)  # pylint: disable=protected-access
    source_profile = rt.characterize_armature(source_armature)
    start, end = rt._action_frame_range(source_armature, bpy.context.scene)  # pylint: disable=protected-access
    frames = list(range(start, end + 1))
    source_samples = _sample_leg_motion(source_armature, source_profile, frames)
    rt._delete_objects(imported)  # pylint: disable=protected-access
    bundle["_source_samples"] = source_samples

    report = {
        "source": {
            "animation": os.path.abspath(args.animation),
            "frame_range": [start, end],
            "profile": rt._profile_summary(source_profile),  # pylint: disable=protected-access
            "reference": source_reference_mode,
            "forward_order": _frame_leg_order(source_samples),
        },
        "variants": [],
    }

    for variant in VARIANTS:
        try:
            report["variants"].append(
                _sample_baked_variant(
                    bundle,
                    target_fbx,
                    args.animation,
                    variant,
                    source_reference_mode,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            traceback.print_exc()
            report["variants"].append({"variant": variant, "error": str(exc)})

    output_path = os.path.join(runs_dir, "retarget_motion_probe.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(output_path)


if __name__ == "__main__":
    main()
