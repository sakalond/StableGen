"""Headless retarget debug runner.

Run with Blender, for example:

blender --background --python stablegen/animation/retarget_debug_runner.py -- \
  --bundle-json C:/tmp/Armature_diagnostics.json \
  --animation C:/tmp/walk.fbx \
  --output-dir C:/tmp/retarget_runs
"""

import argparse
import importlib
import json
import os
import sys
import traceback
import types
from pathlib import Path

import bpy  # pylint: disable=import-error


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_retargeting_module():
    """Load retargeting without importing the whole Blender add-on."""
    stablegen_pkg = types.ModuleType("stablegen")
    stablegen_pkg.__path__ = [str(REPO_ROOT / "stablegen")]
    animation_pkg = types.ModuleType("stablegen.animation")
    animation_pkg.__path__ = [str(REPO_ROOT / "stablegen" / "animation")]
    utils_module = types.ModuleType("stablegen.utils")

    def _sg_modal_active(_context):
        return False

    utils_module.sg_modal_active = _sg_modal_active
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

    parser = argparse.ArgumentParser(description="Run StableGen retarget debug variants.")
    parser.add_argument("--bundle-json", required=True, help="Diagnostics JSON exported from StableGen")
    parser.add_argument("--target-fbx", help="Target rig FBX override. Defaults to bundle JSON target_fbx")
    parser.add_argument("--animation", required=True, help="Source animation FBX/BVH")
    parser.add_argument("--output-dir", help="Where variant FBXs and report JSON should be written")
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variants: normal,swap,flip,swap_flip, or all",
    )
    parser.add_argument(
        "--reference",
        default="stored",
        choices=("stored", "current", "rest"),
        help="Target reference pose from the bundle",
    )
    parser.add_argument(
        "--source-reference",
        default="start",
        choices=("rest", "start"),
        help="Source neutral baseline: source rig rest pose or first animation frame",
    )
    parser.add_argument(
        "--engine",
        default="legacy",
        choices=("legacy", "rest_space"),
        help="Retarget engine used for the exported variants",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Bake every Nth frame")
    parser.add_argument("--copy-root-motion", action="store_true", help="Transfer pelvis/root location")
    return parser.parse_args(argv)


def _selected_variants(value):
    if value == "all":
        return list(VARIANTS.keys())
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise RuntimeError(f"Unknown variant(s): {', '.join(unknown)}")
    return names


def _clear_scene():
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_target_fbx(filepath):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=filepath)
    new_objects = [obj for obj in bpy.data.objects if obj.name not in before]
    armatures = [obj for obj in new_objects if obj.type == "ARMATURE"]
    if not armatures:
        armatures = [obj for obj in bpy.context.selected_objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature was imported from target FBX: {filepath}")
    return armatures[0], new_objects


def _export_result_fbx(armature, filepath):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in rt._rig_export_objects(armature):  # pylint: disable=protected-access
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.context.view_layer.update()
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
    )


def _reference_from_bundle(bundle, target_armature, target_names, mode):
    target = bundle.get("target", {})
    if mode == "stored":
        stored = target.get("stored_reference_pose")
        if rt._reference_pose_is_usable(target_armature, stored, target_names):  # pylint: disable=protected-access
            return stored, "stored"
    if mode in {"stored", "current"}:
        current = target.get("current_reference_pose")
        if rt._reference_pose_is_usable(target_armature, current, target_names):  # pylint: disable=protected-access
            return current, "current"
    return rt._rest_reference_pose(target_armature, target_names), "rest"  # pylint: disable=protected-access


def _run_variant(
        bundle, target_fbx, animation_path, output_dir, variant_name,
        frame_step, copy_root_motion, reference_mode, source_reference_mode,
        engine):
    swap_target_sides, flip_target_forward = VARIANTS[variant_name]
    _clear_scene()
    target_armature, _target_objects = _import_target_fbx(target_fbx)
    target_profile = bundle["target"]["profile"]
    if not rt._profile_is_usable(target_armature, target_profile):  # pylint: disable=protected-access
        target_profile = rt.characterize_armature(target_armature)
    if not rt._profile_is_usable(target_armature, target_profile):  # pylint: disable=protected-access
        raise RuntimeError(f"Could not characterize imported target armature {target_armature.name}")

    source_armature, imported_source = rt._import_animation_source(animation_path)  # pylint: disable=protected-access
    source_profile = rt.characterize_armature(source_armature)
    if not source_profile:
        raise RuntimeError("Could not infer a humanoid profile for the source animation")

    pairs = rt._common_chain_pairs(  # pylint: disable=protected-access
        source_profile,
        target_profile,
        swap_target_sides=swap_target_sides,
        source_armature=source_armature,
        target_armature=target_armature,
        trim_feet=True,
    )
    if len(pairs) < 6:
        raise RuntimeError(f"Not enough pairs for variant {variant_name}: {len(pairs)}")

    start, end = rt._action_frame_range(source_armature, bpy.context.scene)  # pylint: disable=protected-access
    target_names = rt._mapped_target_bones(pairs)  # pylint: disable=protected-access
    reference_pose, resolved_reference = _reference_from_bundle(
        bundle,
        target_armature,
        target_names,
        reference_mode,
    )

    target_armature.animation_data_create()
    output_stem = variant_name if engine == "legacy" else f"{engine}_{variant_name}"
    target_armature.animation_data.action = bpy.data.actions.new(
        f"{target_armature.name}_{output_stem}_retarget"
    )
    rt._clear_retarget_constraints(target_armature)  # pylint: disable=protected-access
    if engine == "legacy":
        rt._bake_pose_delta_retarget(  # pylint: disable=protected-access
            bpy.context,
            source_armature,
            target_armature,
            pairs,
            source_profile,
            target_profile,
            start,
            end,
            max(1, frame_step),
            copy_root_motion=copy_root_motion,
            target_reference_pose=reference_pose,
            flip_target_forward=flip_target_forward,
            source_reference_mode=source_reference_mode,
        )
    else:
        collision_proxy = rt._collision_proxy_for_armature(  # pylint: disable=protected-access
            target_armature, target_profile)
        rt._bake_rest_space_retarget(  # pylint: disable=protected-access
            bpy.context,
            source_armature,
            target_armature,
            pairs,
            source_profile,
            target_profile,
            start,
            end,
            max(1, frame_step),
            copy_root_motion=copy_root_motion,
            target_reference_pose=reference_pose,
            source_reference_mode=source_reference_mode,
            swap_target_sides=swap_target_sides,
            collision_proxy=collision_proxy,
        )

    output_fbx = os.path.join(output_dir, f"{output_stem}_retarget.fbx")
    _export_result_fbx(target_armature, output_fbx)
    return {
        "engine": engine,
        "variant": variant_name,
        "output_fbx": output_fbx,
        "swap_target_sides": swap_target_sides,
        "flip_target_forward": flip_target_forward,
        "reference": resolved_reference,
        "source_reference": source_reference_mode,
        "frame_range": [start, end],
        "pair_count": len(pairs),
        "mapping": rt._pair_summary(pairs),  # pylint: disable=protected-access
        "source_axes": rt._axis_summary("Source", source_armature, source_profile),  # pylint: disable=protected-access
        "target_axes": rt._axis_summary(  # pylint: disable=protected-access
            "Target",
            target_armature,
            target_profile,
            flip_forward=flip_target_forward,
        ),
        "source_objects_removed": [obj.name for obj in imported_source],
    }


def main():
    args = _args()
    with open(args.bundle_json, "r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    target_fbx = args.target_fbx or bundle.get("files", {}).get("target_fbx")
    if not target_fbx or not os.path.exists(target_fbx):
        raise RuntimeError(f"Target FBX does not exist: {target_fbx}")
    if not os.path.exists(args.animation):
        raise RuntimeError(f"Animation file does not exist: {args.animation}")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.bundle_json)),
        "retarget_runner_outputs",
    )
    os.makedirs(output_dir, exist_ok=True)

    source_reference_mode = "START_FRAME" if args.source_reference == "start" else "REST"

    report = {
        "bundle_json": os.path.abspath(args.bundle_json),
        "target_fbx": os.path.abspath(target_fbx),
        "animation": os.path.abspath(args.animation),
        "output_dir": os.path.abspath(output_dir),
        "source_reference": args.source_reference,
        "engine": args.engine,
        "variants": [],
    }

    for variant_name in _selected_variants(args.variants):
        print(f"[StableGen Retarget Runner] Running {variant_name}...")
        try:
            result = _run_variant(
                bundle,
                target_fbx,
                args.animation,
                output_dir,
                variant_name,
                args.frame_step,
                args.copy_root_motion,
                args.reference,
                source_reference_mode,
                args.engine,
            )
            report["variants"].append(result)
            print(f"[StableGen Retarget Runner] Wrote {result['output_fbx']}")
        except Exception as exc:  # pylint: disable=broad-except
            traceback.print_exc()
            report["variants"].append({
                "variant": variant_name,
                "error": str(exc),
            })

    report_path = os.path.join(output_dir, "retarget_runner_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"[StableGen Retarget Runner] Report: {report_path}")


if __name__ == "__main__":
    main()
