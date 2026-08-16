"""TRELLIS.2 mesh generation operator."""

import os
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import json
import uuid
import urllib.request
import urllib.parse
import threading
import traceback
from datetime import datetime
import math
import io
import websocket
from PIL import Image

from ..utils import get_generation_dirs, sg_modal_active
from ..timeout_config import get_timeout
from .._generator_utils import setup_studio_lighting, redraw_ui, upload_image_to_comfyui
from ..texturing.gallery import _PreviewGalleryOverlay
from .rig_display import (
    apply_autorig_bone_display,
    autorig_gltf_import_kwargs,
    configure_gltf_not_exported_visibility,
    record_autorig_bone_heuristic,
    selected_armatures_for_display,
)

_ADDON_PKG = __package__.rsplit('.', 1)[0]
_GARMENT_STABILIZED_GROUP = "SG_GarmentStabilized"


def _autorig_deforming_armature(mesh_object, imported_armatures):
    """Return the imported armature that supplies this mesh's skin groups."""
    for modifier in mesh_object.modifiers:
        if (
            modifier.type == 'ARMATURE'
            and modifier.object in imported_armatures
        ):
            return modifier.object

    parent = mesh_object.parent
    while parent:
        if parent.type == 'ARMATURE' and parent in imported_armatures:
            return parent
        parent = parent.parent
    return None


def _sanitize_autorig_cross_limb_weights(imported_objects):
    """Repair spatially implausible SkinTokens limb weights.

    SkinTokens can assign both feet to the same distal bone groups when the
    shoes are close, or weight hand vertices to a leg, pelvis, spine, or the
    opposite hand.  Those errors are often invisible in the bind pose but form
    long spikes once the limbs move.  This pass uses StableGen's semantic rig
    characterization plus rest-space bone proximity; it does not depend on
    generated bone names.

    The established cross-foot correction is retained.  A second general pass
    classifies only confident distal hand/foot regions and removes any weighted
    bone that is materially farther from the vertex than the nearest valid
    local limb bone.  Removed weight is assigned to that nearest bone, keeping
    the total armature weight unchanged.
    """
    from ..animation.retargeting import characterize_armature

    imported_objects = [
        obj for obj in imported_objects
        if obj and obj.name in bpy.data.objects
    ]
    imported_armatures = {
        obj for obj in imported_objects if obj.type == 'ARMATURE'
    }
    if not imported_armatures:
        return {"changed_vertices": 0, "moved_weight": 0.0, "meshes": 0}

    total_changed = 0
    total_moved = 0.0
    changed_meshes = 0

    for armature in imported_armatures:
        profile = characterize_armature(armature)
        chains = dict((profile or {}).get("chains", {}))
        if (profile or {}).get("orientation_correction") == "ROTATE_180":
            chains["left_leg"], chains["right_leg"] = (
                chains.get("right_leg", []),
                chains.get("left_leg", []),
            )
            chains["left_arm"], chains["right_arm"] = (
                chains.get("right_arm", []),
                chains.get("left_arm", []),
            )

        left_chain = [
            name for name in chains.get("left_leg", [])
            if armature.data.bones.get(name)
        ]
        right_chain = [
            name for name in chains.get("right_leg", [])
            if armature.data.bones.get(name)
        ]
        if len(left_chain) < 2 or len(right_chain) < 2:
            print(
                f"[StableGen] Cross-limb cleanup skipped for {armature.name}: "
                "paired leg chains were not characterized"
            )
            continue

        def bone_point(name, use_tail=False):
            bone = armature.data.bones[name]
            return (bone.tail_local if use_tail else bone.head_local).copy()

        left_foot = (
            bone_point(left_chain[-1]) + bone_point(left_chain[-1], True)
        ) * 0.5
        right_foot = (
            bone_point(right_chain[-1]) + bone_point(right_chain[-1], True)
        ) * 0.5
        side_vector = left_foot - right_foot
        side_separation = side_vector.length
        if side_separation <= 1e-6:
            print(
                f"[StableGen] Cross-limb cleanup skipped for {armature.name}: "
                "the characterized feet have no usable side separation"
            )
            continue
        side_axis = side_vector / side_separation
        side_origin = (left_foot + right_foot) * 0.5

        pelvis_bone = armature.data.bones.get((profile or {}).get("pelvis", ""))
        head_bone = armature.data.bones.get((profile or {}).get("head", ""))
        if pelvis_bone and head_bone:
            up_axis = (
                (head_bone.head_local + head_bone.tail_local) * 0.5
                - (pelvis_bone.head_local + pelvis_bone.tail_local) * 0.5
            )
        else:
            left_hip = bone_point(left_chain[0])
            right_hip = bone_point(right_chain[0])
            up_axis = (left_hip + right_hip) * 0.5 - side_origin
        if up_axis.length <= 1e-6:
            continue
        up_axis.normalize()

        hip_height = sum(
            bone_point(chain[0]).dot(up_axis)
            for chain in (left_chain, right_chain)
        ) * 0.5
        foot_height = sum(
            point.dot(up_axis) for point in (left_foot, right_foot)
        ) * 0.5
        if hip_height < foot_height:
            up_axis.negate()
            hip_height = -hip_height
            foot_height = -foot_height

        leg_height = hip_height - foot_height
        if leg_height <= 1e-6:
            continue

        # Slightly above the anatomical knee on typical 3- or 4-bone chains.
        # Restricting the pass to this region avoids changing skirt/torso skin.
        lower_body_cutoff = foot_height + leg_height * 0.58
        side_margin = side_separation * 0.08
        armature_inverse = armature.matrix_world.inverted_safe()

        chain_pairs = {}
        for source_chain, target_chain in (
            (left_chain, right_chain),
            (right_chain, left_chain),
        ):
            source_last = max(1, len(source_chain) - 1)
            target_last = max(1, len(target_chain) - 1)
            for index, source_name in enumerate(source_chain):
                target_index = round((index / source_last) * target_last)
                chain_pairs[source_name] = target_chain[target_index]

        def bone_segment_distance(point, bone_name):
            bone = armature.data.bones[bone_name]
            start = bone.head_local
            delta = bone.tail_local - start
            length_squared = delta.length_squared
            if length_squared <= 1e-12:
                return (point - start).length
            factor = max(
                0.0,
                min(1.0, (point - start).dot(delta) / length_squared),
            )
            return (point - (start + delta * factor)).length

        def descendants(root_name):
            result = set()
            root_bone = armature.data.bones.get(root_name)
            stack = list(root_bone.children) if root_bone else []
            while stack:
                child = stack.pop()
                if child.name in result:
                    continue
                result.add(child.name)
                stack.extend(child.children)
            return result

        def make_distal_region(label, chain, region_kind):
            chain = [
                name for name in chain if armature.data.bones.get(name)
            ]
            if len(chain) < 2:
                return None
            terminal_descendants = descendants(chain[-1])
            if region_kind == "hand":
                core_names = {chain[-1]} | terminal_descendants
                allowed_names = core_names | {chain[-2]}
            else:
                core_names = set(chain[-2:]) | terminal_descendants
                allowed_names = set(chain[-3:]) | terminal_descendants

            points = []
            for name in core_names:
                bone = armature.data.bones[name]
                points.extend((bone.head_local.copy(), bone.tail_local.copy()))
            if not points:
                return None
            center = sum(points, mathutils.Vector()) / len(points)
            extent = max((point - center).length for point in points)
            proximal_length = armature.data.bones[chain[-2]].length
            radius = max(extent * 1.35, proximal_length * 0.65)
            if radius <= 1e-6:
                return None
            return {
                "label": label,
                "kind": region_kind,
                "owner_family": label.replace(
                    region_kind,
                    "arm" if region_kind == "hand" else "leg",
                ),
                "center": center,
                "radius": radius,
                "max_local_distance": radius * 0.55,
                "distance_margin": max(
                    extent * 0.08,
                    proximal_length * 0.035,
                    radius * 0.02,
                ),
                "core": core_names,
                "allowed": allowed_names,
            }

        distal_region_pairs = []
        for region_kind, chain_suffix in (("hand", "arm"), ("foot", "leg")):
            left_region = make_distal_region(
                f"left_{region_kind}",
                chains.get(f"left_{chain_suffix}", []),
                region_kind,
            )
            right_region = make_distal_region(
                f"right_{region_kind}",
                chains.get(f"right_{chain_suffix}", []),
                region_kind,
            )
            if left_region and right_region:
                distal_region_pairs.append((left_region, right_region))

        all_bone_names = {bone.name for bone in armature.data.bones}
        family_bones = {
            "spine": set(chains.get("spine", []))
            | {
                name for name in (
                    (profile or {}).get("pelvis"),
                    (profile or {}).get("head"),
                ) if name
            },
            "left_arm": set(chains.get("left_arm", []))
            | descendants((chains.get("left_arm") or [""])[-1]),
            "right_arm": set(chains.get("right_arm", []))
            | descendants((chains.get("right_arm") or [""])[-1]),
            "left_leg": set(chains.get("left_leg", []))
            | descendants((chains.get("left_leg") or [""])[-1]),
            "right_leg": set(chains.get("right_leg", []))
            | descendants((chains.get("right_leg") or [""])[-1]),
        }
        bone_family = {
            name: family
            for family, names in family_bones.items()
            for name in names
        }
        spatial_region_stats = {
            region["label"]: {"vertices": 0, "weight": 0.0}
            for pair in distal_region_pairs
            for region in pair
        }

        for mesh_object in imported_objects:
            if mesh_object.type != 'MESH':
                continue
            if _autorig_deforming_armature(mesh_object, imported_armatures) != armature:
                continue

            group_by_name = {
                group.name: group for group in mesh_object.vertex_groups
            }
            garment_marker = group_by_name.get(_GARMENT_STABILIZED_GROUP)
            garment_vertex_indices = set()
            if garment_marker:
                marker_index = garment_marker.index
                garment_vertex_indices = {
                    vertex.index
                    for vertex in mesh_object.data.vertices
                    if any(
                        element.group == marker_index and element.weight > 0.5
                        for element in vertex.groups
                    )
                }
            relevant_names = set(left_chain) | set(right_chain)
            for region_pair in distal_region_pairs:
                for region in region_pair:
                    relevant_names.update(region["allowed"])
                    relevant_names.update(region["core"])
            if not relevant_names.intersection(group_by_name):
                continue
            group_name_by_index = {
                group.index: group.name for group in mesh_object.vertex_groups
            }

            # Spatial proximity is ambiguous when hands rest beside a skirt or
            # coat.  Establish a topology-level owner family before changing
            # any weights.  A disconnected hand/finger surface is normally
            # dominated by its arm family, while nearby garment panels are
            # dominated by spine/leg families even if a few vertices carry bad
            # finger weights.  This prevents the garment from being reassigned
            # to the hand merely because the surfaces touch in bind space.
            component_roots = list(range(len(mesh_object.data.vertices)))

            def component_find(index):
                while component_roots[index] != index:
                    component_roots[index] = component_roots[
                        component_roots[index]
                    ]
                    index = component_roots[index]
                return index

            def component_union(first, second):
                first_root = component_find(first)
                second_root = component_find(second)
                if first_root != second_root:
                    component_roots[second_root] = first_root

            for edge in mesh_object.data.edges:
                component_union(edge.vertices[0], edge.vertices[1])
            component_roots = [
                component_find(index) for index in range(len(component_roots))
            ]
            component_sizes = {}
            component_family_weights = {}
            for vertex, component_root in zip(
                mesh_object.data.vertices, component_roots
            ):
                component_sizes[component_root] = (
                    component_sizes.get(component_root, 0) + 1
                )
                weights_by_family = component_family_weights.setdefault(
                    component_root, {})
                for element in vertex.groups:
                    family = bone_family.get(
                        group_name_by_index.get(element.group)
                    )
                    if family:
                        weights_by_family[family] = (
                            weights_by_family.get(family, 0.0)
                            + float(element.weight)
                        )

            mesh_to_armature = armature_inverse @ mesh_object.matrix_world
            changed_vertex_indices = set()
            mesh_moved = 0.0
            left_set = set(left_chain)
            right_set = set(right_chain)

            for vertex in mesh_object.data.vertices:
                if vertex.index in garment_vertex_indices:
                    continue
                point = mesh_to_armature @ vertex.co
                if point.dot(up_axis) > lower_body_cutoff:
                    continue
                side_value = (point - side_origin).dot(side_axis)
                if abs(side_value) <= side_margin:
                    continue

                correct_set = left_set if side_value > 0.0 else right_set
                opposite_set = (
                    right_set - left_set
                    if side_value > 0.0
                    else left_set - right_set
                )
                weights = {
                    group_name_by_index.get(element.group): float(element.weight)
                    for element in vertex.groups
                    if group_name_by_index.get(element.group) in relevant_names
                }
                transfers = {}
                for opposite_name in opposite_set:
                    weight = weights.get(opposite_name, 0.0)
                    if weight <= 1e-6:
                        continue
                    target_name = chain_pairs.get(opposite_name)
                    if target_name not in correct_set:
                        continue
                    transfers[target_name] = transfers.get(target_name, 0.0) + weight
                    group_by_name[opposite_name].remove([vertex.index])
                    mesh_moved += weight

                if not transfers:
                    continue
                for target_name, weight in transfers.items():
                    if target_name not in group_by_name:
                        group_by_name[target_name] = mesh_object.vertex_groups.new(
                            name=target_name
                        )
                    current_weight = weights.get(target_name, 0.0)
                    group_by_name[target_name].add(
                        [vertex.index], current_weight + weight, 'REPLACE')
                changed_vertex_indices.add(vertex.index)

            # General rest-space plausibility pass.  It deliberately operates
            # only inside compact distal regions and requires one side to be a
            # clearly better geometric match than the other.  This catches
            # hand-to-leg, hand-to-spine, cross-hand, foot-to-pelvis, and
            # similar remote influences without imposing a global nearest-bone
            # rule on clothing or the torso.
            for vertex in mesh_object.data.vertices:
                if vertex.index in garment_vertex_indices:
                    continue
                point = mesh_to_armature @ vertex.co
                candidates = []
                for region_pair in distal_region_pairs:
                    pair_candidates = []
                    for region in region_pair:
                        if (point - region["center"]).length > region["radius"]:
                            pair_candidates.append(None)
                            continue
                        nearest_core_distance = min(
                            bone_segment_distance(point, name)
                            for name in region["core"]
                        )
                        if nearest_core_distance > region["max_local_distance"]:
                            pair_candidates.append(None)
                            continue
                        pair_candidates.append(
                            (nearest_core_distance / region["radius"], region)
                        )

                    valid = [entry for entry in pair_candidates if entry]
                    if not valid:
                        continue
                    valid.sort(key=lambda entry: entry[0])
                    best_score, best_region = valid[0]
                    if len(valid) > 1 and best_score + 0.08 >= valid[1][0]:
                        continue
                    candidates.append((best_score, best_region))

                if not candidates:
                    continue
                candidates.sort(key=lambda entry: entry[0])
                best_score, region = candidates[0]
                if len(candidates) > 1 and best_score + 0.06 >= candidates[1][0]:
                    continue

                component_root = component_roots[vertex.index]
                component_weights = component_family_weights.get(
                    component_root, {})
                component_total = sum(component_weights.values())
                owner_weight = component_weights.get(
                    region["owner_family"], 0.0)
                owner_share = (
                    owner_weight / component_total
                    if component_total > 1e-8 else 0.0
                )
                # Permit mixed hand/finger components when at least a quarter
                # of their consensus supports this side.  A tiny disconnected
                # digit with wholly corrupt weights may use a very strong
                # spatial match as fallback; large garment/body components may
                # not, because proximity is ambiguous there.
                if owner_share < 0.25 and not (
                    component_sizes.get(component_root, 0) <= 2048
                    and best_score <= 0.12
                ):
                    continue

                target_name, target_distance = min(
                    (
                        (name, bone_segment_distance(point, name))
                        for name in region["allowed"]
                    ),
                    key=lambda entry: entry[1],
                )
                initial_weights = {
                    group_name_by_index.get(element.group): float(element.weight)
                    for element in vertex.groups
                    if group_name_by_index.get(element.group) in all_bone_names
                }
                removed_weight = 0.0
                for source_name, weight in initial_weights.items():
                    if source_name in region["allowed"] or weight <= 1e-6:
                        continue
                    source_distance = bone_segment_distance(point, source_name)
                    plausible_limit = max(
                        target_distance * 1.75,
                        target_distance + region["distance_margin"],
                    )
                    if source_distance <= plausible_limit:
                        continue
                    group_by_name[source_name].remove([vertex.index])
                    removed_weight += weight

                if removed_weight <= 0.0:
                    continue
                if target_name not in group_by_name:
                    group_by_name[target_name] = mesh_object.vertex_groups.new(
                        name=target_name
                    )
                target_weight = initial_weights.get(target_name, 0.0)
                group_by_name[target_name].add(
                    [vertex.index], target_weight + removed_weight, 'REPLACE')
                changed_vertex_indices.add(vertex.index)
                mesh_moved += removed_weight
                spatial_region_stats[region["label"]]["vertices"] += 1
                spatial_region_stats[region["label"]]["weight"] += removed_weight

            if changed_vertex_indices:
                mesh_object.data.update()
                changed_meshes += 1
                total_changed += len(changed_vertex_indices)
                total_moved += mesh_moved

        for label, region_stats in spatial_region_stats.items():
            if region_stats["vertices"]:
                print(
                    f"[StableGen] Spatial weight cleanup {label}: "
                    f"{region_stats['vertices']} vertices, "
                    f"{region_stats['weight']:.3f} weight remapped"
                )

    stats = {
        "changed_vertices": total_changed,
        "moved_weight": total_moved,
        "meshes": changed_meshes,
    }
    print(
        "[StableGen] Limb-weight plausibility cleanup: "
        f"{total_changed} vertices across {changed_meshes} mesh(es), "
        f"{total_moved:.3f} total weight remapped"
    )
    return stats


def _stabilize_autorig_garment_weights(imported_objects, mode='STABILIZE'):
    """Create a conservative skeletal blend for long garment surfaces.

    A humanoid skeleton has no representation for free-hanging cloth.  Long
    dresses therefore inherit discontinuous left/right leg weights and open
    into a stretched membrane during wide strides.  This optional pass detects
    large connected lower-body surfaces with mixed spine/leg ownership, then
    replaces their bone weights with a smooth three-bone blend: pelvis plus
    both upper legs.  The waist stays mostly pelvis-anchored while the lower
    garment follows a softened average of the thighs.  ``LEG_FOLLOW`` keeps
    that waist anchor but gives the lower garment stronger, spatially smooth
    same-side thigh/shin ownership.  It also absorbs small disconnected
    surface islands close to a confidently detected garment component.
    """
    from ..animation.retargeting import characterize_armature
    from mathutils import kdtree  # pylint: disable=import-error

    mode = str(mode or 'STABILIZE').upper()
    if mode not in {'STABILIZE', 'LEG_FOLLOW'}:
        raise ValueError(f"Unknown garment weighting mode: {mode}")

    imported_objects = [
        obj for obj in imported_objects
        if obj and obj.name in bpy.data.objects
    ]
    imported_armatures = {
        obj for obj in imported_objects if obj.type == 'ARMATURE'
    }
    total_changed = 0
    total_components = 0
    total_absorbed_components = 0

    for armature in imported_armatures:
        profile = characterize_armature(armature)
        chains = dict((profile or {}).get("chains", {}))
        if (profile or {}).get("orientation_correction") == "ROTATE_180":
            chains["left_leg"], chains["right_leg"] = (
                chains.get("right_leg", []),
                chains.get("left_leg", []),
            )
        left_leg = [
            name for name in chains.get("left_leg", [])
            if armature.data.bones.get(name)
        ]
        right_leg = [
            name for name in chains.get("right_leg", [])
            if armature.data.bones.get(name)
        ]
        spine = [
            name for name in chains.get("spine", [])
            if armature.data.bones.get(name)
        ]
        pelvis_name = (profile or {}).get("pelvis")
        if (
            len(left_leg) < 2
            or len(right_leg) < 2
            or not pelvis_name
            or not armature.data.bones.get(pelvis_name)
        ):
            continue

        left_thigh = left_leg[0]
        right_thigh = right_leg[0]
        left_shin = left_leg[1]
        right_shin = right_leg[1]
        left_hip = armature.data.bones[left_thigh].head_local.copy()
        right_hip = armature.data.bones[right_thigh].head_local.copy()
        side_vector = left_hip - right_hip
        side_separation = side_vector.length
        if side_separation <= 1e-6:
            continue
        side_axis = side_vector / side_separation
        side_origin = (left_hip + right_hip) * 0.5

        pelvis_bone = armature.data.bones[pelvis_name]
        head_bone = armature.data.bones.get((profile or {}).get("head", ""))
        if head_bone:
            up_axis = (
                (head_bone.head_local + head_bone.tail_local) * 0.5
                - (pelvis_bone.head_local + pelvis_bone.tail_local) * 0.5
            )
        else:
            left_foot = armature.data.bones[left_leg[-1]].tail_local
            right_foot = armature.data.bones[right_leg[-1]].tail_local
            up_axis = side_origin - (left_foot + right_foot) * 0.5
        if up_axis.length <= 1e-6:
            continue
        up_axis.normalize()

        hip_height = side_origin.dot(up_axis)
        foot_height = sum(
            armature.data.bones[chain[-1]].tail_local.dot(up_axis)
            for chain in (left_leg, right_leg)
        ) * 0.5
        if hip_height < foot_height:
            up_axis.negate()
            hip_height = -hip_height
            foot_height = -foot_height
        leg_height = hip_height - foot_height
        if leg_height <= 1e-6:
            continue

        family_bones = {
            "spine": set(spine) | {pelvis_name},
            "left_leg": set(left_leg),
            "right_leg": set(right_leg),
            "left_arm": set(chains.get("left_arm", [])),
            "right_arm": set(chains.get("right_arm", [])),
        }
        bone_family = {
            name: family
            for family, names in family_bones.items()
            for name in names
        }
        all_bone_names = {bone.name for bone in armature.data.bones}
        armature_inverse = armature.matrix_world.inverted_safe()

        for mesh_object in imported_objects:
            if mesh_object.type != 'MESH':
                continue
            if _autorig_deforming_armature(
                mesh_object, imported_armatures
            ) != armature:
                continue
            if not mesh_object.data.edges:
                continue

            group_by_name = {
                group.name: group for group in mesh_object.vertex_groups
            }
            if not set((pelvis_name, left_thigh, right_thigh)).intersection(
                group_by_name
            ):
                continue
            group_name_by_index = {
                group.index: group.name for group in mesh_object.vertex_groups
            }
            mesh_to_armature = armature_inverse @ mesh_object.matrix_world

            parents = list(range(len(mesh_object.data.vertices)))

            def find(index):
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            def union(first, second):
                first_root = find(first)
                second_root = find(second)
                if first_root != second_root:
                    parents[second_root] = first_root

            for edge in mesh_object.data.edges:
                union(edge.vertices[0], edge.vertices[1])
            roots = [find(index) for index in range(len(parents))]

            component_data = {}
            points = [
                mesh_to_armature @ vertex.co
                for vertex in mesh_object.data.vertices
            ]
            for vertex, root, point in zip(
                mesh_object.data.vertices, roots, points
            ):
                data = component_data.setdefault(
                    root,
                    {
                        "vertices": [],
                        "families": {},
                        "min_height": float("inf"),
                        "max_height": float("-inf"),
                    },
                )
                data["vertices"].append(vertex.index)
                height = point.dot(up_axis)
                data["min_height"] = min(data["min_height"], height)
                data["max_height"] = max(data["max_height"], height)
                for element in vertex.groups:
                    family = bone_family.get(
                        group_name_by_index.get(element.group)
                    )
                    if family:
                        data["families"][family] = (
                            data["families"].get(family, 0.0)
                            + float(element.weight)
                        )

            minimum_component_size = max(
                300,
                int(len(mesh_object.data.vertices) * 0.0005),
            )
            garment_components = []
            for root, data in component_data.items():
                vertex_count = len(data["vertices"])
                if vertex_count < minimum_component_size:
                    continue
                family_total = sum(data["families"].values())
                if family_total <= 1e-8:
                    continue
                share = {
                    family: data["families"].get(family, 0.0) / family_total
                    for family in family_bones
                }
                leg_share = share["left_leg"] + share["right_leg"]
                arm_share = share["left_arm"] + share["right_arm"]
                vertical_span = data["max_height"] - data["min_height"]
                reaches_upper_leg = (
                    data["max_height"] >= foot_height + leg_height * 0.55
                )
                reaches_below_knee = (
                    data["min_height"] <= foot_height + leg_height * 0.65
                )
                mixed_lower_body = (
                    min(share["left_leg"], share["right_leg"]) >= 0.02
                    or share["spine"] >= 0.25
                )
                single_leg_surface = (
                    vertex_count < max(
                        8000,
                        int(len(mesh_object.data.vertices) * 0.01),
                    )
                    and max(share["left_leg"], share["right_leg"]) >= 0.80
                    and share["spine"] < 0.15
                )
                if not (
                    share["spine"] >= 0.08
                    and leg_share >= 0.25
                    and arm_share <= 0.08
                    and mixed_lower_body
                    and vertical_span >= leg_height * 0.22
                    and reaches_upper_leg
                    and reaches_below_knee
                    and not single_leg_surface
                ):
                    continue
                garment_components.append((root, data, share))

            if not garment_components:
                continue

            absorbed_components = 0
            if mode == 'LEG_FOLLOW':
                # Generated garments commonly contain narrow, disconnected
                # strips along their boundary.  They are too small to pass the
                # semantic component test above, but leaving their old (or
                # absent) weights creates the floating triangles visible once
                # the character leaves its rest pose.  Attach only small
                # lower-body islands that lie very close to a confident core.
                core_roots = {root for root, _data, _share in garment_components}
                core_vertices = [
                    vertex_index
                    for _root, data, _share in garment_components
                    for vertex_index in data["vertices"]
                ]
                proximity_tree = kdtree.KDTree(len(core_vertices))
                for tree_index, vertex_index in enumerate(core_vertices):
                    proximity_tree.insert(points[vertex_index], tree_index)
                proximity_tree.balance()

                island_distance = max(
                    side_separation * 0.16,
                    leg_height * 0.025,
                )
                island_size_limit = max(
                    minimum_component_size * 8,
                    int(len(mesh_object.data.vertices) * 0.02),
                )
                for root, data in component_data.items():
                    if root in core_roots:
                        continue
                    vertex_count = len(data["vertices"])
                    if vertex_count > island_size_limit:
                        continue
                    if data["max_height"] < foot_height - leg_height * 0.04:
                        continue
                    if data["min_height"] > hip_height + leg_height * 0.08:
                        continue

                    family_total = sum(data["families"].values())
                    if family_total > 1e-8:
                        share = {
                            family: data["families"].get(family, 0.0) / family_total
                            for family in family_bones
                        }
                        leg_share = share["left_leg"] + share["right_leg"]
                        arm_share = share["left_arm"] + share["right_arm"]
                        dominant_single_leg = (
                            max(share["left_leg"], share["right_leg"]) >= 0.72
                            and min(share["left_leg"], share["right_leg"]) < 0.08
                            and share["spine"] < 0.18
                        )
                        if (
                            arm_share > 0.35
                            or share["spine"] + leg_share < 0.45
                            or dominant_single_leg
                        ):
                            continue
                        allowed_distance = island_distance
                    else:
                        # An unweighted island must be almost touching the core
                        # before StableGen assumes it belongs to the garment.
                        share = {family: 0.0 for family in family_bones}
                        allowed_distance = island_distance * 0.45

                    nearest_distance = min(
                        proximity_tree.find(points[vertex_index])[2]
                        for vertex_index in data["vertices"]
                    )
                    if nearest_distance > allowed_distance:
                        continue
                    garment_components.append((root, data, share))
                    absorbed_components += 1

                total_absorbed_components += absorbed_components

            target_bones = [pelvis_name, left_thigh, right_thigh]
            if mode == 'LEG_FOLLOW':
                target_bones.extend((left_shin, right_shin))
            for target_name in target_bones:
                if target_name not in group_by_name:
                    group_by_name[target_name] = mesh_object.vertex_groups.new(
                        name=target_name
                    )
            garment_marker = group_by_name.get(_GARMENT_STABILIZED_GROUP)
            if not garment_marker:
                garment_marker = mesh_object.vertex_groups.new(
                    name=_GARMENT_STABILIZED_GROUP
                )

            changed_indices = set()
            if mode == 'LEG_FOLLOW':
                garment_indices = {
                    vertex_index
                    for _root, data, _share in garment_components
                    for vertex_index in data["vertices"]
                    if (
                        points[vertex_index].dot(up_axis)
                        <= hip_height + leg_height * 0.06
                        and points[vertex_index].dot(up_axis)
                        >= foot_height - leg_height * 0.04
                    )
                }
                knee_height = sum(
                    armature.data.bones[name].tail_local.dot(up_axis)
                    for name in (left_thigh, right_thigh)
                ) * 0.5
                if not foot_height < knee_height < hip_height:
                    knee_height = foot_height + leg_height * 0.52

                target_fractions = {}
                bone_totals = {}
                for vertex_index in garment_indices:
                    vertex = mesh_object.data.vertices[vertex_index]
                    point = points[vertex_index]
                    height = point.dot(up_axis)
                    downward = max(
                        0.0,
                        min(1.0, (hip_height - height) / (leg_height * 0.9)),
                    )
                    smooth_downward = downward * downward * (3.0 - 2.0 * downward)
                    follow_amount = max(
                        0.0,
                        min(1.0, (smooth_downward - 0.04) / 0.96),
                    )
                    follow_amount = (
                        follow_amount
                        * follow_amount
                        * (3.0 - 2.0 * follow_amount)
                    )
                    pelvis_fraction = 0.90 - 0.68 * follow_amount
                    side_value = (point - side_origin).dot(side_axis)
                    side_normalized = max(
                        -1.0,
                        min(1.0, side_value / max(side_separation * 0.62, 1e-6)),
                    )
                    side_strength = 0.12 + 0.36 * follow_amount
                    left_split = 0.5 + side_strength * side_normalized
                    left_split = max(0.04, min(0.96, left_split))

                    below_knee = max(
                        0.0,
                        min(
                            1.0,
                            (knee_height - height)
                            / max(knee_height - foot_height, 1e-6),
                        ),
                    )
                    below_knee = below_knee * below_knee * (3.0 - 2.0 * below_knee)
                    shin_share = 0.42 * below_knee * follow_amount
                    thigh_share = 1.0 - shin_share
                    leg_fraction = 1.0 - pelvis_fraction
                    target_fractions[vertex_index] = {
                        pelvis_name: pelvis_fraction,
                        left_thigh: leg_fraction * left_split * thigh_share,
                        right_thigh: leg_fraction * (1.0 - left_split) * thigh_share,
                        left_shin: leg_fraction * left_split * shin_share,
                        right_shin: leg_fraction * (1.0 - left_split) * shin_share,
                    }

                    current_bone_weights = {
                        group_name_by_index.get(element.group): float(element.weight)
                        for element in vertex.groups
                        if group_name_by_index.get(element.group) in all_bone_names
                    }
                    bone_total = sum(current_bone_weights.values())
                    if bone_total <= 1e-8:
                        bone_total = 1.0
                    bone_totals[vertex_index] = bone_total

                # Diffuse the analytic field over the garment topology.  This
                # preserves the broad vertical/lateral motion while removing
                # sharp per-triangle ownership changes from irregular meshes.
                neighbors = {index: [] for index in target_fractions}
                for edge in mesh_object.data.edges:
                    first, second = edge.vertices
                    if first in neighbors and second in neighbors:
                        neighbors[first].append(second)
                        neighbors[second].append(first)
                for _iteration in range(4):
                    smoothed = {}
                    for vertex_index, fractions in target_fractions.items():
                        adjacent = neighbors[vertex_index]
                        if not adjacent:
                            smoothed[vertex_index] = fractions
                            continue
                        adjacent_count = float(len(adjacent))
                        averaged = {
                            name: sum(
                                target_fractions[other][name]
                                for other in adjacent
                            ) / adjacent_count
                            for name in target_bones
                        }
                        blended = {
                            name: fractions[name] * 0.65 + averaged[name] * 0.35
                            for name in target_bones
                        }
                        blend_total = sum(blended.values())
                        smoothed[vertex_index] = {
                            name: weight / blend_total
                            for name, weight in blended.items()
                        }
                    target_fractions = smoothed

                changed_roots = set()
                for vertex_index, fractions in target_fractions.items():
                    vertex = mesh_object.data.vertices[vertex_index]
                    current_bone_weights = {
                        group_name_by_index.get(element.group): float(element.weight)
                        for element in vertex.groups
                        if group_name_by_index.get(element.group) in all_bone_names
                    }
                    bone_total = bone_totals[vertex_index]
                    target_weights = {
                        name: fraction * bone_total
                        for name, fraction in fractions.items()
                    }
                    for name in current_bone_weights:
                        group_by_name[name].remove([vertex_index])
                    for name, weight in target_weights.items():
                        if weight > 1e-8:
                            group_by_name[name].add(
                                [vertex_index], weight, 'REPLACE')
                    changed_indices.add(vertex_index)
                    changed_roots.add(roots[vertex_index])
                    garment_marker.add([vertex_index], 1.0, 'REPLACE')

                if changed_indices:
                    total_components += len(changed_roots)
                    print(
                        "[StableGen] Smooth garment leg follow: "
                        f"{len(changed_indices)} vertices across "
                        f"{len(changed_roots)} component(s), "
                        f"{absorbed_components} nearby island(s) absorbed"
                    )
            else:
                for root, data, share in garment_components:
                    component_changed = 0
                    for vertex_index in data["vertices"]:
                        vertex = mesh_object.data.vertices[vertex_index]
                        point = points[vertex_index]
                        height = point.dot(up_axis)
                        if height > hip_height + leg_height * 0.06:
                            continue
                        if height < foot_height - leg_height * 0.04:
                            continue

                        downward = max(
                            0.0,
                            min(1.0, (hip_height - height) / (leg_height * 0.9)),
                        )
                        smooth_downward = downward * downward * (3.0 - 2.0 * downward)
                        pelvis_fraction = 0.88 - 0.38 * smooth_downward
                        side_value = (point - side_origin).dot(side_axis)
                        side_normalized = max(
                            -1.0,
                            min(1.0, side_value / max(side_separation * 0.7, 1e-6)),
                        )
                        side_strength = 0.18 + 0.08 * smooth_downward
                        left_split = 0.5 + side_strength * side_normalized
                        left_split = max(0.2, min(0.8, left_split))

                        current_bone_weights = {
                            group_name_by_index.get(element.group): float(element.weight)
                            for element in vertex.groups
                            if group_name_by_index.get(element.group) in all_bone_names
                        }
                        bone_total = sum(current_bone_weights.values())
                        if bone_total <= 1e-8:
                            continue

                        leg_fraction = 1.0 - pelvis_fraction
                        target_weights = {
                            pelvis_name: pelvis_fraction * bone_total,
                            left_thigh: leg_fraction * left_split * bone_total,
                            right_thigh: leg_fraction * (1.0 - left_split) * bone_total,
                        }
                        target_names = set(target_weights)
                        already_stabilized = all(
                            (
                                name in target_names
                                or weight <= 1e-6
                            )
                            for name, weight in current_bone_weights.items()
                        ) and all(
                            abs(current_bone_weights.get(name, 0.0) - weight) <= 1e-5
                            for name, weight in target_weights.items()
                        )
                        if already_stabilized:
                            garment_marker.add([vertex_index], 1.0, 'REPLACE')
                            continue
                        for name in current_bone_weights:
                            group_by_name[name].remove([vertex_index])
                        for name, weight in target_weights.items():
                            if weight > 1e-8:
                                group_by_name[name].add(
                                    [vertex_index], weight, 'REPLACE')
                        changed_indices.add(vertex_index)
                        garment_marker.add([vertex_index], 1.0, 'REPLACE')
                        component_changed += 1

                    if component_changed:
                        total_components += 1
                        print(
                            "[StableGen] Garment stabilization: "
                            f"component {root}, {component_changed} vertices, "
                            f"spine {share['spine']:.0%}, "
                            f"legs {(share['left_leg'] + share['right_leg']):.0%}"
                        )
            if changed_indices:
                mesh_object.data.update()
                total_changed += len(changed_indices)

    print(
        "[StableGen] Garment weighting complete: "
        f"mode {mode}, {total_changed} vertices across "
        f"{total_components} component(s), "
        f"{total_absorbed_components} nearby island(s) absorbed"
    )
    return {
        "changed_vertices": total_changed,
        "components": total_components,
        "absorbed_components": total_absorbed_components,
    }

class Trellis2Generate(bpy.types.Operator):
    """Generate a 3D mesh from a reference image using TRELLIS.2 via ComfyUI.

    Requires the PozzettiAndrea/ComfyUI-TRELLIS2 custom node pack installed on the ComfyUI server.
    Uploads the input image, runs the full TRELLIS.2 pipeline (background removal, conditioning,
    shape generation, texture generation, GLB export), downloads the resulting GLB file, and
    imports it into the Blender scene."""
    bl_idname = "object.trellis2_generate"
    bl_label = "Generate 3D Mesh (TRELLIS.2)"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _error = None
    _glb_data = None
    _is_running = False
    _cancelled = False
    _active_ws = None  # WebSocket reference for cancel-time close
    _progress = 0.0
    _stage = "Initializing"
    workflow_manager: object = None

    # ── Preview gallery state ─────────────────────────────────────────
    _gallery_overlay: _PreviewGalleryOverlay | None = None
    _gallery_event: threading.Event | None = None
    _gallery_ready: bool = False
    _gallery_action: str | None = None  # 'select' | 'more' | 'cancel'
    _gallery_selected_bytes: bytes | None = None
    _gallery_selected_seed: int | None = None
    _progress_remap: tuple | None = None  # (base, span) for gallery sub-range scaling

    # ── 3-tier progress ──────────────────────────────────────────────
    _overall_progress: float = 0.0
    _overall_stage: str = "Initializing"
    _phase_progress: float = 0.0
    _phase_stage: str = ""
    _detail_progress: float = 0.0
    _detail_stage: str = ""
    _current_phase: int = 0
    _total_phases: int = 3  # 2 when gen_from == 'image'
    _phase_layout: str = "trellis_only"
    _autorig_phase: int = 0

    _PHASE_LAYOUTS = {
        'txt2img+trellis+rigging+texturing': ([1, 2, 3, 4], [0, 15, 60, 75], [15, 45, 15, 25]),
        'trellis+rigging+texturing': ([1, 2, 3], [0, 60, 75], [60, 15, 25]),
        'txt2img+trellis+texturing': ([1, 2, 3], [0, 15, 65], [15, 50, 35]),
        'trellis+texturing': ([1, 2], [0, 65], [65, 35]),
        'txt2img+trellis+rigging': ([1, 2, 3], [0, 15, 75], [15, 60, 25]),
        'trellis+rigging': ([1, 2], [0, 75], [75, 25]),
        'txt2img+trellis': ([1, 2], [0, 15], [15, 85]),
        'trellis_only': ([1], [0], [100]),
    }

    def _phase_spec(self):
        phases, starts, weights = self._PHASE_LAYOUTS.get(
            getattr(self, '_phase_layout', 'trellis_only'),
            self._PHASE_LAYOUTS['trellis_only'],
        )
        return dict(zip(phases, starts)), dict(zip(phases, weights))

    def _phase_start_pct(self, phase_number):
        starts, _weights = self._phase_spec()
        return starts.get(phase_number, 0.0)

    def _update_overall(self):
        """Recompute *_overall_progress* from current phase + phase progress."""
        starts, weights = self._phase_spec()
        s = starts.get(self._current_phase, 0)
        w = weights.get(self._current_phase, 0)
        self._overall_progress = s + (self._phase_progress / 100.0) * w
        self._overall_progress = max(0.0, min(self._overall_progress, 100.0))
        # Keep legacy _progress in sync for any code that reads it
        self._progress = self._overall_progress

    @staticmethod
    def _clear_imported_custom_normals(mesh_objects):
        """Clear TRELLIS/GLB custom normals without changing object hierarchy."""
        if not mesh_objects:
            return

        previous_active = bpy.context.view_layer.objects.active
        previous_selection = list(bpy.context.selected_objects)
        cleared = 0

        try:
            if bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        try:
            for obj in mesh_objects:
                if obj.type != 'MESH':
                    continue
                if hasattr(obj.data, "has_custom_normals") and not obj.data.has_custom_normals:
                    continue

                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                try:
                    bpy.ops.mesh.customdata_custom_splitnormals_clear()
                    cleared += 1
                except Exception as err:
                    print(f"[TRELLIS2] Warning: Could not clear custom normals on {obj.name}: {err}")
        finally:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in previous_selection:
                if bpy.context.scene.objects.get(obj.name):
                    obj.select_set(True)
            if previous_active and bpy.context.scene.objects.get(previous_active.name):
                bpy.context.view_layer.objects.active = previous_active

        if cleared:
            print(f"[TRELLIS2] Cleared custom split normals on {cleared} imported mesh object(s)")

    @staticmethod
    def _configure_not_exported_helpers():
        """Expose glTF helper visibility through eye/render collection icons."""
        stats = configure_gltf_not_exported_visibility(bpy.context)
        if stats["collections"]:
            print(
                "[TRELLIS2] Hid glTF not_exported helpers through "
                f"{stats['layer_collections']} view-layer eye toggle(s) and "
                f"{stats['collections']} collection render toggle(s); cleared "
                f"{stats['object_flags_cleared']} buried object flag(s)."
            )

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return True  # Allow cancellation
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        if not addon_prefs.server_address or not addon_prefs.server_online:
            cls.poll_message_set("ComfyUI server is not connected")
            return False
        if not os.path.exists(addon_prefs.output_dir):
            cls.poll_message_set("Output directory not set or does not exist (check addon preferences)")
            return False
        if bpy.app.online_access == False:
            cls.poll_message_set("Blender's online access is disabled (File → Preferences → System)")
            return False
        gen_from = getattr(context.scene, 'trellis2_generate_from', 'image')
        if gen_from == 'image' and not context.scene.trellis2_input_image:
            cls.poll_message_set("No input image selected for TRELLIS.2 generation")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def execute(self, context):
        if Trellis2Generate._is_running:
            # Cancel — tell the server to stop and close the WebSocket
            # so the background thread unblocks from ws.recv().
            Trellis2Generate._cancelled = True
            Trellis2Generate._is_running = False

            # Send /interrupt to ComfyUI (same as standard texturing cancel)
            try:
                server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
                data = json.dumps({"client_id": str(uuid.uuid4())}).encode('utf-8')
                req = urllib.request.Request("http://{}/interrupt".format(server_address), data=data)
                urllib.request.urlopen(req)
            except Exception:
                pass  # Best effort — server may already be gone

            # Close the active WebSocket so the thread's ws.recv() raises
            ws = Trellis2Generate._active_ws
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass
                Trellis2Generate._active_ws = None

            # Wake up the gallery event in case the thread is blocked there
            if self._gallery_event:
                self._gallery_action = 'cancel'
                self._gallery_event.set()

            self.report({'WARNING'}, "TRELLIS.2 generation cancelled")
            return {'FINISHED'}

        scene = context.scene
        gen_from = getattr(scene, 'trellis2_generate_from', 'image')
        tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')

        # Validate input image (only required in image mode)
        image_path = None
        if gen_from == 'image':
            image_path = bpy.path.abspath(scene.trellis2_input_image)
            if not os.path.exists(image_path):
                self.report({'ERROR'}, f"Input image not found: {image_path}")
                return {'CANCELLED'}

        Trellis2Generate._is_running = True
        context.scene.sg_last_gen_error = False
        self._error = None
        self._glb_data = None
        self._progress = 0.0
        self._stage = "Initializing"
        self._texture_mode = tex_mode
        self._autorig_enabled = False
        from ..workflows import WorkflowManager
        self.workflow_manager = WorkflowManager(self)

        # Gallery state reset
        self._gallery_overlay = None
        self._gallery_event = threading.Event()
        self._gallery_ready = False
        self._gallery_action = None
        self._gallery_selected_bytes = None
        self._gallery_selected_seed = None

        # 3-tier progress init
        has_txt2img = (gen_from == 'prompt')
        has_texturing = (tex_mode in ('sdxl', 'flux1', 'qwen_image_edit', 'flux2_klein'))
        has_autorig = bool(getattr(scene, 'trellis2_autorig_enabled', False))
        if has_txt2img and has_autorig and has_texturing:
            self._total_phases = 4
            self._phase_layout = 'txt2img+trellis+rigging+texturing'
        elif has_autorig and has_texturing:
            self._total_phases = 3
            self._phase_layout = 'trellis+rigging+texturing'
        elif has_txt2img and has_texturing:
            self._total_phases = 3
            self._phase_layout = 'txt2img+trellis+texturing'
        elif has_txt2img and has_autorig:
            self._total_phases = 3
            self._phase_layout = 'txt2img+trellis+rigging'
        elif has_autorig:
            self._total_phases = 2
            self._phase_layout = 'trellis+rigging'
        elif has_txt2img:
            self._total_phases = 2
            self._phase_layout = 'txt2img+trellis'
        elif has_texturing:
            self._total_phases = 2
            self._phase_layout = 'trellis+texturing'
        else:
            self._total_phases = 1
            self._phase_layout = 'trellis_only'
        self._current_phase = 0
        self._overall_progress = 0.0
        self._overall_stage = "Initializing"
        self._phase_progress = 0.0
        self._phase_stage = ""
        self._detail_progress = 0.0
        self._detail_stage = ""
        self._autorig_enabled = has_autorig
        self._autorig_phase = (2 if has_txt2img else 1) + 1 if has_autorig else 0

        # Compute revision directory on the main thread (may write output_timestamp)
        from ..utils import get_generation_dirs
        gen_dirs = get_generation_dirs(context)
        revision_dir = gen_dirs.get("revision", "")

        # Read parameters for background processing
        decimate_method = getattr(scene, 'trellis2_decimate_method', 'server')
        target_faces = getattr(scene, 'trellis2_decimation', 1000000)

        # Start generation in background thread
        self._thread = threading.Thread(
            target=self._run_trellis2,
            args=(context, image_path, gen_from, revision_dir, decimate_method, target_faces),
            daemon=True
        )
        self._thread.start()

        # Register modal timer
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)

        return {'RUNNING_MODAL'}

    def _cleanup_gallery(self):
        """Remove the gallery overlay and free GPU resources."""
        if self._gallery_overlay:
            self._gallery_overlay.cleanup()
            self._gallery_overlay = None

    def modal(self, context, event):
        # ── Gallery mode: intercept mouse + keyboard ──────────────
        if self._gallery_overlay is not None:
            if event.type == 'MOUSEMOVE':
                if self._gallery_overlay.handle_mouse_move(
                        event.mouse_region_x, event.mouse_region_y):
                    for area in context.screen.areas:
                        area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                action = self._gallery_overlay.handle_click(
                    event.mouse_region_x, event.mouse_region_y)
                if action == 'select':
                    self._gallery_selected_bytes = self._gallery_overlay.selected_image_bytes
                    self._gallery_selected_seed = self._gallery_overlay.selected_seed
                    self._gallery_action = 'select'
                    self._cleanup_gallery()
                    self._gallery_ready = False
                    self._gallery_event.set()
                    return {'RUNNING_MODAL'}
                elif action == 'more':
                    self._gallery_action = 'more'
                    self._cleanup_gallery()
                    self._gallery_ready = False
                    self._gallery_event.set()
                    return {'RUNNING_MODAL'}
                elif action == 'cancel':
                    self._gallery_action = 'cancel'
                    self._cleanup_gallery()
                    self._gallery_ready = False
                    self._gallery_event.set()
                    return {'RUNNING_MODAL'}
                return {'RUNNING_MODAL'}

            if event.type == 'ESC' and event.value == 'PRESS':
                self._gallery_action = 'cancel'
                self._cleanup_gallery()
                self._gallery_ready = False
                self._gallery_event.set()
                return {'RUNNING_MODAL'}

            if event.type == 'TIMER':
                for area in context.screen.areas:
                    area.tag_redraw()
            return {'RUNNING_MODAL'}

        # ── Normal mode ───────────────────────────────────────────
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # Redraw UI for progress updates
        for area in context.screen.areas:
            area.tag_redraw()

        # Check if gallery is ready (thread waiting for user input)
        if self._gallery_ready and self._gallery_overlay is None:
            gallery_data = getattr(self, '_gallery_data', None)
            if gallery_data:
                pil_imgs, seeds = gallery_data
                self._gallery_overlay = _PreviewGalleryOverlay(pil_imgs, seeds)
                for area in context.screen.areas:
                    area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Check if thread is still running
        if self._thread and self._thread.is_alive():
            return {'RUNNING_MODAL'}

        # Thread finished - clean up timer
        context.window_manager.event_timer_remove(self._timer)
        self._timer = None
        was_cancelled = Trellis2Generate._cancelled
        Trellis2Generate._is_running = False
        Trellis2Generate._cancelled = False
        Trellis2Generate._active_ws = None
        self._cleanup_gallery()

        # User cancelled — exit silently (no error toast)
        if was_cancelled:
            context.scene.generation_status = 'idle'
            context.scene.sg_last_gen_error = True
            return {'FINISHED'}

        if self._error:
            self.report({'ERROR'}, f"TRELLIS.2 error: {self._error}")
            context.scene.sg_last_gen_error = True
            return {'CANCELLED'}

        if self._glb_data is None or (isinstance(self._glb_data, dict) and "error" in self._glb_data):
            error_msg = self._glb_data.get("error", "Unknown error") if isinstance(self._glb_data, dict) else "No data received"
            self.report({'ERROR'}, f"TRELLIS.2 failed: {error_msg}")
            context.scene.sg_last_gen_error = True
            return {'CANCELLED'}

        # Surface mesh-corruption warning to the user (set by workflows.py
        # when the GLB validator detects artifacts but recovery failed).
        _mesh_warning = getattr(self, '_warning', None)
        if _mesh_warning:
            self.report({'WARNING'}, _mesh_warning)
            self._warning = None  # consumed

        # Save GLB to revision directory and import into Blender
        try:
            from ..utils import get_generation_dirs
            gen_dirs = get_generation_dirs(context)
            save_dir = gen_dirs.get("revision", "")
            if not save_dir:
                save_dir = context.preferences.addons[_ADDON_PKG].preferences.output_dir
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            glb_filename = f"trellis2_{timestamp}.glb"
            glb_path = os.path.join(save_dir, glb_filename)

            with open(glb_path, 'wb') as f:
                f.write(self._glb_data)

            # Store the TRELLIS.2 input image path for downstream use (IPAdapter/Qwen style)
            input_img = getattr(self, '_input_image_path', None)
            if input_img:
                context.scene.trellis2_last_input_image = input_img

            print(f"[TRELLIS2] Saved GLB to: {glb_path} ({len(self._glb_data)} bytes)")

            autorig_enabled = getattr(self, '_autorig_enabled', False)
            autorig_bone_heuristic = getattr(
                context.scene,
                'trellis2_autorig_bone_heuristic',
                'FORTUNE',
            )
            # Import rigged SkinTokens joints with child-directed bone tails.
            # Plain TRELLIS GLBs have no armature, so retain Blender's normal
            # import defaults for those files.
            if autorig_enabled:
                bpy.ops.import_scene.gltf(**autorig_gltf_import_kwargs(
                    glb_path, autorig_bone_heuristic))
            else:
                bpy.ops.import_scene.gltf(filepath=glb_path)

            # --- Clean hierarchy, unparent meshes, and remove GLTF/GLB structural empties ---
            imported_objects = [obj for obj in context.selected_objects]
            for obj in imported_objects:
                obj["sg_generated"] = True
            if autorig_enabled:
                record_autorig_bone_heuristic(
                    imported_objects, autorig_bone_heuristic)
                apply_autorig_bone_display(
                    imported_objects,
                    getattr(
                        context.scene,
                        'trellis2_autorig_bone_display_type',
                        'OCTAHEDRAL',
                    ),
                )
            self._configure_not_exported_helpers()
            mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
            empty_objects = [obj for obj in imported_objects if obj.type == 'EMPTY']

            if mesh_objects:
                # TRELLIS.2/GLB exports can carry broken custom split normals.
                # Clear them for both plain and rigged imports; this only
                # touches mesh data and leaves armature/object hierarchy intact.
                self._clear_imported_custom_normals(mesh_objects)

                if autorig_enabled:
                    # Rigged GLBs need their armature/object hierarchy intact.
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in imported_objects:
                        obj.select_set(True)
                    bpy.context.view_layer.objects.active = mesh_objects[0]
                else:
                    # 1. Unparent meshes while preserving their world space transforms
                    for obj in mesh_objects:
                        # Save world matrix
                        world_mat = obj.matrix_world.copy()
                        # Unparent
                        obj.parent = None
                        # Restore world matrix
                        obj.matrix_world = world_mat

                    # 2. Delete the structural empties (e.g. "world", "geometry_0")
                    if empty_objects:
                        bpy.ops.object.select_all(action='DESELECT')
                        for obj in empty_objects:
                            obj.select_set(True)
                        bpy.ops.object.delete()
                        print(f"[TRELLIS2] Cleaned up {len(empty_objects)} GLB parent empty objects")

                    # 3. Reselect only the imported mesh objects for downstream processing
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in mesh_objects:
                        obj.select_set(True)
                    bpy.context.view_layer.objects.active = mesh_objects[0]

            # --- Normalise imported mesh to a reasonable Blender-unit size ---
            target_bu = getattr(context.scene, 'trellis2_import_scale', 2.0)
            if target_bu > 0:
                imported_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
                if imported_meshes:
                    all_corners = []
                    for obj in imported_meshes:
                        for corner in obj.bound_box:
                            all_corners.append(obj.matrix_world @ mathutils.Vector(corner))
                    if all_corners:
                        xs = [c.x for c in all_corners]
                        ys = [c.y for c in all_corners]
                        zs = [c.z for c in all_corners]
                        extent = max(
                            max(xs) - min(xs),
                            max(ys) - min(ys),
                            max(zs) - min(zs),
                        )
                        if extent > 1e-6:
                            scale_factor = target_bu / extent
                            if autorig_enabled:
                                root_objects = [obj for obj in imported_objects if obj.parent is None]
                                if not root_objects:
                                    root_objects = imported_objects
                                for obj in root_objects:
                                    obj.scale *= scale_factor
                                print(
                                    f"[TRELLIS2] Scaled rigged import to {target_bu} BU "
                                    f"(factor {scale_factor:.4f}, unapplied)"
                                )
                            else:
                                for obj in imported_meshes:
                                    obj.scale *= scale_factor
                                # Apply scale so downstream code sees unit scale
                                bpy.ops.object.select_all(action='DESELECT')
                                for obj in imported_meshes:
                                    obj.select_set(True)
                                bpy.context.view_layer.objects.active = imported_meshes[0]
                                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                                print(f"[TRELLIS2] Scaled mesh to {target_bu} BU (factor {scale_factor:.4f})")

            if (
                autorig_enabled
                and getattr(
                    context.scene,
                    'trellis2_autorig_cleanup_cross_limb_weights',
                    True,
                )
            ):
                try:
                    _sanitize_autorig_cross_limb_weights(imported_objects)
                except Exception as exc:
                    print(
                        "[StableGen] Cross-limb weight cleanup failed; "
                        f"keeping the imported SkinTokens weights: {exc}"
                    )
                    traceback.print_exc()

            garment_mode = getattr(
                context.scene,
                'trellis2_autorig_garment_mode',
                'NONE',
            )
            if autorig_enabled and garment_mode in {'STABILIZE', 'LEG_FOLLOW'}:
                try:
                    _stabilize_autorig_garment_weights(
                        imported_objects,
                        mode=garment_mode,
                    )
                except Exception as exc:
                    print(
                        "[StableGen] Garment weighting failed; "
                        f"keeping the current skin weights: {exc}"
                    )
                    traceback.print_exc()

            # --- Solidification ---
            solidify_mesh = getattr(context.scene, 'trellis2_solidify', False)
            if autorig_enabled and solidify_mesh:
                print("[TRELLIS2] Skipping solidify because SkinTokens auto-rigging is enabled")
                solidify_mesh = False
            if solidify_mesh:
                from ..texturing.print_export import _make_solid_mesh_object
                imported_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
                for obj in imported_meshes:
                    try:
                        print(f"[TRELLIS2] Solidifying mesh: {obj.name} before decimation/retopology...")
                        solid_mesh = _make_solid_mesh_object(obj)
                        old_mesh = obj.data
                        obj.data = solid_mesh
                        bpy.data.meshes.remove(old_mesh)
                        print(f"[TRELLIS2] Solidified mesh: {obj.name} successfully")
                    except Exception as e:
                        print(f"[TRELLIS2] Error solidifying mesh {obj.name}: {e}")

            # --- Local Decimation & Retopology ---
            decimate_method = getattr(context.scene, 'trellis2_decimate_method', 'server')

            remesh_method = getattr(context.scene, 'trellis2_remesh_method', 'qdc')
            
            imported_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
            if imported_meshes and not autorig_enabled:
                # 1. Fallback decimation on main thread if background decimation was requested but didn't run
                if decimate_method == 'collapse' and not getattr(self, '_trimesh_decimated', False):
                    import trimesh
                    import numpy as np
                    import bmesh
                    
                    target_faces = context.scene.trellis2_decimation
                    for obj in imported_meshes:
                        mesh = obj.data
                        vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                        mesh.vertices.foreach_get("co", vertices)
                        vertices = vertices.reshape((-1, 3))
                        
                        bm = bmesh.new()
                        bm.from_mesh(mesh)
                        bmesh.ops.triangulate(bm, faces=bm.faces)
                        
                        faces = np.empty(len(bm.faces) * 3, dtype=np.int32)
                        for i, face in enumerate(bm.faces):
                            faces[i*3 : (i+1)*3] = [v.index for v in face.verts]
                        bm.free()
                        
                        t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces.reshape((-1, 3)))
                        if len(t_mesh.faces) > target_faces:
                            print(f"[TRELLIS2] Main thread fallback trimesh decimation: {len(t_mesh.faces)} -> {target_faces}")
                            decimated = t_mesh.simplify_quadric_decimation(face_count=target_faces)
                            mesh.clear_geometry()
                            mesh.from_pydata(decimated.vertices, [], decimated.faces.tolist())
                            mesh.update()
                
                # 2. Local remeshing retopology
                if remesh_method == 'quadriflow':
                    self._perform_retopology(context, imported_meshes, remesh_method)
            elif imported_meshes and autorig_enabled and (
                    decimate_method == 'collapse' or remesh_method == 'quadriflow'):
                print("[TRELLIS2] Skipping local decimation/retopology because SkinTokens auto-rigging is enabled")

            # --- Apply shading mode to imported meshes ---
            _shade_mode = getattr(context.scene, 'trellis2_shade_mode', 'flat')
            if _shade_mode == 'smooth':
                bpy.ops.object.shade_smooth()
            elif _shade_mode == 'flat':
                bpy.ops.object.shade_flat()
            elif _shade_mode == 'auto_smooth':
                if hasattr(bpy.ops.object, "shade_auto_smooth"):
                    bpy.ops.object.shade_auto_smooth()
                else:
                    bpy.ops.object.shade_smooth()

            # --- Optional studio lighting for native PBR textures ---
            tex_mode = getattr(self, '_texture_mode', 'native')
            if tex_mode == 'native' and getattr(context.scene, 'trellis2_auto_lighting', False):
                self._setup_studio_lighting(context, target_bu)

            # --- Phase 3: If diffusion texturing, auto-place cameras + start generation ---
            if tex_mode in ('sdxl', 'flux1', 'qwen_image_edit', 'flux2_klein'):
                # Place cameras NOW (while operator context is still valid)
                camera_count = getattr(context.scene, 'trellis2_camera_count', 8)
                imported_objects = [obj for obj in context.selected_objects]

                if imported_objects:
                    bpy.context.view_layer.objects.active = imported_objects[0]
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in imported_objects:
                        obj.select_set(True)

                # Force viewport to standard front view so AddCameras uses a
                # consistent reference direction for sorting and auto-prompts.
                # TRELLIS.2 always imports meshes in standard orientation so the
                # viewport should match.
                # Find the 3D viewport area + WINDOW region so add_cameras
                # gets a full context (region_data etc.) even when invoked
                # from a timer-driven modal callback.
                _v3d_area = None
                _v3d_region = None
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                rv3d = space.region_3d
                                if rv3d:
                                    # Blender front view (Numpad 1): -Y looking at +Y
                                    rv3d.view_rotation = mathutils.Quaternion(
                                        (0.7071068, 0.7071068, 0.0, 0.0)
                                    )
                                    rv3d.view_perspective = 'PERSP'
                        _v3d_area = area
                        for reg in area.regions:
                            if reg.type == 'WINDOW':
                                _v3d_region = reg
                                break
                        break

                try:
                    _pm = getattr(context.scene, 'trellis2_placement_mode', 'normal_weighted')
                    _cam_kwargs = {
                        'placement_mode': _pm,
                        'num_cameras': camera_count,
                        'auto_prompts': getattr(context.scene, 'trellis2_auto_prompts', True),
                        'review_placement': False,
                        'purge_others': True,
                        'exclude_bottom': getattr(context.scene, 'trellis2_exclude_bottom', True),
                        'exclude_bottom_angle': getattr(context.scene, 'trellis2_exclude_bottom_angle', 1.5533),
                        'auto_aspect': getattr(context.scene, 'trellis2_auto_aspect', 'per_camera'),
                        'occlusion_mode': getattr(context.scene, 'trellis2_occlusion_mode', 'none'),
                        'consider_existing': getattr(context.scene, 'trellis2_consider_existing', True),
                        'clamp_elevation': getattr(context.scene, 'trellis2_clamp_elevation', False),
                        'max_elevation_angle': getattr(context.scene, 'trellis2_max_elevation', 1.2217),
                        'min_elevation_angle': getattr(context.scene, 'trellis2_min_elevation', -0.1745),
                    }
                    if _pm == 'greedy_coverage':
                        _cam_kwargs['coverage_target'] = getattr(context.scene, 'trellis2_coverage_target', 0.95)
                        _cam_kwargs['max_auto_cameras'] = getattr(context.scene, 'trellis2_max_auto_cameras', 12)
                    if _pm == 'fan_from_camera':
                        _cam_kwargs['fan_angle'] = getattr(context.scene, 'trellis2_fan_angle', 90.0)

                    # Use temp_override so add_cameras gets proper region_data.
                    # Temporarily bypass the sg_modal_active() poll guard so
                    # add_cameras can run while this TRELLIS.2 modal is active.
                    from .. import utils as _sg_utils
                    _sg_utils._sg_bypass_modal_check = True
                    try:
                        if _v3d_area and _v3d_region:
                            with bpy.context.temp_override(area=_v3d_area, region=_v3d_region):
                                bpy.ops.object.add_cameras(**_cam_kwargs)
                        else:
                            bpy.ops.object.add_cameras(**_cam_kwargs)
                    finally:
                        _sg_utils._sg_bypass_modal_check = False

                except Exception as cam_err:
                    print(f"[TRELLIS2] Warning: Camera placement failed: {cam_err}")
                    traceback.print_exc()

                # Defer texture generation so Blender digests the new cameras
                self._schedule_texture_generation(context)
                self.report({'INFO'}, f"TRELLIS.2: Mesh imported. Camera placement done, texture generation starting...")
            else:
                self.report({'INFO'}, f"TRELLIS.2: Imported 3D mesh from {glb_filename}")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to import GLB: {e}")
            traceback.print_exc()
            return {'CANCELLED'}

    # -----------------------------------------------------------------
    # Studio lighting (three-point rig for PBR showcase)
    # -----------------------------------------------------------------
    def _setup_studio_lighting(self, context, import_scale):
        """Create a three-point studio lighting setup around the imported mesh."""
        return setup_studio_lighting(context, scale=import_scale)

    def _perform_retopology(self, context, meshes, remesh_method):
        """Perform high-quality local retopology on imported meshes.

        Applies shrinkwrap projection to recover fine details from the high-poly source mesh.
        """
        def progress_cb(pct, status_text):
            self._overall_stage = "Local Processing"
            self._phase_stage = status_text
            self._phase_progress = pct
            self._current_phase = self._total_phases
            self._update_overall()
            
        perform_local_retopology(context, meshes, remesh_method, progress_cb=progress_cb)

    def _schedule_texture_generation(self, context):
        """Defer texture generation via a timer so Blender can digest the new cameras.

        Camera placement has already happened in ``modal()``.  This only
        selects all cameras and starts ``object.test_stable``.
        Sets scene-level pipeline flags so the UI can show the overall
        progress bar on top of the ComfyUIGenerate bars.
        """
        # Compute the overall-% at which texturing begins
        phase_start = float(self._phase_start_pct(self._total_phases))

        scene = context.scene
        scene.trellis2_pipeline_active = True
        scene.trellis2_pipeline_phase_start_pct = phase_start
        scene.trellis2_pipeline_total_phases = self._total_phases

        def _deferred_generate():
            try:
                # Defensive VRAM flush before loading the diffusion checkpoint.
                # The TRELLIS post-generation flush should have freed VRAM,
                # but if it silently failed the models are still resident (Gap C).
                # Also clear history to release cached node outputs.
                try:
                    srv = bpy.context.preferences.addons[_ADDON_PKG].preferences.server_address
                    # 1. Set unload flags
                    flush_data = json.dumps({"unload_models": True, "free_memory": True}).encode('utf-8')
                    flush_req = urllib.request.Request(
                        f"http://{srv}/free", data=flush_data,
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(flush_req, timeout=get_timeout('api'))
                    # 2. Clear history/cache
                    hist_data = json.dumps({"clear": True}).encode('utf-8')
                    hist_req = urllib.request.Request(
                        f"http://{srv}/history", data=hist_data,
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(hist_req, timeout=get_timeout('api'))
                    # 3. Wait for VRAM release
                    import time
                    time.sleep(3)
                    print("[TRELLIS2] Pre-texturing VRAM flush sent (unload+history clear)")
                except Exception as flush_err:
                    print(f"[TRELLIS2] Pre-texturing flush warning: {flush_err}")

                bpy.ops.object.select_all(action='DESELECT')
                scene_cams = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
                for obj in scene_cams:
                    obj.select_set(True)

                from .. import utils as _sg_utils
                _sg_utils._sg_bypass_modal_check = True
                try:
                    bpy.ops.object.test_stable('INVOKE_DEFAULT')
                finally:
                    _sg_utils._sg_bypass_modal_check = False
                print("[TRELLIS2] Texture generation started")
            except Exception as e:
                print(f"[TRELLIS2] Warning: Texture generation failed to start: {e}")
                traceback.print_exc()
            return None  # Run once

        # Remember which cameras exist before texturing so we can delete
        # the ones we placed if the user opted in.
        _pre_tex_cameras = {obj.name for obj in bpy.context.scene.objects if obj.type == 'CAMERA'}

        def _pipeline_watcher():
            """Clear the pipeline flag when texturing finishes or is cancelled."""
            if bpy.context.scene.generation_status in ('idle', 'waiting'):
                bpy.context.scene.trellis2_pipeline_active = False
                print("[TRELLIS2] Pipeline complete — overall bar removed")

                # Delete auto-placed cameras if the user requested it
                if getattr(bpy.context.scene, 'trellis2_delete_cameras', False):
                    to_remove = [obj for obj in bpy.context.scene.objects
                                 if obj.type == 'CAMERA' and obj.name in _pre_tex_cameras]
                    if to_remove:
                        bpy.ops.object.select_all(action='DESELECT')
                        for obj in to_remove:
                            obj.select_set(True)
                        bpy.ops.object.delete()
                        print(f"[TRELLIS2] Deleted {len(to_remove)} auto-placed cameras")

                return None  # Stop timer
            return 1.0  # Check again in 1s

        bpy.app.timers.register(_deferred_generate, first_interval=0.5)
        bpy.app.timers.register(_pipeline_watcher, first_interval=2.0)

    def _run_trimesh_decimation(self, glb_bytes, target_faces):
        """Runs the trimesh decimation directly inside Blender's python (on the background thread)."""
        import io
        import trimesh
        
        try:
            print(f"[TRELLIS2] Starting trimesh decimation in Blender's python (target: {target_faces} faces)...")
            # Load glb from bytes using an in-memory file-like object
            glb_file = io.BytesIO(glb_bytes)
            scene = trimesh.load(glb_file, file_type='glb')
            
            if scene.is_empty:
                print("[TRELLIS2] Scene is empty")
                return None
                
            simplified_any = False
            for name, geom in list(scene.geometry.items()):
                if isinstance(geom, trimesh.Trimesh):
                    orig_faces = len(geom.faces)
                    if orig_faces > target_faces:
                        print(f"[TRELLIS2] Simplifying {name} directly in Blender Python: {orig_faces} -> {target_faces} faces")
                        scene.geometry[name] = geom.simplify_quadric_decimation(face_count=target_faces)
                        simplified_any = True
                    else:
                        print(f"[TRELLIS2] Skipping {name}: already has {orig_faces} faces (target {target_faces})")
            
            if simplified_any:
                # Export to bytes
                out_bytes = scene.export(file_type='glb')
                return out_bytes
            else:
                return glb_bytes
                
        except Exception as e:
            print(f"[TRELLIS2] Error during trimesh decimation: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _run_trellis2(self, context, image_path, gen_from, revision_dir, decimate_method, target_faces):
        """Background thread: runs the TRELLIS.2 pipeline.

        If *gen_from* is ``'prompt'`` the method first generates an input
        image via a lightweight txt2img ComfyUI workflow, saves it to the
        revision directory and passes that to the TRELLIS.2 mesh workflow.

        When the preview gallery is enabled (``trellis2_preview_gallery_enabled``),
        the prompt path generates N images with different seeds and pauses to
        let the user pick one via the viewport overlay before continuing.
        """
        import random as _rng
        try:
            # --- Phase 1: Image acquisition ---
            if gen_from == 'prompt':
                self._current_phase = 1
                self._phase_stage = "Generating Input Image"
                self._phase_progress = 0
                self._detail_progress = 0
                self._detail_stage = "Flushing stale models"
                self._overall_stage = f"Phase 1/{self._total_phases}: Input Image"
                self._update_overall()

                # Flush any stale models from prior runs before loading a
                # diffusion checkpoint for txt2img (Gap A).
                try:
                    server_addr = context.preferences.addons[_ADDON_PKG].preferences.server_address
                    self.workflow_manager._flush_comfyui_vram(server_addr, label="Pre-txt2img")
                except Exception:
                    pass

                self._detail_stage = "Starting txt2img"

                gallery_enabled = getattr(context.scene, 'trellis2_preview_gallery_enabled', False)
                gallery_count = max(1, int(getattr(context.scene, 'trellis2_preview_gallery_count', 4)))

                if gallery_enabled and gallery_count >= 1:
                    # ── Preview gallery loop ──────────────────────────
                    img_result = None  # will hold the chosen image bytes

                    # Seed a local RNG for deterministic gallery sequences.
                    # Same scene seed ➜ same gallery images every run.
                    base_seed = int(getattr(context.scene, 'seed', 0))
                    if base_seed == 0:
                        gallery_rng = _rng.Random()       # truly random
                    else:
                        gallery_rng = _rng.Random(base_seed)  # deterministic

                    while True:
                        pil_images = []
                        seeds = []
                        # Reset progress for each batch
                        self._phase_progress = 0
                        self._update_overall()
                        for i in range(gallery_count):
                            self._detail_stage = f"Generating preview {i + 1}/{gallery_count}"
                            # Set up remapping so WebSocket progress (0-100 per image)
                            # maps to the correct slice of the overall phase bar.
                            base = (i / gallery_count) * 90
                            span = (1 / gallery_count) * 90
                            self._progress_remap = (base, span)
                            self._phase_progress = base
                            self._update_overall()

                            rand_seed = gallery_rng.randint(1, 2**31 - 1)
                            result = self.workflow_manager.generate_txt2img(
                                context, seed_override=rand_seed)
                            if isinstance(result, dict) and "error" in result:
                                self._error = f"txt2img failed (seed {rand_seed}): {result['error']}"
                                return

                            pil_img = Image.open(io.BytesIO(result))
                            pil_images.append(pil_img)
                            seeds.append(rand_seed)

                        # Clear remapping before waiting
                        self._progress_remap = None

                        # Hand off to the main thread for user selection
                        self._gallery_data = (pil_images, seeds)
                        self._gallery_ready = True
                        self._detail_stage = "Waiting for selection"
                        self._phase_progress = 95
                        self._update_overall()

                        # Block until the modal sets the event
                        self._gallery_event.wait()
                        self._gallery_event.clear()

                        if self._gallery_action == 'select':
                            img_result = self._gallery_selected_bytes
                            chosen_seed = self._gallery_selected_seed
                            if chosen_seed is not None:
                                context.scene.seed = chosen_seed
                            break
                        elif self._gallery_action == 'more':
                            # Loop around and generate another batch
                            continue
                        else:  # cancel
                            self._error = "Preview gallery cancelled"
                            return

                    if img_result is None:
                        self._error = "No image selected from gallery"
                        return
                else:
                    # ── Single image (legacy path) ───────────────────
                    img_result = self.workflow_manager.generate_txt2img(context)
                    if isinstance(img_result, dict) and "error" in img_result:
                        self._error = f"txt2img failed: {img_result['error']}"
                        return

                # Phase 1 complete
                self._phase_progress = 100
                self._update_overall()

                # Early exit if cancelled during txt2img
                if self._cancelled:
                    return

                # Save the generated image bytes to the revision directory
                save_dir = revision_dir if revision_dir else (
                    context.preferences.addons[_ADDON_PKG].preferences.output_dir
                )
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(save_dir, f"trellis2_input_{timestamp}.png")
                with open(image_path, 'wb') as f:
                    f.write(img_result)
                print(f"[TRELLIS2] Saved txt2img result to: {image_path}")

                # Flush VRAM so the txt2img model (SDXL/Flux) is evicted
                # before TRELLIS loads its own models via raw PyTorch.
                # Without this, both models coexist and OOM on <=16 GB GPUs.
                # (Gap B – between txt2img and TRELLIS Phase 1)
                self._detail_stage = "Flushing txt2img models"
                try:
                    server_addr = context.preferences.addons[_ADDON_PKG].preferences.server_address
                    self.workflow_manager._flush_comfyui_vram(server_addr, label="Post-txt2img")
                except Exception:
                    pass

                # Early exit if cancelled during txt2img
                if self._cancelled:
                    return

            # --- Phase 2 (or 1 if no txt2img): TRELLIS.2 mesh generation ---
            trellis_phase = 2 if gen_from == 'prompt' else 1
            self._current_phase = trellis_phase
            self._phase_stage = "TRELLIS.2 Mesh Generation"
            self._phase_progress = 0
            self._detail_progress = 0
            self._detail_stage = "Uploading image"
            self._overall_stage = f"Phase {trellis_phase}/{self._total_phases}: 3D Mesh"
            self._update_overall()

            # Store the final input image path for later use (IPAdapter/Qwen style)
            self._input_image_path = image_path
            self._autorig_enabled = bool(getattr(context.scene, 'trellis2_autorig_enabled', False))

            result = self.workflow_manager.generate_trellis2(context, image_path)

            # Suppress error reporting when the user cancelled
            if self._cancelled:
                return

            if isinstance(result, dict) and "error" in result:
                self._error = result["error"]
            else:
                self._trimesh_decimated = False
                if decimate_method == 'collapse' and not self._autorig_enabled:
                    self._detail_stage = "Decimating mesh via trimesh"
                    self._phase_stage = "Trimesh decimation"
                    self._phase_progress = 90
                    self._update_overall()
                    
                    try:
                        decimated_bytes = self._run_trimesh_decimation(result, target_faces)
                        if decimated_bytes:
                            result = decimated_bytes
                            self._trimesh_decimated = True
                            print("[TRELLIS2] Background trimesh decimation completed successfully")
                    except Exception as dec_err:
                        print(f"[TRELLIS2] Background trimesh decimation failed: {dec_err}")
                        # Fallback: result is unmodified raw GLB
                elif decimate_method == 'collapse' and self._autorig_enabled:
                    print("[TRELLIS2] Skipping background trimesh decimation because SkinTokens auto-rigging is enabled")
                self._glb_data = result

        except Exception as e:
            if self._cancelled:
                return  # Swallow exceptions caused by cancel-time WS close
            self._error = str(e)
            traceback.print_exc()


class StableGenApplyBoneDisplay(bpy.types.Operator):
    """Apply the configured bone presentation to selected generated rigs."""

    bl_idname = "stablegen.apply_autorig_bone_display"
    bl_label = "Apply Bone Display"
    bl_description = (
        "Apply the selected SkinTokens bone display style to the selected "
        "armature or the armature driving the selected mesh"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            return False
        armatures = selected_armatures_for_display(context)
        if not armatures:
            cls.poll_message_set("Select an armature or a rigged mesh")
            return False
        return True

    def execute(self, context):
        armatures = selected_armatures_for_display(context)
        display_type = getattr(
            context.scene, 'trellis2_autorig_bone_display_type', 'OCTAHEDRAL')
        count = apply_autorig_bone_display(armatures, display_type)
        self.report(
            {'INFO'},
            f"Applied {display_type.title()} bone display to {count} armature(s)",
        )
        return {'FINISHED'}


class StableGenAutoRigSelectedMesh(bpy.types.Operator):
    """Run SkinTokens auto-rigging on the selected character mesh objects."""

    bl_idname = "object.stablegen_autorig_selected_mesh"
    bl_label = "Auto-Rig Selected Character"
    bl_description = (
        "Join copies of the selected character's mesh parts when needed, export them to GLB, "
        "run SkinTokens in ComfyUI, and import the rigged GLB"
    )
    bl_options = {'REGISTER', 'UNDO'}

    hide_source: bpy.props.BoolProperty(
        name="Preserve Original Source",
        description=(
            "Keep the original mesh objects and hide them with Blender's eye-style "
            "viewport toggle after the rigged result imports; when disabled, the "
            "original objects are deleted after a successful import"
        ),
        default=True,
    )  # type: ignore
    use_transfer: bpy.props.BoolProperty(
        name="Transfer Original",
        description="Transfer the generated rig back to the joined source mesh to preserve materials and scale",
        default=True,
    )  # type: ignore
    use_skeleton: bpy.props.BoolProperty(
        name="Use Existing Skeleton",
        description=(
            "Preserve an existing skeleton and generate only skin weights; "
            "when disabled, any existing armature and skin binding are ignored"
        ),
        default=False,
    )  # type: ignore
    use_postprocess: bpy.props.BoolProperty(
        name="Skin Postprocess",
        description="Enable SkinTokens voxel skin post-processing before export",
        default=False,
    )  # type: ignore
    cleanup_cross_limb_weights: bpy.props.BoolProperty(
        name="Clean Limb Weights",
        description=(
            "After import, repair spatially implausible hand and foot weights "
            "such as influences from the opposite limb, spine, pelvis, or legs"
        ),
        default=True,
    )  # type: ignore
    garment_mode: bpy.props.EnumProperty(
        name="Garment Handling",
        description="Optional post-import handling for long dresses and skirts",
        items=(
            ('NONE', "Original Weights", "Keep the generated garment weights"),
            (
                'STABILIZE',
                "Stabilized Dress",
                "Detect long garment surfaces and apply a smooth pelvis/thigh blend",
            ),
            (
                'LEG_FOLLOW',
                "Smooth Leg Follow",
                "Smoothly drive detected long garments from the pelvis, thighs, and lower legs",
            ),
        ),
        default='NONE',
    )  # type: ignore
    bone_display_type: bpy.props.EnumProperty(
        name="Bone Display",
        description="Viewport shape used for the imported SkinTokens bones",
        items=(
            ('OCTAHEDRAL', "Octahedral (Mixamo-like)", "Classic tapered Blender/Mixamo-style bones"),
            ('STICK', "Stick / Joint Circles", "Thin bone lines with circular joint markers"),
            ('BBONE', "B-Bone", "Box-like bendy-bone display"),
            ('ENVELOPE', "Envelope", "Display bone deformation envelopes"),
            ('WIRE', "Wire", "Wireframe bone display"),
            ('IMPORTED', "Imported", "Keep the display and custom shapes from the imported GLB"),
        ),
        default='OCTAHEDRAL',
    )  # type: ignore
    bone_heuristic: bpy.props.EnumProperty(
        name="Bone Direction",
        description=(
            "How Blender constructs display/deform bone axes from SkinTokens "
            "glTF joint transforms"
        ),
        items=(
            (
                'FORTUNE',
                "Fortune / Child-Directed",
                "Point one-child bones at their child joints; best-looking default, "
                "but non-uniform joint scaling can be less accurate",
            ),
            (
                'BLENDER',
                "Blender / Legacy Round-Trip",
                "Preserve glTF local joint axes and imported joint-marker shapes; "
                "best for glTF round trips",
            ),
            (
                'TEMPERANCE',
                "Temperance / Nearest Axis",
                "Point toward the closest local axis; safer for non-uniform scaling "
                "but less anatomically aligned than Fortune",
            ),
        ),
        default='FORTUNE',
    )  # type: ignore
    top_k: bpy.props.IntProperty(
        name="Top K",
        description="SkinTokens sampling top-k",
        default=5,
        min=1,
        max=200,
    )  # type: ignore
    top_p: bpy.props.FloatProperty(
        name="Top P",
        description="SkinTokens nucleus sampling top-p",
        default=0.95,
        min=0.1,
        max=1.0,
        step=0.01,
    )  # type: ignore
    temperature: bpy.props.FloatProperty(
        name="Temperature",
        description="SkinTokens sampling temperature",
        default=1.0,
        min=0.1,
        max=2.0,
        step=0.1,
    )  # type: ignore
    repetition_penalty: bpy.props.FloatProperty(
        name="Repetition Penalty",
        description="SkinTokens repetition penalty",
        default=2.0,
        min=0.5,
        max=3.0,
        step=0.1,
    )  # type: ignore
    num_beams: bpy.props.IntProperty(
        name="Beams",
        description="Number of candidate rig-token sequences retained during SkinTokens generation",
        default=10,
        min=1,
        max=20,
    )  # type: ignore
    model_ckpt: bpy.props.StringProperty(
        name="Model Checkpoint",
        description="Optional SkinTokens checkpoint path relative to SKINTOKENS_ROOT",
        default="",
    )  # type: ignore
    hf_path: bpy.props.StringProperty(
        name="HF Path",
        description="Optional Hugging Face path passed to SkinTokens",
        default="",
    )  # type: ignore

    _SETTING_BINDINGS = (
        ("use_transfer", "trellis2_autorig_use_transfer", True),
        ("use_skeleton", "trellis2_autorig_use_skeleton", False),
        ("use_postprocess", "trellis2_autorig_use_postprocess", False),
        (
            "cleanup_cross_limb_weights",
            "trellis2_autorig_cleanup_cross_limb_weights",
            True,
        ),
        ("garment_mode", "trellis2_autorig_garment_mode", "NONE"),
        (
            "bone_display_type",
            "trellis2_autorig_bone_display_type",
            "OCTAHEDRAL",
        ),
        (
            "bone_heuristic",
            "trellis2_autorig_bone_heuristic",
            "FORTUNE",
        ),
        ("top_k", "trellis2_autorig_top_k", 5),
        ("top_p", "trellis2_autorig_top_p", 0.95),
        ("temperature", "trellis2_autorig_temperature", 1.0),
        ("repetition_penalty", "trellis2_autorig_repetition_penalty", 2.0),
        ("num_beams", "trellis2_autorig_num_beams", 10),
        ("model_ckpt", "trellis2_autorig_model_ckpt", ""),
        ("hf_path", "trellis2_autorig_hf_path", ""),
    )

    _timer = None
    _thread = None
    _error = None
    _rigged_data = None
    _is_running = False
    _cancelled = False
    _active_ws = None
    _progress = 0.0
    _stage = "Initializing"
    _overall_progress = 0.0
    _overall_stage = "Initializing"
    _phase_progress = 0.0
    _phase_stage = ""
    _current_phase = 0
    _total_phases = 4
    workflow_manager: object = None

    _PHASE_WEIGHTS = (12.0, 75.0, 11.0, 2.0)

    def _set_pipeline_phase(self, phase, label, percent, detail=None):
        """Update standalone auto-rig overall and current-phase progress."""
        phase = max(1, min(int(phase), self._total_phases))
        percent = max(0.0, min(float(percent), 100.0))
        start = sum(self._PHASE_WEIGHTS[:phase - 1])
        overall = start + self._PHASE_WEIGHTS[phase - 1] * percent / 100.0

        self._current_phase = phase
        self._overall_progress = max(self._overall_progress, overall)
        self._overall_stage = f"Phase {phase}/{self._total_phases}: {label}"
        self._phase_progress = percent
        self._phase_stage = detail or label

        # Keep the original fields synchronized for integrations which still
        # inspect the standalone operator's legacy single-bar state.
        self._progress = self._overall_progress
        self._stage = self._phase_stage

    def _load_settings_from_scene(self, scene):
        for operator_name, scene_name, fallback in self._SETTING_BINDINGS:
            setattr(self, operator_name, getattr(scene, scene_name, fallback))

    def _store_settings_in_scene(self, scene):
        for operator_name, scene_name, _fallback in self._SETTING_BINDINGS:
            setattr(scene, scene_name, getattr(self, operator_name))

    def invoke(self, context, _event):
        # The same button doubles as Cancel while a job is running.  Do not put
        # a confirmation dialog in front of that time-sensitive action.
        if StableGenAutoRigSelectedMesh._is_running:
            return self.execute(context)
        self._load_settings_from_scene(context.scene)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        mesh_count = len(self._collect_character_meshes(context.selected_objects))
        if mesh_count > 1:
            layout.label(
                text=f"{mesh_count} mesh parts will be joined temporarily for SkinTokens",
                icon='MESH_DATA',
            )

        behavior_box = layout.box()
        behavior_box.label(text="Rigging Behavior", icon='ARMATURE_DATA')
        behavior_box.prop(self, "use_transfer")
        behavior_box.prop(self, "use_skeleton")
        behavior_box.prop(self, "use_postprocess")
        behavior_box.prop(self, "cleanup_cross_limb_weights")
        behavior_box.prop(self, "garment_mode")
        behavior_box.prop(self, "bone_heuristic")
        behavior_box.prop(self, "bone_display_type")
        behavior_box.prop(self, "hide_source")
        if self.hide_source:
            behavior_box.label(
                text="Original: preserved and hidden with the Outliner eye toggle",
                icon='HIDE_ON',
            )
        else:
            warning = behavior_box.row()
            warning.alert = True
            warning.label(
                text="Replace mode: original mesh objects are deleted after success",
                icon='ERROR',
            )

        sampling_box = layout.box()
        sampling_box.label(text="Sampling", icon='SETTINGS')
        grid = sampling_box.grid_flow(
            row_major=True,
            columns=2,
            even_columns=True,
            even_rows=False,
            align=True,
        )
        grid.prop(self, "top_k")
        grid.prop(self, "top_p")
        grid.prop(self, "temperature")
        grid.prop(self, "repetition_penalty")
        sampling_box.prop(self, "num_beams")

        model_box = layout.box()
        model_box.label(text="Optional Model Overrides", icon='FILE_BLEND')
        model_box.prop(self, "model_ckpt")
        model_box.prop(self, "hf_path")

    @staticmethod
    def _collect_character_meshes(selected_objects):
        """Return selected meshes plus meshes below selected hierarchy roots.

        Imported FBX/GLTF characters commonly expose an Empty or Armature as the
        convenient selectable root.  Walking its descendants lets the standalone
        operator handle those characters without requiring every part to be
        selected manually.  Object identity is used for de-duplication because a
        mesh can be both explicitly selected and below a selected root.
        """
        meshes = []
        seen = set()
        explicitly_selected = {
            obj.as_pointer() if hasattr(obj, "as_pointer") else id(obj)
            for obj in selected_objects
        }

        def visit(obj):
            object_key = obj.as_pointer() if hasattr(obj, "as_pointer") else id(obj)
            if object_key in seen:
                return
            seen.add(object_key)
            if obj.type == 'MESH':
                is_explicit = object_key in explicitly_selected
                preserved_source = bool(
                    obj.get("sg_autorig_preserved_source", False))
                disabled = bool(obj.hide_viewport or obj.hide_get())
                # A selected hierarchy root commonly contains both the visible
                # rigged result and the source copy hidden by a previous run.
                # Do not silently join that preserved copy back into the next
                # SkinTokens input.  Explicitly selecting the mesh itself still
                # allows the user to rerig the original on purpose.
                if is_explicit or not (preserved_source or disabled):
                    meshes.append(obj)
            for child in obj.children:
                visit(child)

        for selected in selected_objects:
            visit(selected)
        return meshes

    @staticmethod
    def _related_armatures(selected_objects, meshes):
        """Find armatures explicitly selected or used by the source meshes."""
        armatures = []
        seen = set()

        def add(obj):
            if not obj or obj.type != 'ARMATURE':
                return
            object_key = obj.as_pointer() if hasattr(obj, "as_pointer") else id(obj)
            if object_key not in seen:
                seen.add(object_key)
                armatures.append(obj)

        for obj in selected_objects:
            add(obj)
        for mesh in meshes:
            ancestor = mesh.parent
            while ancestor is not None:
                add(ancestor)
                ancestor = ancestor.parent
            for modifier in mesh.modifiers:
                if modifier.type == 'ARMATURE':
                    add(getattr(modifier, 'object', None))
        return armatures

    @staticmethod
    def _make_joined_export_copy(
            context, source_meshes, strip_existing_rig=False):
        """Create one temporary mesh from copies of the source mesh objects.

        Blender's join operation keeps the active copy's transform/modifiers and
        merges material slots and same-named vertex groups from the other copies.
        This gives SkinTokens the single mesh node it expects while leaving every
        source object and datablock untouched.  When ``strip_existing_rig`` is
        true, Armature modifiers and armature ancestry are removed from the
        copies so a previously rigged character is exported as plain rest mesh.

        Returns ``(joined_object, copied_mesh_datablocks)``.  The caller owns and
        must remove both after export.
        """
        copies = []
        copied_meshes = []
        try:
            for index, source in enumerate(source_meshes):
                duplicate = source.copy()
                duplicate.data = source.data.copy()
                duplicate.name = f"SG_AutoRigPart_{index:03d}"
                duplicate.data.name = f"SG_AutoRigPartMesh_{index:03d}"
                context.collection.objects.link(duplicate)
                duplicate.hide_select = False
                duplicate.hide_viewport = False
                duplicate.hide_render = False
                duplicate.hide_set(False)
                if strip_existing_rig:
                    world_matrix = duplicate.matrix_world.copy()
                    ancestor = duplicate.parent
                    has_armature_ancestor = False
                    while ancestor is not None:
                        if ancestor.type == 'ARMATURE':
                            has_armature_ancestor = True
                            break
                        ancestor = ancestor.parent
                    if has_armature_ancestor:
                        duplicate.parent = None
                        duplicate.matrix_world = world_matrix
                    for modifier in list(duplicate.modifiers):
                        if modifier.type == 'ARMATURE':
                            duplicate.modifiers.remove(modifier)
                copies.append(duplicate)
                copied_meshes.append(duplicate.data)

            bpy.ops.object.select_all(action='DESELECT')
            for duplicate in copies:
                duplicate.select_set(True)

            # Prefer a mesh already driven by an armature as active so its
            # Armature modifier survives the join when use_skeleton is enabled.
            active_copy = next(
                (
                    duplicate for duplicate in copies
                    if any(modifier.type == 'ARMATURE' for modifier in duplicate.modifiers)
                ),
                copies[0],
            )
            context.view_layer.objects.active = active_copy
            if len(copies) > 1:
                bpy.ops.object.join()
            joined = context.view_layer.objects.active
            joined.name = "SG_AutoRigJoinedInput"
            joined.data.name = "SG_AutoRigJoinedInputMesh"
            joined["sg_autorig_temporary_join"] = True
            return joined, copied_meshes
        except Exception:
            for duplicate in copies:
                try:
                    if duplicate and duplicate.name in bpy.data.objects:
                        bpy.data.objects.remove(duplicate, do_unlink=True)
                except ReferenceError:
                    pass
            for mesh_data in copied_meshes:
                try:
                    if mesh_data and mesh_data.name in bpy.data.meshes and mesh_data.users == 0:
                        bpy.data.meshes.remove(mesh_data)
                except ReferenceError:
                    pass
            raise

    @staticmethod
    def _remove_joined_export_copy(joined, copied_meshes):
        try:
            if joined and joined.name in bpy.data.objects:
                bpy.data.objects.remove(joined, do_unlink=True)
        except ReferenceError:
            pass
        for mesh_data in copied_meshes:
            try:
                if mesh_data and mesh_data.name in bpy.data.meshes and mesh_data.users == 0:
                    bpy.data.meshes.remove(mesh_data)
            except ReferenceError:
                pass

    @staticmethod
    def _preserve_autorig_sources(source_names):
        """Keep source meshes with ordinary Outliner eye-style hiding."""
        preserved = 0
        for name in source_names:
            obj = bpy.data.objects.get(name)
            if not obj or obj.type != 'MESH':
                continue
            obj["sg_autorig_preserved_source"] = True
            # Older StableGen versions used hide_viewport, whose monitor toggle
            # is independent of the familiar Outliner eye.  Clear it before
            # applying the reversible eye-style hide.
            obj.hide_viewport = False
            obj.hide_set(True)
            obj.hide_render = True
            preserved += 1
        return preserved

    @staticmethod
    def _replace_autorig_sources(source_names, source_armature_names):
        """Delete superseded source meshes and now-unused source armatures."""
        removed_meshes = 0
        for name in source_names:
            obj = bpy.data.objects.get(name)
            if not obj or obj.type != 'MESH':
                continue
            bpy.data.objects.remove(obj, do_unlink=True)
            removed_meshes += 1

        removed_armatures = 0
        for name in source_armature_names:
            armature = bpy.data.objects.get(name)
            if not armature or armature.type != 'ARMATURE':
                continue

            still_used = False
            for obj in bpy.data.objects:
                if obj.type != 'MESH':
                    continue
                if any(
                        modifier.type == 'ARMATURE'
                        and getattr(modifier, 'object', None) == armature
                        for modifier in obj.modifiers):
                    still_used = True
                    break
                parent = obj.parent
                while parent is not None:
                    if parent == armature:
                        still_used = True
                        break
                    parent = parent.parent
                if still_used:
                    break

            if not still_used:
                bpy.data.objects.remove(armature, do_unlink=True)
                removed_armatures += 1

        return removed_meshes, removed_armatures

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return True
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        if not addon_prefs.server_address or not addon_prefs.server_online:
            cls.poll_message_set("ComfyUI server is not connected")
            return False
        if bpy.app.online_access == False:
            cls.poll_message_set("Blender's online access is disabled (File -> Preferences -> System)")
            return False
        if not getattr(context.scene, 'skintokens_autorig_available', False):
            cls.poll_message_set("StableGen SkinTokens node was not detected in ComfyUI")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        if not cls._collect_character_meshes(context.selected_objects):
            cls.poll_message_set("Select a mesh object or an imported character root to auto-rig")
            return False
        return True

    def execute(self, context):
        if StableGenAutoRigSelectedMesh._is_running:
            StableGenAutoRigSelectedMesh._cancelled = True
            StableGenAutoRigSelectedMesh._is_running = False

            try:
                server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
                data = json.dumps({"client_id": str(uuid.uuid4())}).encode('utf-8')
                req = urllib.request.Request(f"http://{server_address}/interrupt", data=data)
                urllib.request.urlopen(req, timeout=get_timeout('api'))
            except Exception:
                pass

            ws = StableGenAutoRigSelectedMesh._active_ws
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass
                StableGenAutoRigSelectedMesh._active_ws = None

            self.report({'WARNING'}, "Auto-rigging cancelled")
            return {'FINISHED'}

        selected_objects = list(context.selected_objects)
        selected_meshes = self._collect_character_meshes(selected_objects)
        if not selected_meshes:
            self.report({'ERROR'}, "Select a mesh object or an imported character root to auto-rig")
            return {'CANCELLED'}

        # Commit dialog values only after the user confirms and the selection is
        # valid.  Cancelling the dialog therefore leaves scene settings intact.
        self._store_settings_in_scene(context.scene)

        self.report(
            {'INFO'},
            "The exported GLB will be uploaded to the ComfyUI input directory",
        )

        try:
            gen_dirs = get_generation_dirs(context)
            save_dir = gen_dirs.get("revision", "") or context.preferences.addons[_ADDON_PKG].preferences.output_dir
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(save_dir, f"stablegen_autorig_source_{timestamp}.glb")

            previous_active = context.view_layer.objects.active
            joined_export = None
            copied_mesh_data = []
            try:
                related_armatures = self._related_armatures(
                    selected_objects,
                    selected_meshes,
                )
                strip_existing_rig = not self.use_skeleton
                needs_export_copy = (
                    len(selected_meshes) > 1
                    or (strip_existing_rig and bool(related_armatures))
                )
                if needs_export_copy:
                    joined_export, copied_mesh_data = self._make_joined_export_copy(
                        context,
                        selected_meshes,
                        strip_existing_rig=strip_existing_rig,
                    )
                    export_meshes = [joined_export]
                    if len(selected_meshes) > 1:
                        print(
                            f"[StableGen] Auto-rig staging joined {len(selected_meshes)} "
                            "source mesh objects into one temporary SkinTokens input"
                        )
                    if strip_existing_rig and related_armatures:
                        print(
                            "[StableGen] Auto-rig staging removed the existing "
                            "armature and skin binding from the SkinTokens input"
                        )
                else:
                    export_meshes = selected_meshes

                export_objects = list(export_meshes)
                if self.use_skeleton:
                    for armature in related_armatures:
                        if armature not in export_objects:
                            export_objects.append(armature)

                bpy.ops.object.select_all(action='DESELECT')
                for obj in export_objects:
                    obj.select_set(True)
                context.view_layer.objects.active = export_meshes[0]
                bpy.ops.export_scene.gltf(
                    filepath=export_path,
                    export_format='GLB',
                    use_selection=True,
                )
            finally:
                self._remove_joined_export_copy(joined_export, copied_mesh_data)
                bpy.ops.object.select_all(action='DESELECT')
                for obj in selected_objects:
                    if context.scene.objects.get(obj.name):
                        obj.select_set(True)
                if previous_active and context.scene.objects.get(previous_active.name):
                    context.view_layer.objects.active = previous_active

            if not os.path.isfile(export_path) or os.path.getsize(export_path) == 0:
                self.report({'ERROR'}, "Selected mesh export did not create a valid GLB")
                return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to export selected mesh for auto-rigging: {exc}")
            traceback.print_exc()
            return {'CANCELLED'}

        StableGenAutoRigSelectedMesh._is_running = True
        StableGenAutoRigSelectedMesh._cancelled = False
        self._error = None
        self._rigged_data = None
        self._progress = 0.0
        self._stage = "Queued"
        self._overall_progress = 0.0
        self._overall_stage = "Queued"
        self._phase_progress = 0.0
        self._phase_stage = "Preparing source"
        self._current_phase = 0
        self._source_names = [obj.name for obj in selected_meshes]
        self._source_armature_names = [
            obj.name for obj in related_armatures
        ]
        self._source_mesh_count = len(selected_meshes)
        self._export_path = export_path

        from ..workflows import WorkflowManager
        self.workflow_manager = WorkflowManager(self)
        self._thread = threading.Thread(
            target=self._run_autorig,
            args=(context, export_path),
            daemon=True,
        )
        self._thread.start()

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        if len(selected_meshes) > 1:
            self.report(
                {'INFO'},
                f"Auto-rigging started with {len(selected_meshes)} joined mesh parts",
            )
        else:
            self.report({'INFO'}, "Auto-rigging started")
        return {'RUNNING_MODAL'}

    def _run_autorig(self, context, export_path):
        try:
            result = self.workflow_manager.generate_skintokens_autorig_file(context, export_path)
            if self._cancelled:
                return
            if isinstance(result, dict) and "error" in result:
                self._error = result["error"]
            else:
                self._rigged_data = result
        except Exception as exc:
            if self._cancelled:
                return
            self._error = str(exc)
            traceback.print_exc()

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        for area in context.screen.areas:
            area.tag_redraw()

        if self._thread and self._thread.is_alive():
            return {'RUNNING_MODAL'}

        context.window_manager.event_timer_remove(self._timer)
        self._timer = None
        was_cancelled = StableGenAutoRigSelectedMesh._cancelled
        StableGenAutoRigSelectedMesh._is_running = False
        StableGenAutoRigSelectedMesh._cancelled = False
        StableGenAutoRigSelectedMesh._active_ws = None

        if was_cancelled:
            return {'FINISHED'}
        if self._error:
            self.report({'ERROR'}, f"Auto-rigging failed: {self._error}")
            return {'CANCELLED'}
        if not self._rigged_data:
            self.report({'ERROR'}, "Auto-rigging failed: no rigged GLB was returned")
            return {'CANCELLED'}

        try:
            self._set_pipeline_phase(4, "Blender Import", 0, "Writing rigged GLB")
            gen_dirs = get_generation_dirs(context)
            save_dir = gen_dirs.get("revision", "") or context.preferences.addons[_ADDON_PKG].preferences.output_dir
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(save_dir, f"stablegen_autorig_rigged_{timestamp}.glb")
            with open(output_path, 'wb') as handle:
                handle.write(self._rigged_data)

            bpy.ops.import_scene.gltf(**autorig_gltf_import_kwargs(
                output_path, getattr(self, 'bone_heuristic', 'FORTUNE')))
            self._set_pipeline_phase(4, "Blender Import", 45, "Preparing imported rig")
            imported_objects = list(context.selected_objects)
            for obj in imported_objects:
                obj["sg_generated"] = True
            record_autorig_bone_heuristic(
                imported_objects,
                getattr(self, 'bone_heuristic', 'FORTUNE'),
            )
            apply_autorig_bone_display(
                imported_objects,
                getattr(self, 'bone_display_type', 'OCTAHEDRAL'),
            )

            Trellis2Generate._configure_not_exported_helpers()
            mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
            if mesh_objects:
                Trellis2Generate._clear_imported_custom_normals(mesh_objects)

            if self.cleanup_cross_limb_weights:
                self._set_pipeline_phase(4, "Blender Import", 65, "Cleaning limb weights")
                try:
                    _sanitize_autorig_cross_limb_weights(imported_objects)
                except Exception as exc:
                    self.report(
                        {'WARNING'},
                        "Cross-limb weight cleanup failed; imported the original "
                        f"SkinTokens weights: {exc}",
                    )
                    traceback.print_exc()

            if self.garment_mode in {'STABILIZE', 'LEG_FOLLOW'}:
                garment_stage = (
                    "Smoothing garment leg follow"
                    if self.garment_mode == 'LEG_FOLLOW'
                    else "Stabilizing garment weights"
                )
                self._set_pipeline_phase(4, "Blender Import", 82, garment_stage)
                try:
                    _stabilize_autorig_garment_weights(
                        imported_objects,
                        mode=self.garment_mode,
                    )
                except Exception as exc:
                    self.report(
                        {'WARNING'},
                        "Garment weighting failed; imported the current "
                        f"skin weights: {exc}",
                    )
                    traceback.print_exc()

            bpy.ops.object.select_all(action='DESELECT')
            for obj in imported_objects:
                obj.select_set(True)
            if mesh_objects:
                context.view_layer.objects.active = mesh_objects[0]
            elif imported_objects:
                context.view_layer.objects.active = imported_objects[0]

            source_names = list(getattr(self, '_source_names', []))
            if self.hide_source:
                preserved = self._preserve_autorig_sources(source_names)
                print(
                    f"[StableGen] Preserved {preserved} source mesh object(s) "
                    "with eye-style hiding"
                )
            else:
                removed_meshes, removed_armatures = self._replace_autorig_sources(
                    source_names,
                    list(getattr(self, '_source_armature_names', [])),
                )
                print(
                    f"[StableGen] Replace-original mode removed {removed_meshes} "
                    f"source mesh object(s) and {removed_armatures} unused "
                    "source armature object(s)"
                )

            source_count = int(getattr(self, '_source_mesh_count', 1))
            part_note = f" from {source_count} joined parts" if source_count > 1 else ""
            self._set_pipeline_phase(4, "Blender Import", 100, "Import complete")
            self.report(
                {'INFO'},
                f"Auto-rigged mesh imported{part_note}: {os.path.basename(output_path)}",
            )
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import rigged GLB: {exc}")
            traceback.print_exc()
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Standalone Local Post-Processing & Retopology Utilities
# ---------------------------------------------------------------------------

def perform_local_retopology(context, meshes, remesh_method, progress_cb=None):
    """Perform high-quality local retopology on meshes.
    
    Applies shrinkwrap projection to recover fine details from the high-poly source mesh.
    """
    import tempfile
    import subprocess
    import math
    import os
    import traceback
    
    print(f"[StableGen] Starting local retopology solver: remesh={remesh_method}")
    wm = context.window_manager
    wm.progress_begin(0, 100)
    
    try:
        for obj_idx, obj in enumerate(meshes):
            if obj.type != 'MESH':
                continue
                
            def update_progress(pct, status_text):
                base = (obj_idx / len(meshes)) * 100
                span = (1 / len(meshes)) * 100
                overall_pct = base + (pct / 100) * span
                
                wm.progress_update(overall_pct)
                context.workspace.status_text_set(f"StableGen [Local Retopo]: {status_text} ({int(overall_pct)}%)")
                
                if progress_cb:
                    progress_cb(pct, status_text)
                    
                # Force Blender to redraw all panels and viewports immediately
                try:
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                except Exception:
                    pass

            # 1. Duplicate original high-detail mesh as reference for shrinkwrap projection
            update_progress(10, "Preserving high-detail reference...")
            high_poly = obj.copy()
            high_poly.data = obj.data.copy()
            high_poly.name = f"{obj.name}_HighPoly_Reference"
            context.collection.objects.link(high_poly)
            high_poly.hide_set(True) # Hide reference mesh

            # Set active and select the low-poly target mesh
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj

            # 2. Solver Setup
            target_faces = context.scene.trellis2_decimation
            

            def run_quadriflow(faces_count):
                if faces_count > 30000:
                    print(f"[StableGen] Capping Quadriflow target faces from {faces_count} to 30000 for stability")
                    faces_count = 30000
                print(f"[StableGen] Running native C++ Quadriflow solver (target faces: {faces_count})")
                res = {'CANCELLED'}
                try:
                    res = bpy.ops.object.quadriflow_remesh(
                        target_faces=faces_count,
                        use_mesh_symmetry=False,
                        use_preserve_sharp=False,
                        use_preserve_boundary=False
                    )
                except Exception as q_err:
                    print(f"[StableGen] Native Quadriflow custom call failed, retrying with simple signature: {q_err}")
                    res = bpy.ops.object.quadriflow_remesh(target_faces=faces_count)
                
                if 'FINISHED' not in res:
                    raise RuntimeError(f"Quadriflow operator returned non-finished status: {res}")

            def run_solver(faces_count):
                if remesh_method == 'quadriflow':
                    run_quadriflow(faces_count)

            # --- Local Remeshing Solver Stage ---
            update_progress(50, f"Executing {remesh_method} retopology solver...")
            try:
                run_solver(target_faces)
            except Exception as solve_err:
                print(f"[StableGen] Solver failed directly on mesh: {solve_err}. Attempting high-resolution OpenVDB Voxel Repair healing...")
                
                target_bu = getattr(context.scene, 'trellis2_import_scale', 2.0)
                voxel_size = max(0.002, target_bu / 512.0)
                print(f"[StableGen] Healing mesh with OpenVDB Voxel Remesh (size: {voxel_size:.4f})")
                
                obj.data.remesh_voxel_size = voxel_size
                res_vr = {'CANCELLED'}
                try:
                    res_vr = bpy.ops.object.voxel_remesh()
                except Exception as remesh_err:
                    print(f"[StableGen] OpenVDB Voxel Remesh healing failed: {remesh_err}")
                
                # Retry solver on the healed mesh
                print("[StableGen] Retrying solver execution on healed manifold mesh...")
                try:
                    run_solver(target_faces)
                except Exception as retry_err:
                    print(f"[StableGen] Solver failed even after voxel healing: {retry_err}")
                    if target_faces > 30000:
                        stable_target = 20000
                        print(f"[StableGen] Retrying with a mathematically stable target count of {stable_target} quads...")
                        try:
                            run_solver(stable_target)
                        except Exception as final_err:
                            print(f"[StableGen] Fatal: Solver failed even at stable target: {final_err}")
                    else:
                        print(f"[StableGen] Fatal: Solver execution failed: {retry_err}")

            # 4. Detail Projection back from HighPoly_Reference via temporary SHRINKWRAP modifier
            update_progress(80, "Projecting high-detail reference back...")
            print("[StableGen] Snapping quad grid and projecting high-detail reference details back")
            sw_mod = obj.modifiers.new(name="SG_Shrinkwrap", type='SHRINKWRAP')
            sw_mod.target = high_poly
            sw_mod.wrap_method = 'NEAREST_SURFACEPOINT'
            sw_mod.wrap_mode = 'ON_SURFACE'
            
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_apply(modifier=sw_mod.name)
            except Exception as sw_err:
                print(f"[StableGen] Warning: Detail projection shrinkwrap application failed: {sw_err}")

            # 5. Clean up HighPoly_Reference duplicate mesh
            update_progress(95, "Cleaning up reference assets...")
            bpy.ops.object.select_all(action='DESELECT')
            high_poly.select_set(True)
            try:
                bpy.ops.object.delete()
                print(f"[StableGen] Retopology completed successfully for object: {obj.name}")
            except Exception as del_err:
                print(f"[StableGen] Warning: Could not delete high-poly reference: {del_err}")

            # Reselect the final low-poly mesh
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            update_progress(100, "Local retopology completed!")
            
    finally:
        wm.progress_end()
        context.workspace.status_text_set(None)


class StableGenLocalPostProcess(bpy.types.Operator):
    """Run local post-processing (decimation and/or retopology) on the active mesh"""
    bl_idname = "object.stablegen_local_postprocess"
    bl_label = "Run Local Post-Processing"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        scene = context.scene
        decimate_method = getattr(scene, 'trellis2_decimate_method', 'server')
        remesh_method = getattr(scene, 'trellis2_remesh_method', 'qdc')
        solidify = getattr(scene, 'trellis2_solidify', False)
        
        # 1. Check if all post-processing options are disabled/none
        if decimate_method == 'none' and remesh_method == 'none' and not solidify:
            self.report({'WARNING'}, "No local post-processing is enabled (Decimation, Retopology, and Solidification are all disabled/set to None).")
            return {'CANCELLED'}
            
        # 2. Error if set to server-side post-processing
        if decimate_method == 'server' or remesh_method == 'qdc':
            self.report({'ERROR'}, "Server-side processing cannot be run locally. Please select a local decimation/remesh method.")
            return {'CANCELLED'}
            
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object.")
            return {'CANCELLED'}
            
        target_faces = scene.trellis2_decimation
        do_decimate = (decimate_method == 'collapse')
        
        print(f"[StableGen] Running manual local post-processing on {obj.name}: decimate={decimate_method}, remesh={remesh_method}, solidify={solidify}, target={target_faces}")
        
        # Switch to object mode
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        # 3. Run solidification locally on the main thread first if enabled
        if solidify:
            from ..texturing.print_export import _make_solid_mesh_object
            try:
                print(f"[StableGen] Solidifying active mesh: {obj.name}...")
                solid_mesh = _make_solid_mesh_object(obj)
                old_mesh = obj.data
                obj.data = solid_mesh
                bpy.data.meshes.remove(old_mesh)
                print(f"[StableGen] Solidification completed for {obj.name}")
            except Exception as e:
                self.report({'ERROR'}, f"Failed to solidify mesh: {e}")
                return {'CANCELLED'}

            
        # If we need local decimation, run it asynchronously
        if do_decimate:
            import numpy as np
            import bmesh
            
            print(f"[StableGen] Starting async trimesh decimation for {obj.name}...")
            
            # Extract mesh data on main thread
            mesh = obj.data
            vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", vertices)
            vertices = vertices.reshape((-1, 3))
            
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            
            faces = np.empty(len(bm.faces) * 3, dtype=np.int32)
            for i, face in enumerate(bm.faces):
                faces[i*3 : (i+1)*3] = [v.index for v in face.verts]
            bm.free()
            
            faces = faces.reshape((-1, 3))
            
            # Define worker and callback
            def bg_decimate_work():
                import trimesh
                t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                orig_faces = len(t_mesh.faces)
                if orig_faces > target_faces:
                    decimated = t_mesh.simplify_quadric_decimation(face_count=target_faces)
                    return {
                        "vertices": decimated.vertices.tolist(),
                        "faces": decimated.faces.tolist(),
                        "success": True
                    }
                return {"success": False}
                
            def on_decimate_done(result):
                if not result:
                    print("[StableGen] Async decimation failed or was aborted.")
                    return
                
                # Check if object still exists
                if not obj or obj.name not in bpy.context.scene.objects:
                    print("[StableGen] Active mesh object no longer exists.")
                    return
                    
                if result.get("success"):
                    mesh = obj.data
                    mesh.clear_geometry()
                    mesh.from_pydata(result["vertices"], [], result["faces"])
                    mesh.update()
                    print(f"[StableGen] Async decimation written back successfully.")
                else:
                    print(f"[StableGen] Mesh already has fewer faces than target. Bypassed decimation.")
                
                # Post-decimation stage (retopology)
                try:
                    if remesh_method == 'quadriflow':
                        perform_local_retopology(bpy.context, [obj], remesh_method)
                    bpy.context.workspace.status_text_set(None)
                    # Trigger viewport redraw
                    for area in bpy.context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                except Exception as post_err:
                    print(f"[StableGen] Error in post-decimation stage: {post_err}")
            
            # Run async decimation
            from ..core.state import _run_async
            bpy.context.workspace.status_text_set("StableGen: Decimating mesh in background...")
            _run_async(bg_decimate_work, on_decimate_done)
            self.report({'INFO'}, "Trimesh decimation started in the background...")
            return {'FINISHED'}
            
        else:
            # No background decimation needed, run retopology synchronously on main thread
            try:
                if remesh_method == 'quadriflow':
                    perform_local_retopology(context, [obj], remesh_method)
                    
                self.report({'INFO'}, "Local post-processing completed successfully!")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f"Failed to run local post-processing: {e}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}
