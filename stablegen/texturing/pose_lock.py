"""Keep an animated character in its visible pose during texturing.

Blender reevaluates Actions whenever ``Scene.frame_set`` is called.  Several
texturing render paths intentionally change frames, so a pose edited by the
user (including StableGen's Rest/T/A pose buttons) would otherwise be replaced
by the Action pose.  This module snapshots the visible pose, temporarily
disconnects its animation inputs, and restores them after generation.
"""

import bpy  # pylint: disable=import-error


_ACTIVE_TEXTURING_FRAME = "_sg_active_texturing_frame"


def _has_animation(armature):
    animation_data = armature.animation_data
    if not animation_data:
        return False
    if animation_data.action is not None:
        return True
    if any(track.strips for track in animation_data.nla_tracks):
        return True
    return bool(animation_data.drivers)


def _target_armatures(meshes):
    armatures = []
    seen = set()
    for mesh in meshes:
        if not mesh or mesh.type != 'MESH':
            continue
        candidates = []
        try:
            found = mesh.find_armature()
            if found is not None:
                candidates.append(found)
        except (AttributeError, RuntimeError):
            pass
        candidates.extend(
            modifier.object
            for modifier in mesh.modifiers
            if modifier.type == 'ARMATURE' and modifier.object is not None
        )
        if mesh.parent and mesh.parent.type == 'ARMATURE':
            candidates.append(mesh.parent)
        for armature in candidates:
            pointer = armature.as_pointer()
            if pointer not in seen and _has_animation(armature):
                seen.add(pointer)
                armatures.append(armature)
    return armatures


def set_active_texturing_frame(scene, frame):
    """Publish the run frame to render helpers which do not own the operator."""
    previous = (
        _ACTIVE_TEXTURING_FRAME in scene,
        scene.get(_ACTIVE_TEXTURING_FRAME),
    )
    scene[_ACTIVE_TEXTURING_FRAME] = int(frame)
    return previous


def restore_active_texturing_frame(scene, previous):
    """Restore a pre-existing runtime frame marker, if there was one."""
    existed, value = previous
    if existed:
        scene[_ACTIVE_TEXTURING_FRAME] = value
    elif _ACTIVE_TEXTURING_FRAME in scene:
        del scene[_ACTIVE_TEXTURING_FRAME]


def ensure_texturing_frame(
        scene, legacy_frame=1, *, honor_consistent=True,
        force_configured=False):
    """Select the run's captured frame, or retain the configured legacy rule."""
    active_frame = scene.get(_ACTIVE_TEXTURING_FRAME)
    if active_frame is not None:
        target = int(active_frame)
    elif (force_configured
          or (honor_consistent
              and getattr(scene, 'sg_rework_consistent_frame', False))):
        target = int(getattr(scene, 'sg_texturing_frame', legacy_frame))
    else:
        target = int(legacy_frame)
    if scene.frame_current != target:
        scene.frame_set(target)
    return target


class TexturingPoseLock:
    """Freeze animated armatures at the evaluated pose visible at click time."""

    def __init__(self, context, meshes):
        self._context = context
        self._scene = context.scene
        self._frame = int(context.scene.frame_current)
        self._armatures = []
        self._frozen = False
        self._restored = False

        for armature in _target_armatures(meshes):
            animation_data = armature.animation_data
            self._armatures.append({
                "object": armature,
                "matrix_world": armature.matrix_world.copy(),
                "pose_position": armature.data.pose_position,
                "action": animation_data.action,
                "use_nla": getattr(animation_data, "use_nla", None),
                "nla_mutes": [
                    (track, bool(track.mute))
                    for track in animation_data.nla_tracks
                ],
                "driver_mutes": [
                    (driver, bool(driver.mute))
                    for driver in animation_data.drivers
                ],
                "object_constraint_mutes": [
                    (constraint, bool(constraint.mute))
                    for constraint in armature.constraints
                ],
                "bone_constraint_mutes": [
                    (constraint, bool(constraint.mute))
                    for pose_bone in armature.pose.bones
                    for constraint in pose_bone.constraints
                ],
                "bone_matrices": {
                    pose_bone.name: pose_bone.matrix.copy()
                    for pose_bone in armature.pose.bones
                },
                "bone_bases": {
                    pose_bone.name: pose_bone.matrix_basis.copy()
                    for pose_bone in armature.pose.bones
                },
            })

    @property
    def active(self):
        return bool(self._armatures)

    @property
    def frame(self):
        return self._frame

    def _apply_captured_pose(self):
        for state in self._armatures:
            armature = state["object"]
            if armature.name not in bpy.data.objects:
                continue
            armature.data.pose_position = 'POSE'
            armature.matrix_world = state["matrix_world"]
            pose_bones = sorted(
                armature.pose.bones,
                key=lambda pose_bone: len(pose_bone.parent_recursive),
            )
            for pose_bone in pose_bones:
                matrix = state["bone_matrices"].get(pose_bone.name)
                if matrix is not None:
                    pose_bone.matrix = matrix
        self._context.view_layer.update()

    def freeze(self):
        if not self.active or self._frozen:
            return self.active
        for state in self._armatures:
            armature = state["object"]
            animation_data = armature.animation_data
            animation_data.action = None
            if state["use_nla"] is not None:
                animation_data.use_nla = False
            for track, _mute in state["nla_mutes"]:
                track.mute = True
            for driver, _mute in state["driver_mutes"]:
                driver.mute = True
            for constraint, _mute in (
                    state["object_constraint_mutes"]
                    + state["bone_constraint_mutes"]):
                constraint.mute = True
        self._apply_captured_pose()
        self._frozen = True
        return True

    def restore(self):
        if self._restored or not self.active:
            return
        # Restore every animation input first. Assigning an Action tags the rig
        # for dependency-graph reevaluation, so applying the manual pose before
        # this point would let the next viewport redraw overwrite it.
        for state in self._armatures:
            armature = state["object"]
            if armature.name not in bpy.data.objects:
                continue
            animation_data = armature.animation_data
            animation_data.action = state["action"]
            if state["use_nla"] is not None:
                animation_data.use_nla = state["use_nla"]
            for track, mute in state["nla_mutes"]:
                track.mute = mute
            for driver, mute in state["driver_mutes"]:
                driver.mute = mute
            for constraint, mute in (
                    state["object_constraint_mutes"]
                    + state["bone_constraint_mutes"]):
                constraint.mute = mute
            armature.data.pose_position = state["pose_position"]

        # Deliberately evaluate the restored animation at the click-time frame,
        # then reproduce the exact unkeyed/manual state as the final mutation.
        # This is equivalent to the user pressing StableGen's Rest/T/A button
        # after selecting that frame: animation remains attached and resumes on
        # the next explicit timeline change.
        self._scene.frame_set(self._frame)
        self._context.view_layer.update()
        for state in self._armatures:
            armature = state["object"]
            if armature.name not in bpy.data.objects:
                continue
            armature.matrix_world = state["matrix_world"]
            for pose_bone in armature.pose.bones:
                basis = state["bone_bases"].get(pose_bone.name)
                if basis is not None:
                    pose_bone.matrix_basis = basis
        self._restored = True
