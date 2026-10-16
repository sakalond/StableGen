"""Read existing projection images without loading files or changing materials."""
import json
import re
from pathlib import Path

import bpy

from ..texturing.view_plan import generation_cameras, order_cameras

# Masked Edit output folders: '<timestamp>-masked-<hex>' (see masked_edit.SG_OT_MaskedEdit).
_MASKED_RUN = re.compile(r'-masked-[0-9a-f]+$', re.IGNORECASE)


def is_masked_edit_layer(node):
    """A Masked Edit view layer reuses projection labels but is not a camera's generated image.

    Masked Edit tags its layers; older files are recognised by the run folder holding the image.
    """
    if node.get('sg_masked_edit'):
        return True
    path = Path(bpy.path.abspath(node.image.filepath, library=node.image.library))
    return any(_MASKED_RUN.search(part) for part in path.parts[:-1])


def images_from_manifest(cameras, textures, manifest):
    """Keep persisted source indices even when cameras are renamed or removed."""
    by_id = {}
    for camera in cameras:
        if camera.get('sg_view_uuid'):
            by_id.setdefault(str(camera['sg_view_uuid']), []).append(camera)
    by_name = {c.name: c for c in cameras}
    result = {}
    for fallback_index, record in enumerate(manifest.get('views', [])):
        if not isinstance(record, dict):
            continue
        identity = str(record.get('view_id') or '')
        matches = by_id.get(identity, [])
        camera = (matches[0] if len(matches) == 1 else
                  next((c for c in matches if c.name == record.get('name')), None)) if identity else by_name.get(record.get('name'))
        index = record.get('index', fallback_index)
        if isinstance(index, str) and index.isdigit():
            index = int(index)
        if not isinstance(index, int):
            continue
        if camera is not None and index in textures:
            result[camera.name] = textures[index]
    return result


def projected_images(context, *, objects=None, materials=None, manifests=None):
    """Resolve actual texture nodes and saved camera identities, with no recovery.

    Optional object/material inputs let viewport overlays use the original
    projection while reversible diagnostic materials are being displayed.
    Manifest reads can be cached outside the draw callback.
    """
    scene = context.scene
    if objects is None:
        objects = context.selected_objects if scene.texture_objects == 'selected' else context.view_layer.objects
    if materials is None:
        materials = {slot.material for obj in objects if obj.type == 'MESH' and not obj.hide_get()
                     for slot in obj.material_slots if slot.material}
    textures, latest = {}, -1
    for material in materials:
        if not material.use_nodes or not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            if node.type != 'TEX_IMAGE' or node.image is None:
                continue
            parts = node.label.split('-')
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                continue
            if is_masked_edit_layer(node):
                continue
            index, version = map(int, parts)
            if version > latest:
                latest, textures = version, {}
            if version == latest:
                textures.setdefault(index, node.image)
    if not textures:
        return {}
    custom = [item.name for item in scene.sg_camera_order] if scene.sg_use_custom_camera_order else ()
    cameras = order_cameras(generation_cameras(scene), custom)
    attempted = set()
    for image in textures.values():
        folder = Path(bpy.path.abspath(image.filepath, library=image.library)).parent
        if folder.name.lower() != 'generated' or folder in attempted:
            continue
        attempted.add(folder)
        path = folder.parent / 'view_plan.json'
        try:
            if manifests is not None and path in manifests:
                manifest = manifests[path]
            else:
                manifest = json.loads(path.read_text(encoding='utf8'))
                if manifests is not None:
                    manifests[path] = manifest
            if isinstance(manifest, dict) and isinstance(manifest.get('views'), list) and manifest['views']:
                return images_from_manifest(cameras, textures, manifest)
        except (OSError, ValueError, TypeError):
            continue
    # Older generations without a view plan retain the existing order fallback.
    return {camera.name: textures[i] for i, camera in enumerate(cameras) if i in textures}
