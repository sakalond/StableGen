"""Read existing projection images without loading files or changing materials."""
import json
import os
import re
from pathlib import Path

import bpy

from ..texturing.view_plan import generation_cameras, order_cameras

# Masked Edit output folders: '<timestamp>-masked-<hex>' (see masked_edit.SG_OT_MaskedEdit).
_MASKED_RUN = re.compile(r'-masked-[0-9a-f]+$', re.IGNORECASE)
# A run's generated views: generated_image<view>-<material>-0001.png (utils.get_file_path)
_GENERATED_FILE = re.compile(r'generated_image(\d+)-(\d+)(?:-\d+)?\.png$', re.IGNORECASE)
_run_folders = {}       # (timestamp, .blend, image count, output folder) -> the run folder, or None
_run_views = {}         # run folder -> (view_plan.json and generated/ mtimes, {view id or name: file})


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


def _run_folder(context):
    """The folder of the scene's run (named after its output timestamp) holding its view plan: where StableGen
    writes it, or where an image the file uses (a bake, a projected view) came from after the .blend was renamed."""
    scene = context.scene
    stamp = scene.output_timestamp
    if not stamp:
        return None
    from ..core import get_addon_prefs
    prefs = get_addon_prefs(context)
    output = bpy.path.abspath(prefs.output_dir) if prefs is not None and prefs.output_dir else ''
    key = (stamp, bpy.data.filepath, len(bpy.data.images), output)
    if key not in _run_folders:
        candidates = []
        if output:
            stem = Path(bpy.data.filepath).stem or scene.name
            candidates.append(Path(output) / stem / stamp)
        for image in bpy.data.images:
            if image.source != 'FILE' or not image.filepath:
                continue
            path = Path(bpy.path.abspath(image.filepath, library=image.library))
            candidates.extend(parent for parent in path.parents if parent.name == stamp)
        folder = None
        for candidate in dict.fromkeys(candidates):
            try:
                if (candidate / 'view_plan.json').is_file():
                    folder = candidate
                    break
            except OSError:
                continue
        _run_folders[key] = folder
    return _run_folders[key]


def generated_files(context):
    """Each scene camera's generated image file in the scene's own run ({camera name: path}). The cards show these
    when the camera's view is not in a visible material, as after Bake Textures, whose materials hold the bake."""
    folder = _run_folder(context)
    if folder is None:
        return {}
    plan, generated = folder / 'view_plan.json', folder / 'generated'
    try:
        stamp = (plan.stat().st_mtime, generated.stat().st_mtime)
    except OSError:
        return {}
    cached = _run_views.get(folder)
    if cached is None or cached[0] != stamp:
        files = {}      # view index -> the file of its lowest material id
        try:
            names = [entry.name for entry in os.scandir(generated) if entry.is_file()]
            manifest = json.loads(plan.read_text(encoding='utf8'))
        except (OSError, ValueError):
            return {}
        for name in names:
            match = _GENERATED_FILE.fullmatch(name)
            if match:
                index, material = int(match.group(1)), int(match.group(2))
                if index not in files or material < files[index][0]:
                    files[index] = (material, str(generated / name))
        views = {}
        for record in manifest.get('views', []) if isinstance(manifest, dict) else ():
            index = record.get('index') if isinstance(record, dict) else None
            index = int(index) if isinstance(index, str) and index.isdigit() else index
            if isinstance(index, int) and index in files:
                for key in (record.get('view_id'), record.get('name')):
                    if key:
                        views.setdefault(str(key), files[index][1])
        cached = _run_views[folder] = (stamp, views)
    views = cached[1]
    result = {}
    for camera in generation_cameras(context.scene):
        path = views.get(str(camera.get('sg_view_uuid') or '')) or views.get(camera.name)
        if path:
            result[camera.name] = path
    return result
