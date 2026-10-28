"""Generation-scoped shader reuse; never trust a saved or edited topology."""

import os
import hashlib
from array import array
import bpy
from ..core.architectures import EDIT_ARCHITECTURES, QWEN_ARCHITECTURES


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bpy.types.ID):
        return value.as_pointer()
    try:
        return tuple(value)
    except TypeError:
        return str(value)


def graph_state(material):
    """Include functional edits, socket values and links, but not UI layout."""
    return _tree_state(material.node_tree, set())


def _tree_state(tree, visited):
    if tree.as_pointer() in visited:
        return tree.as_pointer()
    visited.add(tree.as_pointer())
    nodes = []
    for node in tree.nodes:
        if node.type == 'FRAME':
            continue
        properties = []
        for prop in node.bl_rna.properties:
            name = prop.identifier
            if (prop.type in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'}
                    and name not in {'name', 'label', 'select', 'hide', 'show_options',
                                     'show_preview', 'show_texture', 'width', 'height',
                                     'location', 'location_absolute', 'dimensions',
                                     'color', 'use_custom_color',
                                     'bl_label', 'bl_description', 'bl_icon'}):
                properties.append((name, _plain(getattr(node, name))))
        defaults = tuple(_plain(s.default_value) if hasattr(s, 'default_value') else None
                         for s in node.inputs)
        outputs = (tuple(_plain(s.default_value) for s in node.outputs)
                   if node.type in {'VALUE', 'RGB'} else ())
        image = getattr(node, 'image', None)
        image_state = ((image.as_pointer(), image.filepath,
                        image.colorspace_settings.name, image.alpha_mode,
                        tuple(image.size)) if image else None)
        group = getattr(node, 'node_tree', None)
        group_state = _tree_state(group, visited) if group else None
        ramp = getattr(node, 'color_ramp', None)
        ramp_state = ((ramp.interpolation, ramp.color_mode, ramp.hue_interpolation,
                       tuple((e.position, tuple(e.color)) for e in ramp.elements))
                      if ramp else None)
        nodes.append((node.as_pointer(), node.name, node.label, tuple(properties), defaults,
                      outputs, image_state, group_state, ramp_state))
    edges = tuple((l.from_node.as_pointer(), l.from_socket.identifier,
                   l.to_node.as_pointer(), l.to_socket.identifier) for l in tree.links)
    return tuple(nodes), edges


def eligible(scene):
    # Edit passes, baked fields and joint surface solves change topology. Base
    # sequential projection is shared by all texturing architectures; late
    # write-mask/image resources are bound before activating their view.
    qwen = scene.model_architecture in QWEN_ARCHITECTURES
    confidence = (qwen and scene.qwen_use_confidence_context
                  and scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'})
    voronoi = (scene.model_architecture in EDIT_ARCHITECTURES
               and scene.qwen_voronoi_mode)
    surface = (scene.sg_guard_feather_method == 'SURFACE_BLEND'
               and scene.sg_rework_projection_guard and scene.sg_silhouette_feather > 0)
    return (bpy.app.version >= (5, 1, 0)
            and scene.generation_method == 'sequential'
            and scene.generation_mode == 'standard'
            and (not qwen or scene.qwen_generation_method == 'generate')
            and not scene.bake_visibility_weights
            and not confidence and not surface and not voronoi)


def asset_state(paths):
    """Track missing files too, so later arrival cannot silently change a view."""
    return tuple((path, os.stat(path).st_mtime_ns, os.stat(path).st_size)
                 if os.path.isfile(path) else (path, None, None) for path in paths)


def configuration(scene, cameras, mat_id, assets, meshes):
    # Read the StableGen property groups that can affect projection. Progress,
    # the active camera is runtime state. Include each effective projection
    # resolution, including the scene fallback for cameras without overrides.
    from ..cameras.geometry import _get_camera_resolution
    from ..core.feature_flags import experimental_enabled
    prefixes = ('sg_guard_', 'sg_rework_', 'sg_silhouette_', 'sg_projection_',
                'discard_', 'weight_', 'early_', 'visibility_', 'refine_',
                'qwen_', 'bake_visibility', 'sg_allow_backface', 'sequential_', 'use_ipadapter',
                'fallback_color', 'view_blend_')
    modes = {'model_architecture', 'generation_method', 'generation_mode'}
    settings = tuple((p.identifier, _plain(getattr(scene, p.identifier)))
                     for p in scene.bl_rna.properties
                     if (p.identifier in modes or p.identifier.startswith(prefixes))
                     and p.type in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'})
    views = tuple((c.as_pointer(), _get_camera_resolution(c, scene),
                   tuple(v for row in c.matrix_world for v in row),
                   tuple((p.identifier, _plain(getattr(c.data, p.identifier)))
                         for p in c.data.bl_rna.properties
                         if p.type in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'}),
                   tuple(sorted((k, _plain(v)) for k, v in c.items()))) for c in cameras)
    files = asset_state(assets)
    geometry = []
    for obj in meshes:
        coordinates = array('f', [0.0]) * (len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get('co', coordinates)
        topology = array('i', [0]) * len(obj.data.loops)
        obj.data.loops.foreach_get('vertex_index', topology)
        modifiers = tuple((m.type, tuple(
            (p.identifier, _plain(getattr(m, p.identifier)))
            for p in m.bl_rna.properties
            if p.type in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING', 'POINTER'}
            and p.identifier != 'rna_type')) for m in obj.modifiers)
        geometry.append((obj.data.as_pointer(), tuple(v for row in obj.matrix_world for v in row),
                         hashlib.blake2b(coordinates.tobytes() + topology.tobytes()).digest(),
                         modifiers))
    # Saved flags alone do not determine behavior: preferences gate experiments.
    write_masks = experimental_enabled(scene, 'qwen_use_projection_write_masks')
    return mat_id, settings, write_masks, views, files, tuple(geometry)
