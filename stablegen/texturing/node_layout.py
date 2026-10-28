"""Presentation-only layout for StableGen projection shader graphs.

This module never creates or removes shader nodes or links. Frames, node labels,
widths and positions are the only things it changes. Projection keeps its own
construction order, including incremental sequential and appended edit passes.
"""

from collections import defaultdict
from .jobs import operation


_VIEW_STAGES = (
    'Surface geometry',
    'Visibility and angle',
    'Camera basis and frustum',
    'Raycast visibility',
    'Facing and angle',
    'Frame edge and vignette',
    'Acquisition quality',
    'Image and UV',
    'Sequential write mask',
    'Generated alpha',
    'Depth and part guard',
    'Guard feather',
    'Occlusion fade',
    'Local edit edge mask',
    'Source confidence',
)

_BLEND_STAGES = (
    'Surface blend',
    'Guard coverage rescue',
    'Occlusion handoff',
    'Normalize and sharpen',
    'Boundary crossfade',
    'Local unsharpen',
    'Guard feathering',
    'Local edit ramps',
    'Early view priority',
    'Combine view colors',
    'Source confidence output',
)


def _frame_color(title):
    """Keep view, guard, blend, and output regions distinct at a glance."""
    if title.startswith('Projection pass'):
        return (0.10, 0.12, 0.17)
    if title.startswith('View '):
        return (0.10, 0.17, 0.19)
    if title == 'Graph controls':
        return (0.11, 0.23, 0.17)
    if title == 'Graph control wiring':
        return (0.12, 0.17, 0.14)
    if title.startswith('Handoff level') or title == 'Boundary crossfade':
        return (0.20, 0.14, 0.22)
    if 'guard' in title.lower() or 'feather' in title.lower() or 'occlusion' in title.lower():
        return (0.19, 0.16, 0.11)
    if title == 'Material output':
        return (0.12, 0.20, 0.14)
    if title in {'Combine view colors', 'Normalize and sharpen', 'Surface blend'}:
        return (0.12, 0.15, 0.22)
    return (0.13, 0.16, 0.18)


def _view_stage(node):
    if node.get('sg_layout_stage') in _VIEW_STAGES:
        return node['sg_layout_stage']
    if node.type == 'NEW_GEOMETRY':
        return 'Surface geometry'
    label = node.label
    if (node.type == 'TEX_IMAGE' and label.split('-')[0].isdigit()
            or node.type == 'ATTRIBUTE' and node.attribute_name.startswith('ProjectionUV_')):
        return 'Image and UV'
    if label.startswith(('Confidence',)):
        return 'Source confidence'
    if label.startswith(('EdgeFeather-', 'EF_Weight-')):
        return 'Local edit edge mask'
    if label.startswith(('ProjectionOcclusion', 'Occlusion', 'SGOcclusion')):
        return 'Occlusion fade'
    if label.startswith(('ProjectionGuardFeather-', 'DepthPartFeathered-',
                         'ProjectionGuardInfluence-')):
        return 'Guard feather'
    if label.startswith(('ProjectionGuard-', 'ProjectionGuardWeight-',
                         'ProjectionGuardFloor-', 'DepthPartGuarded-')):
        return 'Depth and part guard'
    if label.startswith(('Qwen21GeneratedAlpha-', 'SDXLFeatherAlpha-')):
        return 'Generated alpha'
    if label.startswith(('SequentialWrite', 'QwenWriteMasked-')):
        return 'Sequential write mask'
    if label.startswith(('AcquisitionQuality-', 'ProjectedDensity-')):
        return 'Acquisition quality'
    if label.startswith(('FrameEdgeQuality-', 'Feather-')):
        return 'Frame edge and vignette'
    return 'Visibility and angle'


def _blend_stage(node, feather_method):
    if node.get('sg_layout_stage') in _BLEND_STAGES:
        return node['sg_layout_stage']
    label = node.label
    if label.startswith('SurfaceBlend'):
        return 'Surface blend'
    if label.startswith(('GuardRescue', 'GuardCoverage')):
        return 'Guard coverage rescue'
    if label.startswith(('OcclusionHandoff', 'GuardOcclusion')):
        return 'Occlusion handoff'
    if label.startswith('SharpW-Feather-'):
        return 'Boundary crossfade'
    if label.startswith(('WeightMax', 'SafeMax', 'NormW-', 'SharpW-')):
        return 'Normalize and sharpen'
    if label.startswith('Crossfade'):
        return 'Boundary crossfade'
    if label.startswith('PostExponentFeather-'):
        return {'BOUNDARY_CROSSFADE': 'Boundary crossfade',
                'LOCAL_UNSHARPEN': 'Local unsharpen'}.get(
                    feather_method, 'Guard feathering')
    if label.startswith(('FeatherPreference-', 'FeatherVisible-', 'FeatherActive-',
                         'FeatherInfluence', 'FeatherSharp', 'FeatherSoft',
                         'ProjectionFeatherFloor-')):
        return 'Local unsharpen'
    if node.type == 'VALTORGB':
        return 'Local edit ramps'
    if node.type == 'MAP_RANGE':
        return 'Early view priority'
    if label.startswith('SGBestSourceQuality'):
        return 'Source confidence output'
    return 'Combine view colors'


def _label_unlabeled(node, stage):
    if node.label:
        return
    if node.type == 'MATH':
        operation = node.operation.replace('_', ' ').title()
        if stage == 'Combine view colors':
            node.label = {
                'ADD': 'Sum view weights',
                'DIVIDE': 'Mix factor from weights',
                'MULTIPLY': 'Combine visibility factors',
            }.get(node.operation, operation)
        else:
            node.label = operation
    elif node.type == 'VECT_MATH':
        node.label = node.operation.replace('_', ' ').title()
    elif node.type == 'MIX_RGB':
        node.label = 'Mix projected images'
    elif node.type == 'VALTORGB':
        node.label = 'Local edit visibility ramp'
    elif node.type == 'TEX_IMAGE':
        node.label = 'Feature mask'
    elif node.type == 'ATTRIBUTE':
        node.label = node.attribute_name or 'Mesh attribute'
    elif node.type == 'OUTPUT_MATERIAL':
        node.label = 'Projected material output'
    elif node.type == 'NEW_GEOMETRY':
        node.label = 'Surface position and normal'
    elif node.type == 'MAP_RANGE':
        node.label = 'Prioritize earlier view'
    else:
        node.label = node.bl_label


def _create_frame(nodes, title, parent, x, y, width, height):
    frame = nodes.new('NodeFrame')
    frame.name = 'SG Layout | ' + title
    frame.label = title
    frame.label_size = 26 if parent is None else 21
    frame.use_custom_color = True
    frame.color = _frame_color(title)
    frame.shrink = False
    frame.width = width
    frame.height = height
    if parent is not None:
        frame.parent = parent
    frame.location = (x, y)
    return frame


def _estimated_height(node):
    if node.hide:
        return 60
    if node.type == 'TEX_IMAGE':
        return 300
    if node.type in {'VALTORGB', 'SCRIPT'}:
        return 220
    return max(125, min(240, 65 + 25 * len(node.inputs)))


def _arrange_group(nodes, links, items, title, parent, x, y):
    """Place a feature's nodes in dependency columns inside a labeled frame."""
    members = {node.as_pointer(): node for node in items}
    parents = defaultdict(list)
    children = defaultdict(list)
    for source in members:
        for target in links.get(source, ()):
            if target in members and source != target:
                parents[target].append(source)
                children[source].append(target)

    depth = {pointer: 0 for pointer in members}
    indegree = {pointer: len(set(parents[pointer])) for pointer in members}
    ready = [pointer for pointer, count in indegree.items() if count == 0]
    while ready:
        pointer = ready.pop()
        for target in set(children[pointer]):
            depth[target] = max(depth[target], depth[pointer] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    columns = defaultdict(list)
    for node in items:
        columns[depth[node.as_pointer()]].append(node)
    column_heights = {
        index: sum(_estimated_height(node) + 35 for node in column)
        for index, column in columns.items()
    }
    width = max(380, 60 + (max(columns) + 1) * 250)
    height = max(270, 105 + max(column_heights.values()))
    frame = _create_frame(nodes, title, parent, x, y, width, height)
    for index, column in columns.items():
        top = -65
        for node in column:
            stage = title.split(' | ', 1)[-1]
            _label_unlabeled(node, stage)
            node.parent = frame
            node.width = (240 if node.type == 'VALUE' else
                          205 if node.type in {'TEX_IMAGE', 'SCRIPT', 'VALTORGB'} else 180)
            node.location = (35 + 250 * index, top)
            top -= _estimated_height(node) + 35
    return width, height


@operation
def _arrange_crossfade(nodes, links, items, parent, x, y):
    """Separate the six threshold levels of the exact crossfade graph."""
    levels = defaultdict(list)
    final = []
    for node in items:
        label = node.label
        if label.startswith('PostExponentFeather-'):
            final.append(node)
        else:
            parts = label.split('-')
            index = 2 if label.startswith('SharpW-Feather-') else 1
            if len(parts) > index and parts[index].isdigit():
                levels[int(parts[index])].append(node)
            else:
                final.append(node)
    sections = [(f'Handoff level {index + 1:02d}', levels[index])
                for index in sorted(levels)]
    if final:
        sections.append(('Final view weights', final))
    frame = _create_frame(nodes, 'Boundary crossfade', parent, x, y, 500, 400)
    top, width = -70, 0
    for offset in range(0, len(sections), 2):
        left_title, left_items = sections[offset]
        left_width, left_height = _arrange_group(
            nodes, links, left_items, left_title, frame, 35, top)
        yield None
        row_width, row_height = left_width, left_height
        if offset + 1 < len(sections):
            right_title, right_items = sections[offset + 1]
            right_width, right_height = _arrange_group(
                nodes, links, right_items, right_title,
                frame, 35 + left_width + 55, top)
            yield None
            row_width += 55 + right_width
            row_height = max(row_height, right_height)
        width = max(width, row_width)
        top -= row_height + 60
    frame.width = width + 70
    frame.height = -top + 30
    return frame.width, frame.height


@operation
def organize_projection_material(material, new_nodes, camera_nodes, cameras, stop_index,
                                 local_edit=False, feather_method='ORIGINAL'):
    """Arrange only nodes made by this build, preserving earlier edit passes."""
    items = [node for node in new_nodes if node.type != 'FRAME']
    if not items:
        return
    tree = material.node_tree
    nodes = tree.nodes
    # Index once for the whole layout pass. Feature frames only need edges
    # leaving their own members, rather than rescanning the complete graph.
    links = defaultdict(list)
    for link in tree.links:
        links[link.from_node.as_pointer()].append(link.to_node.as_pointer())
    view_items = defaultdict(list)
    global_items = []
    for node in items:
        view = camera_nodes.get(node.as_pointer())
        if view is None:
            global_items.append(node)
        else:
            view_items[view].append(node)

    old_frames = [node for node in nodes if node.type == 'FRAME'
                  and node.name.startswith('SG Layout | Projection pass')]
    if old_frames:
        origin_x = max(frame.location_absolute.x + frame.width for frame in old_frames) + 320
    else:
        origin_x = 0
    pass_number = len(old_frames) + 1
    pass_title = (f'Projection pass {pass_number:02d} | Local edit'
                  if local_edit else f'Projection pass {pass_number:02d} | Base projection')
    pass_frame = _create_frame(nodes, pass_title, None, origin_x, 0, 600, 400)

    controls = [node for node in global_items
                if node.get('sg_layout_stage') == 'Graph controls']
    control_wiring = [node for node in global_items
                      if node.get('sg_layout_stage') == 'Graph control wiring']
    shared = [node for node in global_items if node.label == 'SGMaterialIndex']
    output = [node for node in global_items if node.type == 'OUTPUT_MATERIAL']
    blend_items = [node for node in global_items
                   if node not in controls and node not in control_wiring
                   and node not in shared and node not in output]

    controls_width = controls_height = 0
    if controls:
        controls_width, controls_height = _arrange_group(
            nodes, links, controls, 'Graph controls', pass_frame, 35, -70)
    wiring_width = wiring_height = 0
    if control_wiring:
        wiring_width, wiring_height = _arrange_group(
            nodes, links, control_wiring, 'Graph control wiring', pass_frame,
            35 + controls_width + (55 if controls else 0), -70)
    shared_width = shared_height = 0
    if shared:
        shared_width, shared_height = _arrange_group(
            nodes, links, shared, 'Material index', pass_frame,
            35 + controls_width + wiring_width
            + (55 if controls else 0)
            + (55 if control_wiring else 0), -70)
    view_origin_x = 35 + controls_width + (55 if controls else 0)
    if control_wiring:
        view_origin_x += wiring_width + 55
    if shared:
        view_origin_x += shared_width + 70
    view_right = view_origin_x
    view_top = -70
    for index, camera in enumerate(cameras):
        current = view_items.get(index)
        if not current:
            continue
        stages = defaultdict(list)
        for node in current:
            stages[_view_stage(node)].append(node)
        camera_title = f'View {index + 1:02d} | {camera.name}'
        if index > stop_index:
            camera_title += ' | pending'
        view_frame = _create_frame(nodes, camera_title, pass_frame,
                                   view_origin_x, view_top, 400, 300)
        section_x, section_height = 35, 0
        for stage in _VIEW_STAGES:
            members = stages.get(stage)
            if not members:
                continue
            width, height = _arrange_group(nodes, links, members, stage,
                                            view_frame, section_x, -70)
            yield None
            section_x += width + 45
            section_height = max(section_height, height)
        view_frame.width = section_x + 10
        view_frame.height = section_height + 155
        view_right = max(view_right, view_origin_x + view_frame.width)
        view_top -= view_frame.height + 90

    stage_items = defaultdict(list)
    for node in blend_items:
        stage_items[_blend_stage(node, feather_method)].append(node)
    blend_x = view_right + 120
    blend_height = 0
    for stage in _BLEND_STAGES:
        members = stage_items.get(stage)
        if not members:
            continue
        if stage == 'Boundary crossfade':
            width, height = yield from _arrange_crossfade(
                nodes, links, members, pass_frame, blend_x, -70, _steps=True)
        else:
            width, height = _arrange_group(
                nodes, links, members, stage, pass_frame, blend_x, -70)
        blend_x += width + 60
        blend_height = max(blend_height, height)
        yield None
    if output:
        width, height = _arrange_group(nodes, links, output, 'Material output',
                                        pass_frame, blend_x, -70)
        blend_x += width + 40
        blend_height = max(blend_height, height)
    pass_frame.width = max(blend_x + 35, view_right + 60)
    pass_frame.height = max(-view_top + 30, blend_height + 155,
                            controls_height + 155, wiring_height + 155,
                            shared_height + 155)
