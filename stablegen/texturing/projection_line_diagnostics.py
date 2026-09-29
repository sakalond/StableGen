"""Replayable, shader-derived contribution probes for StableGen projections.

This module deliberately has no imports from the add-on: a copy of this file
can be run with ``blender --background snapshot.blend --python replay.py --
manifest.json`` even when the source checkout is no longer available.
"""

import csv
import html
import json
import math
import os
import sys
from array import array
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def _projection_image_nodes(material):
    if not material or not material.use_nodes:
        return {}
    result = {}
    for node in material.node_tree.nodes:
        if node.type != 'TEX_IMAGE' or node.image is None:
            continue
        pieces = node.label.split('-')
        if len(pieces) == 2 and pieces[0].isdigit() and pieces[1].isdigit():
            result[int(pieces[0])] = node
    return result


def _constant(tree, value):
    node = tree.nodes.new('ShaderNodeValue')
    node.label = 'Line diagnostic DPPG bypass'
    node.outputs[0].default_value = value
    return node.outputs[0]


def bypass_dppg(material):
    """Bypass just the guard fields in an already-built projection graph."""
    tree = material.node_tree
    one, zero = _constant(tree, 1.0), _constant(tree, 0.0)
    counts = {'hard_guard': 0, 'feather': 0, 'occlusion': 0, 'influence': 0}
    for node in list(tree.nodes):
        if node.label.startswith('ProjectionGuardWeight-'):
            replacement, key = one, 'hard_guard'
        elif node.label.startswith('ProjectionGuardFeather-'):
            replacement, key = one, 'feather'
        elif node.name.startswith('SGOcclusionFeather-'):
            replacement, key = one, 'occlusion'
        elif (node.name.startswith('SGOcclusionInfluence-')
              or node.name.startswith('SGFeatherInfluence-')):
            replacement, key = zero, 'influence'
        else:
            continue
        for link in list(node.outputs[0].links):
            tree.links.new(replacement, link.to_socket)
            counts[key] += 1
    return counts


def disable_coverage_rescue(material):
    """Keep DPPG guards/feathers, but make rescue retain them at full strength."""
    tree = material.node_tree
    one = _constant(tree, 1.0)
    count = 0
    for node in list(tree.nodes):
        if node.label != 'GuardRescueStrength':
            continue
        for link in list(node.outputs[0].links):
            tree.links.new(one, link.to_socket)
            count += 1
    return count


def _prepare_materials(scene, mode):
    """Keep geometry and the complete mix tree; substitute one-hot colors."""
    if mode not in ('current', 'guard_only', 'ordinary'):
        raise ValueError(mode)
    probes = []
    modification_counts = {'dppg_bypass_links': {}, 'rescue_bypass_links': 0}
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        for slot in obj.material_slots:
            original = slot.material
            if original is None:
                continue
            clone = original.copy()
            clone.name = 'SG line probe ' + original.name
            slot.material = clone
            sources = _projection_image_nodes(clone)
            tree = clone.node_tree if clone.use_nodes else None
            if tree is None:
                clone.use_nodes = True
                tree = clone.node_tree
            if clone.get('sg_surface_blend') or any(
                    n.label.startswith('SurfaceBlendWeight-') for n in tree.nodes):
                raise RuntimeError('Surface Blend stores DPPG in baked fields; this graph-bypass comparison does not apply')
            if mode == 'ordinary':
                counts = bypass_dppg(clone)
                for key, value in counts.items():
                    bypass = modification_counts['dppg_bypass_links']
                    bypass[key] = bypass.get(key, 0) + value
            elif mode == 'guard_only':
                modification_counts['rescue_bypass_links'] += disable_coverage_rescue(clone)
            # The base-case fallback is a constant Color2 on the final MixRGB.
            # Make it black so a missing view contributes zero to every probe.
            for node in tree.nodes:
                if (node.type == 'MIX_RGB' and node.inputs['Color1'].is_linked
                        and node.inputs['Fac'].is_linked
                        and not node.inputs['Color2'].is_linked):
                    node.inputs['Color2'].default_value = (0, 0, 0, 1)
            output = next((n for n in tree.nodes
                           if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
            if output is None:
                output = tree.nodes.new('ShaderNodeOutputMaterial')
            source = (output.inputs['Surface'].links[0].from_socket
                      if output.inputs['Surface'].links else None)
            emit = tree.nodes.new('ShaderNodeEmission')
            emit.label = 'Line diagnostic: final projection color'
            emit.inputs['Strength'].default_value = 1.0
            if source is not None and sources:
                tree.links.new(source, emit.inputs['Color'])
            else:
                emit.inputs['Color'].default_value = (0, 0, 0, 1)
            tree.links.new(emit.outputs[0], output.inputs['Surface'])
            for index, node in sources.items():
                rgb = tree.nodes.new('ShaderNodeRGB')
                rgb.label = f'Line diagnostic: view {index}'
                rgb.outputs[0].default_value = (0, 0, 0, 1)
                for link in list(node.outputs['Color'].links):
                    tree.links.new(rgb.outputs[0], link.to_socket)
                probes.append((index, rgb))
    if not probes:
        raise RuntimeError('No StableGen projection image nodes found in visible mesh materials')
    return probes, modification_counts


def _render_setup(scene, manifest, directory):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 4
    scene.cycles.seed = 0
    if hasattr(scene.cycles, 'use_animated_seed'):
        scene.cycles.use_animated_seed = False
    scene.render.resolution_x = int(manifest.get('render_width', 1024))
    scene.render.resolution_y = int(manifest.get('render_height', 768))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '32'
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = 'Raw'
    scene.render.filepath = str(directory / 'probe')
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    for light in scene.objects:
        if light.type == 'LIGHT':
            light.hide_render = True
    camera = bpy.data.cameras.new('SG line diagnostic camera')
    orthographic = manifest['view'].get('perspective') == 'ORTHO'
    camera.type = 'ORTHO' if orthographic else 'PERSP'
    if orthographic:
        camera.ortho_scale = float(manifest['view'].get('view_distance', 10.0)) * 1.5
    else:
        camera.lens = float(manifest['view']['lens'])
    camera_obj = bpy.data.objects.new('SG line diagnostic camera', camera)
    scene.collection.objects.link(camera_obj)
    camera_obj.matrix_world = Matrix(manifest['view']['matrix_world'])
    scene.camera = camera_obj
    # Preserve the capture direction but widen the frame if the line reaches
    # beyond the original viewport bounds.
    points = [Vector(s['point']) for s in manifest['samples'] if s.get('point')]
    for _ in range(12):
        projected = [world_to_camera_view(scene, camera_obj, p) for p in points]
        if all(p.z > 0 and .035 < p.x < .965 and .035 < p.y < .965
               for p in projected):
            break
        if orthographic:
            camera.ortho_scale *= 1.25
        else:
            camera.lens *= .8
    return camera_obj


def _read_exr(path, width, height):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = array('f', [0]) * (width * height * 4)
        image.pixels.foreach_get(pixels)
        return pixels
    finally:
        bpy.data.images.remove(image)


def _sample_render(scene, camera, manifest, pixels):
    width, height = scene.render.resolution_x, scene.render.resolution_y
    depsgraph = bpy.context.evaluated_depsgraph_get()
    out = []
    for sample in manifest['samples']:
        if not sample.get('point'):
            out.append(None)
            continue
        uv = world_to_camera_view(scene, camera, Vector(sample['point']))
        x, y = int(uv.x * width), int(uv.y * height)
        if uv.z <= 0 or x < 0 or x >= width or y < 0 or y >= height:
            out.append(None)
            continue
        # A different foreground surface can otherwise yield plausible but
        # false weights for a stored point near an occlusion boundary.
        target = Vector(sample['point'])
        if camera.data.type == 'ORTHO':
            origin = target + camera.matrix_world.to_quaternion() @ Vector((0, 0, 1000))
        else:
            origin = camera.matrix_world.translation
        ray = target - origin
        distance = ray.length
        hit, location, _normal, _face, obj, _matrix = scene.ray_cast(
            depsgraph, origin, ray.normalized(),
            distance=distance + max(.005, distance * .002))
        if (not hit or obj is None or obj.name != sample['object']
                or (location - target).length > max(.005, distance * .002)):
            out.append(None)
            continue
        offset = 4 * (y * width + x)
        out.append(tuple(max(0.0, min(1.0, pixels[offset + c]))
                         for c in range(3)))
    return out


def _measure(manifest, directory, mode):
    # Reload for each condition so every counterfactual starts from the same
    # saved material, never from a partly modified prior probe.
    bpy.ops.wm.open_mainfile(filepath=manifest['snapshot'])
    scene = bpy.context.scene
    scene.frame_set(int(manifest['frame']))
    camera = _render_setup(scene, manifest, directory)
    probes, modification_counts = _prepare_materials(scene, mode)
    count = len(manifest['cameras'])
    data = [[0.0] * count if s.get('point') else None
            for s in manifest['samples']]
    for start in range(0, count, 3):
        for index, rgb in probes:
            rgb.outputs[0].default_value = tuple(
                1.0 if index == start + channel else 0.0
                for channel in range(3)) + (1.0,)
        scene.render.filepath = str(directory / f'probe_{mode}_{start:03d}')
        print(f'[SG line] {mode}, views {start + 1}-{min(start + 3, count)}', flush=True)
        bpy.ops.render.render(write_still=True)
        path = Path(scene.render.filepath + '.exr')
        pixels = _read_exr(path, scene.render.resolution_x, scene.render.resolution_y)
        triplets = _sample_render(scene, camera, manifest, pixels)
        for row, rgb_values in zip(data, triplets):
            if row is not None and rgb_values is not None:
                for channel, value in enumerate(rgb_values):
                    if start + channel < count:
                        row[start + channel] = value
        path.unlink(missing_ok=True)
    return data, modification_counts


def _svg(data, labels, title, palette, signed=False):
    width, left, right, top = 1000, 62, 20, 35
    bottom = 62 + 16 * math.ceil(len(labels) / 4)
    height = top + 205 + bottom
    inner_w, inner_h = width - left - right, height - top - bottom
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<rect width="{width}" height="{height}" fill="#20242a"/>',
             f'<text x="{left}" y="23" fill="#fff" font-size="17">{html.escape(title)}</text>']
    marks = (-1, -.5, 0, .5, 1) if signed else (0, .25, .5, .75, 1)
    for mark in marks:
        y = top + inner_h * (1 - mark) / (2 if signed else 1)
        parts.append(f'<path d="M{left} {y:.1f}H{width-right}" stroke="#50545b" stroke-width="1"/>')
        parts.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" fill="#b9bcc4" font-size="12">{mark:+.2f}</text>')
    parts.append(f'<text x="{left}" y="{top+inner_h+22}" fill="#b9bcc4" font-size="12">Start</text>')
    parts.append(f'<text x="{width-right}" y="{top+inner_h+22}" text-anchor="end" fill="#b9bcc4" font-size="12">End</text>')
    for i, label in enumerate(labels):
        segments, points = [], []
        for sample_index, row in enumerate(data):
            if row is None or i >= len(row) or row[i] is None:
                if points:
                    segments.append(points)
                    points = []
                continue
            x = left + inner_w * sample_index / max(1, len(data)-1)
            value = max(-1, min(1, row[i])) if signed else max(0, min(1, row[i]))
            y = top + inner_h * (1 - value) / (2 if signed else 1)
            points.append(f'{x:.1f},{y:.1f}')
        if points:
            segments.append(points)
        color = palette[i % len(palette)]
        for segment in segments:
            parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left + (i % 4)*220}" y="{height-17 - (i//4)*16}" fill="{color}" font-size="12">{html.escape(label)}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


def _write_report(manifest, directory, current, guard_only, ordinary, modifications):
    palette = ['#ff7575', '#80aaff', '#84e19d', '#e9c777', '#b08bff',
               '#80dfe9', '#f69bd5', '#b7da78', '#ffad80', '#8f91eb']
    labels = manifest['cameras']
    rows = []
    graphs = {name: [] for name in ('current', 'guard_only', 'ordinary',
                                    'guard_effect', 'rescue_effect', 'total_dppg_effect')}
    for index, sample in enumerate(manifest['samples']):
        a, b, c = current[index], guard_only[index], ordinary[index]
        if a is None or b is None or c is None:
            for graph in graphs.values():
                graph.append(None)
            rows.append({'sample': index, 't': sample['t'],
                         'status': 'miss' if sample.get('point') is None else 'not_visible_in_replay'})
            continue
        guard_effect = [guard - usual for guard, usual in zip(b, c)]
        rescue_effect = [full - guard for full, guard in zip(a, b)]
        total_effect = [full - usual for full, usual in zip(a, c)]
        values = {'current': a, 'guard_only': b, 'ordinary': c,
                  'guard_effect': guard_effect, 'rescue_effect': rescue_effect,
                  'total_dppg_effect': total_effect}
        for name, graph in graphs.items():
            graph.append(values[name])
        rows.append({'sample': index, 't': sample['t'], 'status': 'hit',
                     'object': sample['object'], 'face': sample['face'],
                     'world_point': sample['point'], 'world_normal': sample['normal'],
                     **values, 'current_total': sum(a), 'guard_only_total': sum(b),
                     'ordinary_total': sum(c),
                     # Preserve the original two-stage fields for older analysis scripts.
                     'dppg_on': a, 'dppg_off': c,
                     'off_minus_on': [usual - full for full, usual in zip(a, c)],
                     'on_total': sum(a), 'off_total': sum(c)})
    handoffs = []
    for previous, following in zip(rows, rows[1:]):
        if previous['status'] != 'hit' or following['status'] != 'hit':
            continue
        for camera_index, camera_name in enumerate(labels):
            changes = {stage: following[stage][camera_index] - previous[stage][camera_index]
                       for stage in ('ordinary', 'guard_only', 'current')}
            handoffs.append({'from_sample': previous['sample'], 'to_sample': following['sample'],
                             'camera_index': camera_index, 'camera': camera_name,
                             **changes,
                             'guard_effect': changes['guard_only'] - changes['ordinary'],
                             'rescue_effect': changes['current'] - changes['guard_only']})
    result = {'schema': 'stablegen.projection-line.v2', 'manifest': 'manifest.json',
              'camera_order': labels, **modifications, 'samples': rows,
              'handoffs': handoffs,
              'attribution_order': ['ordinary', 'guard_only', 'current'],
              'effect_definitions': {
                  'guard_effect': 'guard_only - ordinary',
                  'rescue_effect': 'current - guard_only',
                  'total_dppg_effect': 'current - ordinary'}}
    (directory / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (directory / 'samples.csv').open('w', newline='', encoding='utf-8') as stream:
        fieldnames = ['sample', 't', 'status', 'object', 'face', 'world_x', 'world_y', 'world_z',
                      'ordinary_total', 'guard_only_total', 'current_total']
        for mode in graphs:
            fieldnames.extend(f'{mode}_view_{i:03d}' for i in range(len(labels)))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key, '') for key in fieldnames[:5]}
            if row['status'] == 'hit':
                flat.update(zip(('world_x', 'world_y', 'world_z'), row['world_point']))
                flat.update(ordinary_total=row['ordinary_total'],
                            guard_only_total=row['guard_only_total'],
                            current_total=row['current_total'])
                for mode in graphs:
                    values = row[mode]
                    for i, value in enumerate(values):
                        flat[f'{mode}_view_{i:03d}'] = value
            writer.writerow(flat)
    valid = [r for r in rows if r['status'] == 'hit']
    biggest = sorted(valid, key=lambda r: max((abs(v) for v in r['total_dppg_effect']), default=0), reverse=True)[:8]
    peak = ''.join('<tr><td>{}</td><td>{:.3f}</td><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td></tr>'.format(
        r['sample'], r['t'], html.escape(r['object']),
        max(abs(v) for v in r['guard_effect']),
        max(abs(v) for v in r['rescue_effect']),
        max(abs(v) for v in r['total_dppg_effect']))
        for r in biggest)
    steepest = sorted(handoffs, key=lambda h: abs(h['current']), reverse=True)[:10]
    handoff_table = ''.join(
        '<tr><td>{}–{}</td><td>{}</td><td>{:+.3f}</td><td>{:+.3f}</td><td>{:+.3f}</td><td>{:+.3f}</td><td>{:+.3f}</td></tr>'.format(
            h['from_sample'], h['to_sample'], html.escape(h['camera']),
            h['ordinary'], h['guard_only'], h['current'],
            h['guard_effect'], h['rescue_effect']) for h in steepest)
    settings = ''.join(f'<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>'
                       for key, value in manifest.get('settings', {}).items())
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>StableGen projection line</title>
<style>body{{background:#15181d;color:#e6e9ef;font:14px system-ui;margin:30px auto;max-width:1100px}}
svg{{width:100%;height:auto;border:1px solid #41464e;margin:10px 0}}a{{color:#87baff}}
table{{border-collapse:collapse}}td,th{{border:1px solid #555;padding:6px 12px}}</style></head><body>
<h1>Projection line: view-weight attribution</h1>
<p>{len(valid)} of {len(rows)} positions hit the model. Each colored curve is one camera's contribution to the final projected color. Gaps mean the line missed the model; contribution sums can be below one where fallback/coverage remains.</p>
<p><b>Read the stages in order:</b> Ordinary weighting keeps the saved camera angles, exponent, image alpha and mix but bypasses DPPG guards and feather. DPPG guards only restores the saved guards and feather with coverage rescue disabled. Current setup uses the complete saved graph. Thus <b>guard effect = guards only − ordinary</b>, and <b>coverage rescue effect = current − guards only</b>. Positive values give that camera more ownership; negative values take ownership away. Effects are an ordered counterfactual decomposition, so interactions are assigned to the later rescue stage.</p>
<p><a href="samples.csv">CSV samples</a> · <a href="results.json">Full JSON</a> · <a href="manifest.json">Replay manifest</a> · <a href="replay.py">Replay script</a> · <a href="run.log">Run log</a></p>
<h2>Camera ownership at each stage</h2>
{_svg(graphs['ordinary'], labels, '1. Ordinary weighting (DPPG bypassed)', palette)}
{_svg(graphs['guard_only'], labels, '2. DPPG guards and feather, no coverage rescue', palette)}
{_svg(graphs['current'], labels, '3. Current setup, including coverage rescue', palette)}
<h2>What each DPPG stage changes</h2>
{_svg(graphs['guard_effect'], labels, 'DPPG guard effect: stage 2 minus stage 1', palette, signed=True)}
{_svg(graphs['rescue_effect'], labels, 'Coverage rescue effect: stage 3 minus stage 2', palette, signed=True)}
<p>These curves measure camera ownership, not differences in the cameras' source-image colors. A visible color seam can remain even when the ownership transition is smooth.</p>
<h2>Sharpest camera handoffs</h2>
<p>Each row is one camera's ownership change between adjacent line samples. The ordinary column shows any sharp transition already present before DPPG. The final two columns show how DPPG guards and coverage rescue changed that same transition; their sum with ordinary equals current.</p>
<table><tr><th>Samples</th><th>Camera</th><th>Ordinary change</th><th>Guard-only change</th><th>Current change</th><th>Guard effect on change</th><th>Rescue effect on change</th></tr>{handoff_table}</table>
<h2>Largest total ownership changes</h2><table><tr><th>Sample</th><th>Along line</th><th>Object</th><th>Guard effect</th><th>Rescue effect</th><th>Total DPPG effect</th></tr>{peak}</table>
<h2>Captured settings</h2><table><tr><th>Setting</th><th>Value</th></tr>{settings}</table>
<p>Reproduce: <code>blender --background snapshot.blend --python replay.py -- manifest.json</code></p>
</body></html>'''
    (directory / 'index.html').write_text(page, encoding='utf-8')


def replay(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    snapshot = Path(manifest['snapshot'])
    manifest['snapshot'] = str((directory / snapshot).resolve() if not snapshot.is_absolute()
                               else snapshot)
    if not Path(manifest['snapshot']).is_file():
        raise FileNotFoundError(manifest['snapshot'])
    ordinary, ordinary_mods = _measure(manifest, directory, 'ordinary')
    guard_only, guard_mods = _measure(manifest, directory, 'guard_only')
    current, _ = _measure(manifest, directory, 'current')
    modifications = {'dppg_bypass_links': ordinary_mods['dppg_bypass_links'],
                     'rescue_bypass_links': guard_mods['rescue_bypass_links']}
    _write_report(manifest, directory, current, guard_only, ordinary, modifications)
    print(f'[SG line] Report ready: {directory / "index.html"}', flush=True)


if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    if len(args) != 1:
        raise SystemExit('Usage: blender --background snapshot.blend --python replay.py -- manifest.json')
    replay(args[0])
