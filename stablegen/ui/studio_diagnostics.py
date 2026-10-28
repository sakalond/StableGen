"""Reversible diagnostic displays used by Studio, backed by saved Blender IDs.

Existing diagnostics may replace UVs, materials and render settings. Run them
on private mesh/material copies, retaining the originals across undo and saves.
"""
import json
import uuid

import bpy

from ..utils import sg_modal_active

MODES = [
    ('TEXTURES', 'Projected textures', 'Return to the original mesh and materials'),
    ('SOLID', 'View colors', 'One solid color per camera, through the real projection pipeline'),
    ('GRID', 'Projection grid', 'Check projected image alignment and distortion'),
    ('COVERAGE', 'Coverage', 'Average or strongest camera visibility, shown on a logarithmic heatmap to reveal weak coverage; not normalized RGB contributions'),
    ('VISIBILITY', 'Visibility', 'Inspect which surfaces the cameras can see'),
    ('WEIGHT', 'Camera visibility weight', 'Unnormalized visibility weight for the chosen camera: black is zero, white is full; not its relative share of the final RGB mixture'),
    ('FEATHER', 'Image-frame fade', 'Chosen camera’s image-frame fade: white is full strength, black is faded out. This is not DPPG guard feathering'),
    ('CONFIDENCE', 'Confidence', 'Inspect the actual projection confidence graph'),
    ('SEAMS', 'UV seams', 'Inspect the mesh UV layout'),
    ('DPPG', 'DPPG masks', 'Depth / part projection guard effects: red is guard attenuation, blue is feather attenuation, magenta is both, dark gray is unchanged. Only valid source footprints are counted'),
]
# Keep the panel summary concise; detailed signal scales and legends belong in
# enum/hover help, including for DPPG where the permanent legend was removed.
DISPLAY_SUMMARIES = {
    'TEXTURES': 'Original projected textures.', 'SOLID': 'One source color per camera.',
    'GRID': 'Projected image alignment and distortion.', 'COVERAGE': 'Surface observation across cameras.',
    'VISIBILITY': 'Surfaces visible to the chosen camera.', 'WEIGHT': 'Chosen camera’s visibility weight.',
    'FEATHER': 'Chosen camera’s image-frame fade.', 'CONFIDENCE': 'Source quality and context masks.',
    'SEAMS': 'Mesh UV layout.', 'DPPG': 'Guard rejection and boundary feathering.',
}
OPERATORS = {
    'SOLID': 'debug_solid_colors', 'GRID': 'debug_grid_pattern',
    'COVERAGE': 'debug_coverage_heatmap', 'VISIBILITY': 'debug_visibility_material',
    'WEIGHT': 'debug_per_camera_weight', 'FEATHER': 'debug_feather_preview',
    'CONFIDENCE': 'debug_confidence_diagnostic', 'SEAMS': 'debug_uv_seam_viz',
    'DPPG': 'debug_dppg_material',
}
OPTIONS = {'mode', 'camera', 'grid_size', 'coverage', 'confidence', 'scope', 'stage'}
_OWNED = '_sg_studio_preview'


class SGStudioPreviewSlot(bpy.types.PropertyGroup):
    material: bpy.props.PointerProperty(type=bpy.types.Material)
    object_material: bpy.props.PointerProperty(type=bpy.types.Material)
    link: bpy.props.StringProperty()


class SGStudioPreviewMesh(bpy.types.PropertyGroup):
    object: bpy.props.PointerProperty(type=bpy.types.Object)
    mesh: bpy.props.PointerProperty(type=bpy.types.Mesh)
    slots: bpy.props.CollectionProperty(type=SGStudioPreviewSlot)
    active_index: bpy.props.IntProperty()


class SGStudioPreviewCamera(bpy.types.PropertyGroup):
    object: bpy.props.PointerProperty(type=bpy.types.Object)
    had_color: bpy.props.BoolProperty()
    color: bpy.props.StringProperty()


class SGStudioDiagnostics(bpy.types.PropertyGroup):
    mode: bpy.props.EnumProperty(name='Viewport display', items=MODES, default='TEXTURES',
                                description='Choose the diagnostic material to inspect. Projected textures restores the original meshes, materials and render settings')
    camera: bpy.props.PointerProperty(name='Camera', type=bpy.types.Object,
                                      description='Camera whose diagnostic signal is displayed; this choice is separate from viewport object selection',
                                      poll=lambda _self, obj: obj.type == 'CAMERA')
    grid_size: bpy.props.IntProperty(name='Grid cell size', default=64, min=8, max=512,
                                    description='Checkerboard cell size in pixels')
    coverage: bpy.props.EnumProperty(name='Coverage metric', default='AVERAGE',
                                    description='Average divides summed camera visibility by camera count; Best camera uses the strongest visibility weight. The heatmap stretches low values logarithmically', items=[
        ('AVERAGE', 'Average', 'Mean visibility weight across all cameras; improves only when more cameras support the surface'),
        ('MAX', 'Best camera', 'Strongest camera visibility weight, regardless of how many other views cover it')])
    confidence: bpy.props.EnumProperty(name='Confidence signal', default='QUALITY',
                                      description='Choose the geometric quality or context mask from the projection graph. Black is 0 and white is 1; meanings depend on the selected signal', items=[
        ('QUALITY', 'Best source quality', 'Best accepted source quality, independent of which view dominates RGB; brighter means better observed'),
        ('ACQUISITION', 'Camera quality', 'Chosen camera’s raw geometric acquisition quality; brighter means better observed'),
        ('TRUSTED', 'Trusted context', 'White means source quality exceeds the current trusted-quality threshold'),
        ('REPAIR', 'Repair mask', 'White means source quality does not exceed the trusted-quality threshold and needs repair'),
        ('WRITE', 'Write mask', 'White permits this camera’s generated pixels to contribute; black excludes them'),
        ('GUARD', 'Depth / part guard', 'Chosen camera’s guard acceptance: brighter permits more projection')])
    scope: bpy.props.EnumProperty(name='Camera scope', default='ALL',
                                 description='Inspect the strongest guard/feather effect across all valid cameras, or only the camera chosen in this inspector', items=[
        ('ALL', 'All cameras', 'Strongest effect from any valid camera'),
        ('ACTIVE', 'Chosen camera', 'Inspect only the camera chosen below, independently of viewport selection')])
    stage: bpy.props.EnumProperty(name='Guard stage', default='CURRENT',
                                 description='Compare original guard/feather fields before rescue with effective fields after coverage rescue and visibility limits', items=[
        ('CURRENT', 'Effective masks', 'After coverage rescue and occlusion limits'),
        ('BEFORE_RESCUE', 'Before coverage rescue', 'Original guard and feather fields')])
    # Persistent RNA references keep the baseline alive, including after undo or
    # saving while a diagnostic display is active. No name-based restoration.
    meshes: bpy.props.CollectionProperty(type=SGStudioPreviewMesh)
    cameras: bpy.props.CollectionProperty(type=SGStudioPreviewCamera)
    token: bpy.props.StringProperty()
    render_state: bpy.props.StringProperty()
    original_camera: bpy.props.PointerProperty(type=bpy.types.Object)


def active(scene):
    state = scene.sg_studio_diagnostics
    return bool(state.token and state.meshes)


def _targets(context):
    from ..debug_tools import _get_target_meshes
    return [obj for obj in _get_target_meshes(context)
            if obj.library is None and obj.data.library is None]


def _snapshot(context, state):
    targets = _targets(context)
    if not targets:
        raise ValueError('Select an editable mesh to inspect.')
    state.token = uuid.uuid4().hex
    for obj in targets:
        record = state.meshes.add()
        record.object, record.mesh = obj, obj.data
        record.active_index = obj.active_material_index
        for slot in obj.material_slots:
            item = record.slots.add()
            item.material, item.link = slot.material, slot.link
            # Blender retains an object override even while the slot uses DATA.
            # Preserve it too; otherwise a discarded preview can retain images
            # or silently become visible when the user changes the slot linkage.
            slot.link = 'OBJECT'
            item.object_material = slot.material
            slot.link = item.link
    for camera in context.scene.objects:
        if camera.type == 'CAMERA':
            record = state.cameras.add()
            record.object = camera
            record.had_color = '_sg_debug_projection_color' in camera
            if record.had_color:
                record.color = json.dumps(tuple(camera['_sg_debug_projection_color']))
    scene = context.scene
    state.original_camera = scene.camera
    values = {'engine': scene.render.engine}
    for key in ('device', 'shading_system', 'use_osl'):
        if hasattr(scene.cycles, key):
            values[key] = getattr(scene.cycles, key)
    state.render_state = json.dumps(values)


def _cleanup():
    # Remove only data created by this preview and no longer in use. A user who
    # deliberately reuses a preview material elsewhere retains that material.
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for data in list(collection):
            if data.get(_OWNED) and data.users == 0:
                collection.remove(data)
    while True:
        unused = [group for group in bpy.data.node_groups if group.get(_OWNED) and group.users == 0]
        if not unused:
            break
        for group in unused:
            bpy.data.node_groups.remove(group)


def restore(scene, *, finish=True):
    state = scene.sg_studio_diagnostics
    if not state.token:
        state.mode = 'TEXTURES'
        return
    for record in state.meshes:
        obj = record.object
        if obj is None or record.mesh is None or obj.type != 'MESH':
            continue
        obj.data = record.mesh
        for slot, saved in zip(obj.material_slots, record.slots):
            slot.link = 'OBJECT'
            slot.material = saved.object_material
            slot.link = saved.link
        obj.active_material_index = record.active_index
    for record in state.cameras:
        if record.object is not None:
            if record.had_color:
                record.object['_sg_debug_projection_color'] = tuple(json.loads(record.color))
            elif '_sg_debug_projection_color' in record.object:
                del record.object['_sg_debug_projection_color']
    values = json.loads(state.render_state or '{}')
    if 'engine' in values:
        scene.render.engine = values.pop('engine')
    for key, value in values.items():
        setattr(scene.cycles, key, value)
    scene.camera = state.original_camera
    if finish:
        state.meshes.clear()
        state.cameras.clear()
        state.token, state.render_state = '', ''
        state.original_camera = None
        state.mode = 'TEXTURES'
    _cleanup()


def _private_meshes(state):
    materials, groups = {}, {}

    def private_group(source):
        if source not in groups:
            group = source.copy()
            group.use_fake_user = False
            groups[source] = group
            group[_OWNED] = state.token
            for node in group.nodes:
                if node.type == 'GROUP' and node.node_tree:
                    node.node_tree = private_group(node.node_tree)
        return groups[source]

    def private_material(source):
        if source is None:
            return None
        if source not in materials:
            mat = source.copy()
            mat.use_fake_user = False
            materials[source] = mat
            mat[_OWNED] = state.token
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'GROUP' and node.node_tree:
                        node.node_tree = private_group(node.node_tree)
        return materials[source]

    targets = []
    for record in state.meshes:
        obj = record.object
        if obj is None or obj.type != 'MESH' or record.mesh is None:
            continue
        obj.data = record.mesh.copy()
        obj.data.use_fake_user = False
        obj.data[_OWNED] = state.token
        for index, source in enumerate(record.mesh.materials):
            obj.data.materials[index] = private_material(source)
        for slot, saved in zip(obj.material_slots, record.slots):
            slot.link = 'OBJECT'
            slot.material = private_material(saved.object_material)
            slot.link = saved.link
        targets.append(obj)
    if not targets:
        raise ValueError('The inspected meshes no longer exist.')
    return targets


def apply(context):
    state, scene = context.scene.sg_studio_diagnostics, context.scene
    if state.mode == 'TEXTURES':
        restore(scene)
        return
    mode = state.mode
    if state.token:
        restore(scene, finish=False)
    else:
        _snapshot(context, state)
    collections = (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.images)
    before = [set(collection) for collection in collections]
    original_selection = set(context.selected_objects)
    original_active = context.view_layer.objects.active
    try:
        meshes = _private_meshes(state)
        camera = state.camera
        if camera is None or camera.name not in scene.objects:
            camera = next((obj for obj in context.selected_objects if obj.type == 'CAMERA'), scene.camera)
        if camera is None or camera.name not in scene.objects:
            camera = next((obj for obj in scene.objects if obj.type == 'CAMERA'), None)
        if camera is None:
            raise ValueError('Add cameras before inspecting projection.')
        state.camera = camera
        selected = meshes + [camera]
        kwargs = {'grid_size': state.grid_size} if mode == 'GRID' else (
            {'mode': state.coverage} if mode == 'COVERAGE' else
            {'mode': state.confidence} if mode == 'CONFIDENCE' else
            {'scope': state.scope, 'stage': state.stage} if mode == 'DPPG' else {})
        with context.temp_override(selected_objects=selected, selected_editable_objects=selected,
                                   active_object=camera, object=camera):
            op = getattr(bpy.ops.stablegen, OPERATORS[mode])
            if not op.poll():
                raise ValueError('This diagnostic requires meshes, cameras and Debug enabled in preferences.')
            result = op('EXEC_DEFAULT', **kwargs)
            if 'FINISHED' not in result:
                raise ValueError('This signal is unavailable. Reproject textures with the relevant feature enabled.')
    except Exception:
        # Include new IDs even when an existing diagnostic fails halfway through.
        for collection, previous in zip(collections, before):
            for data in set(collection) - previous:
                data[_OWNED] = state.token
        restore(scene)
        raise
    finally:
        for obj in context.view_layer.objects:
            obj.select_set(obj in original_selection)
        if original_active and original_active.name in context.view_layer.objects:
            context.view_layer.objects.active = original_active
    for collection, previous in zip(collections, before):
        for data in set(collection) - previous:
            data[_OWNED] = state.token


class SG_OT_StudioDiagnostic(bpy.types.Operator):
    bl_idname = 'stablegen.studio_diagnostic'
    bl_label = 'Change Diagnostic Display'
    bl_description = 'Inspect projection using temporary data; restore the original textures at any time'
    bl_options = {'REGISTER', 'UNDO'}

    setting: bpy.props.StringProperty(default='mode', options={'HIDDEN'})
    value: bpy.props.StringProperty(default='"TEXTURES"', options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and not sg_modal_active(context)

    def execute(self, context):
        if self.setting not in OPTIONS:
            return {'CANCELLED'}
        state = context.scene.sg_studio_diagnostics
        try:
            value = json.loads(self.value)
            if self.setting == 'camera':
                value = bpy.data.objects.get(value) if value else None
            setattr(state, self.setting, value)
            apply(context)
        except Exception as exc:
            restore(context.scene)
            self.report({'WARNING'}, str(exc).split('\n')[0])
            return {'FINISHED'}  # rollback itself must be undoable
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return {'FINISHED'}


CLASSES = [SGStudioPreviewSlot, SGStudioPreviewMesh, SGStudioPreviewCamera,
           SGStudioDiagnostics, SG_OT_StudioDiagnostic]


def register():
    bpy.types.Scene.sg_studio_diagnostics = bpy.props.PointerProperty(type=SGStudioDiagnostics)


def unregister():
    for scene in bpy.data.scenes:
        restore(scene)
    del bpy.types.Scene.sg_studio_diagnostics
