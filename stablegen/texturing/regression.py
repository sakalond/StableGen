"""Unattended, isolated preset comparisons using the normal generation and bake operators."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import json
import math
import time

import bpy
from mathutils import Vector

from ..core import ADDON_PKG


# A second entry is a real second generation pass over the first pass's material.
CASES = (
    ('SDXL Characters', ('CHARACTERS',)),
    ('SDXL Quick Draft', ('QUICK DRAFT',)),
    ('SDXL Characters + UV Inpainting', ('CHARACTERS', 'UV INPAINTING')),
    ('SDXL Characters + Local Refine', ('CHARACTERS', 'LOCAL REFINE')),
    ('Qwen Precise', ('QWEN PRECISE',)),
    ('Qwen Safe', ('QWEN SAFE',)),
    ('Qwen Safe + Local Edit', ('QWEN SAFE', 'LOCAL EDIT (QWEN)')),
    ('Qwen Safe + Refine (Qwen)', ('QWEN SAFE', 'REFINE (QWEN)')),
)

DETAIL_PRESETS = frozenset({'LOCAL REFINE', 'LOCAL EDIT (QWEN)'})
_camera_enum_cache = {}
_model_cache = {}
_model_enum_history = []  # Keep Blender's borrowed enum strings alive after refresh.
_SELECT_MODEL = '__SELECT_MODEL__'


def model_cache(context):
    wrapper = context.preferences.addons.get(ADDON_PKG) if context else None
    server = wrapper.preferences.server_address if wrapper else ''
    if server not in _model_cache:
        _model_cache[server] = {'loading': False, 'message': '', 'ids': {},
                                'sdxl': [], 'qwen_image_edit': []}
    return server, _model_cache[server]


def set_model_items(cache, architecture, names):
    items = [(_SELECT_MODEL, 'Select a model', 'Choose an installed ComfyUI model', 0)]
    for name in sorted(set(names or [])):
        key = (architecture, name)
        if key not in cache['ids']:
            cache['ids'][key] = len(cache['ids']) + 1
        items.append((name, name, 'ComfyUI model: ' + name, cache['ids'][key]))
    cache[architecture] = items
    _model_enum_history.append(items)


def regression_model_items(context, architecture):
    _, cache = model_cache(context)
    if not cache[architecture]:
        set_model_items(cache, architecture, [])
    return cache[architecture]


def sdxl_model_items(self, context):
    return regression_model_items(context, 'sdxl')


def qwen_model_items(self, context):
    return regression_model_items(context, 'qwen_image_edit')


def refresh_regression_models(context):
    from ..core.server_api import _fetch_api_list, fetch_diffusion_models
    from ..core.state import _run_async
    from .qwen_models import is_qwen_image_21_model
    server, cache = model_cache(context)
    if cache['loading']:
        return
    cache['loading'] = True
    cache['message'] = 'Loading models from ComfyUI...'

    def fetch():
        qwen_models = fetch_diffusion_models(server)
        if qwen_models is not None:
            qwen_models = [name for name in qwen_models
                           if not is_qwen_image_21_model(name)]
        return {'sdxl': _fetch_api_list(server, '/models/checkpoints'),
                'qwen_image_edit': qwen_models}

    def done(result):
        cache['loading'] = False
        for architecture in ('sdxl', 'qwen_image_edit'):
            set_model_items(cache, architecture, result.get(architecture) if result else [])
        if not result or any(result.get(architecture) is None for architecture in ('sdxl', 'qwen_image_edit')):
            cache['message'] = 'Could not load all model lists. Check ComfyUI and refresh.'
        elif len(cache['sdxl']) == 1 or len(cache['qwen_image_edit']) == 1:
            cache['message'] = 'No models found for one architecture. Install models and refresh.'
        else:
            cache['message'] = ''
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

    _run_async(fetch, done)


class SG_OT_RefreshRegressionModels(bpy.types.Operator):
    bl_idname = 'stablegen.refresh_regression_models'
    bl_label = 'Refresh Model Lists'
    bl_description = 'Load SDXL checkpoints and Qwen diffusion/GGUF models from ComfyUI'

    @classmethod
    def poll(cls, context):
        server, cache = model_cache(context)
        return bool(server) and not cache['loading'] and SG_OT_TexturingRegression._active is None

    def execute(self, context):
        refresh_regression_models(context)
        return {'FINISHED'}


def detail_camera_items(self, context):
    # Blender retains enum string pointers; keep each returned list alive.
    names = tuple(sorted(obj.name for obj in context.scene.objects if obj.type == 'CAMERA')) if context else ()
    if names not in _camera_enum_cache:
        _camera_enum_cache[names] = [('NONE', 'None (skip local cases)', 'Skip Local Refine and Local Edit')]
        _camera_enum_cache[names].extend(('CAMERA:' + name, name, 'Use this camera for the local second passes only')
                                         for name in names)
    return _camera_enum_cache[names]


def park_detail_camera(scene):
    """Keep the copied detail camera out of every scene-camera enumeration."""
    for camera in list(scene.objects):
        if camera.type != 'CAMERA' or not camera.get('_sg_regression_detail_camera'):
            continue
        # An ID reference preserves the isolated camera, its parent/constraints
        # and lens through orphan cleanup without linking it into the scene.
        scene['_sg_regression_pending_detail_camera'] = camera
        if scene.camera == camera:
            scene.camera = next((obj for obj in scene.objects
                                 if obj.type == 'CAMERA' and obj != camera), None)
        for collection in list(camera.users_collection):
            collection.objects.unlink(camera)


def configure_camera_scope(scene, preset, edit_prompt):
    detail = preset in DETAIL_PRESETS
    scene['_sg_regression_camera_scope'] = 'detail' if detail else 'base'
    if detail:
        camera = scene.get('_sg_regression_pending_detail_camera')
        if camera is not None:
            scene.collection.objects.link(camera)
            del scene['_sg_regression_pending_detail_camera']
            scene.camera = camera
        scene.comfyui_prompt = edit_prompt
        scene.texture_prompt = edit_prompt
        scene.use_separate_texture_prompt = True
        scene.refine_prompt = edit_prompt
        scene.use_camera_prompts = False


@contextmanager
def internal_operators():
    from .. import utils
    previous = utils._sg_bypass_modal_check
    utils._sg_bypass_modal_check = True
    try:
        yield
    finally:
        utils._sg_bypass_modal_check = previous


def copy_scene(context, source, name):
    """FULL_COPY also remaps rigs/constraints; repair string-based camera references."""
    context.window.scene = source
    key = '_sg_regression_source_name'
    saved = [(obj, key in obj, obj.get(key)) for obj in source.objects]
    try:
        for obj, _, _ in saved:
            obj[key] = obj.name
        if bpy.ops.scene.new(type='FULL_COPY') != {'FINISHED'}:
            raise RuntimeError('Could not duplicate the source scene')
        scene = context.window.scene
    finally:
        for obj, existed, value in saved:
            if existed:
                obj[key] = value
            else:
                del obj[key]
    scene.name = name
    mapping = {obj[key]: obj.name for obj in scene.objects if key in obj}
    for obj in scene.objects:
        original = source.objects.get(obj.get(key, ''))
        if original and obj.data and obj.data == original.data:
            obj.data = obj.data.copy()
    for collection in (scene.camera_prompts, scene.sg_camera_order):
        for item in collection:
            item.name = mapping.get(item.name, item.name)
    # Node groups can remain shared even when object/material IDs are copied.
    groups, materials = {}, {}
    def isolate_groups(tree):
        for node in tree.nodes:
            if node.type == 'GROUP' and node.node_tree:
                old = node.node_tree
                if old not in groups:
                    groups[old] = old.copy()
                    isolate_groups(groups[old])
                node.node_tree = groups[old]
    for obj in scene.objects:
        for slot in obj.material_slots:
            mat = slot.material
            if not mat:
                continue
            if mat not in materials:
                materials[mat] = mat.copy()
                if materials[mat].use_nodes:
                    isolate_groups(materials[mat].node_tree)
            slot.material = materials[mat]
    if scene.world:
        scene.world = scene.world.copy()
        if scene.world.use_nodes:
            isolate_groups(scene.world.node_tree)
    return scene


def targets(context):
    from .rendering import _texture_bake_targets
    candidates = [o for o in context.view_layer.objects
                  if o.type == 'MESH' and not o.hide_get() and not o.hide_render]
    if context.scene.texture_objects == 'selected':
        candidates = [o for o in candidates if o.select_get()]
    return _texture_bake_targets(candidates)[0]


def save_scene(scene, path):
    bpy.data.libraries.write(str(path), {scene}, path_remap='ABSOLUTE', fake_user=True)


def verify_baked(objects, output_dir):
    """Do not treat a vanished modal (including exceptions) as bake success."""
    for obj in objects:
        expected = Path(output_dir) / (obj.name + '.png')
        mat = obj.active_material
        if not expected.is_file() or not mat or not mat.use_nodes:
            raise RuntimeError(f'Bake did not finish for {obj.name}')
        images = [n.image for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
        if not any(Path(bpy.path.abspath(im.filepath)).resolve() == expected.resolve() for im in images):
            raise RuntimeError(f'Baked material was not applied to {obj.name}')
        for image in images:
            image.pack()


def build_comparison(context, entries, name):
    """Freeze evaluated, baked meshes so moving them cannot move their projection."""
    comparison = bpy.data.scenes.new(name)
    cursor = 0.0
    for index, entry in enumerate(entries):
        collection = bpy.data.collections.new(f'{index + 1:02d} {entry["label"]}')
        comparison.collection.children.link(collection)
        meshes = []
        source = bpy.data.scenes.get(entry.get('scene', ''))
        if entry['status'] == 'passed' and source:
            context.window.scene = source
            depsgraph = context.evaluated_depsgraph_get()
            for obj_name in entry['objects']:
                obj = source.objects[obj_name]
                evaluated = obj.evaluated_get(depsgraph)
                mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
                mesh.materials.clear()
                mesh.materials.append(obj.active_material)
                for face in mesh.polygons:
                    face.material_index = 0
                result = bpy.data.objects.new(f'{index + 1:02d} {obj.name}', mesh)
                collection.objects.link(result)
                result.matrix_world = evaluated.matrix_world.copy()
                meshes.append(result)
        corners = [obj.matrix_world @ Vector(c) for obj in meshes for c in obj.bound_box]
        lower = Vector(tuple(min(c[i] for c in corners) for i in range(3))) if corners else Vector((0, 0, 0))
        upper = Vector(tuple(max(c[i] for c in corners) for i in range(3))) if corners else Vector((2, 2, 2))
        width = max(upper.x - lower.x, 1.0)
        height = max(upper.z - lower.z, 1.0)
        shift = Vector((cursor - lower.x, -lower.y, -lower.z))
        for obj in meshes:
            obj.location += shift
        curve = bpy.data.curves.new(f'Label {index + 1:02d}', 'FONT')
        curve.body = f'{index + 1:02d}  {entry["label"]}\n{entry["status"].upper()}'
        if entry.get('skip_reason'):
            curve.body += '\n' + entry['skip_reason']
        curve.size = min(height * .075, width / max(len(entry['label']), 12) * 1.5)
        label = bpy.data.objects.new(curve.name, curve)
        collection.objects.link(label)
        label.location = (cursor, -.2 * height, 0)
        label.rotation_euler.x = math.pi / 2
        label['result'] = entry.get('error', '') or entry.get('skip_reason', '') or entry.get('file', '')
        cursor += width + max(width * .3, height * .2)
    context.window.scene = comparison
    return comparison


class SG_OT_TexturingRegression(bpy.types.Operator):
    bl_idname = 'stablegen.texturing_regression'
    bl_label = 'Run Texturing Regression'
    bl_description = ('Texture independent copies of the current scene with eight workflows, '
                      'bake each result and create a labeled comparison scene. Uses the current '
                      'prompt, seed, target selection and camera placement. May take a long time')

    bake_resolution: bpy.props.IntProperty(name='Bake Resolution', default=2048, min=128, max=8192)
    sdxl_model: bpy.props.EnumProperty(name='SDXL Model', items=sdxl_model_items,
                                     description='Choose a checkpoint installed in ComfyUI')
    qwen_model: bpy.props.EnumProperty(name='Qwen Model', items=qwen_model_items,
                                     description='Choose a Qwen diffusion or GGUF model installed in ComfyUI')
    detail_camera: bpy.props.EnumProperty(name='Detail Camera', items=detail_camera_items,
                                         description='Excluded from base generation; used alone for Local Refine and Local Edit')
    detail_edit_prompt: bpy.props.StringProperty(name='Detail Edit Prompt',
                                               description='Specific change to make in the detail camera view, for both local second passes')
    _active = None
    status = ''

    @classmethod
    def poll(cls, context):
        from ..utils import sg_modal_active
        from ..ui import queue
        return (context.window is not None and context.mode == 'OBJECT'
                and cls._active is None and not sg_modal_active(context)
                and not queue._queue_processing
                and not getattr(context.window_manager, 'sg_batch_running', False))

    def invoke(self, context, event):
        from ..core import state
        _, cache = model_cache(context)
        architecture = context.scene.model_architecture
        if architecture in {'sdxl', 'qwen_image_edit'} and state._cached_checkpoint_architecture == architecture:
            names = [item[0] for item in state._cached_checkpoint_list
                     if not item[0].startswith(('NO_', 'NONE_'))]
            set_model_items(cache, architecture, names)
        if architecture in {'sdxl', 'qwen_image_edit'}:
            if context.scene.model_name in {item[0] for item in regression_model_items(context, architecture)}:
                setattr(self, 'sdxl_model' if architecture == 'sdxl' else 'qwen_model', context.scene.model_name)
        refresh_regression_models(context)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        for index, (label, _) in enumerate(CASES):
            layout.label(text=f'{index + 1}. {label}')
        layout.separator()
        layout.prop(self, 'bake_resolution')
        layout.prop(self, 'sdxl_model')
        layout.prop(self, 'qwen_model')
        layout.operator('stablegen.refresh_regression_models', icon='FILE_REFRESH')
        _, cache = model_cache(context)
        if cache['message']:
            layout.label(text=cache['message'], icon='INFO')
        layout.separator()
        layout.prop(self, 'detail_camera')
        detail = layout.column()
        detail.enabled = self.detail_camera != 'NONE'
        detail.prop(self, 'detail_edit_prompt')
        if self.detail_camera == 'NONE':
            layout.label(text='Local Refine and Local Edit will be skipped.', icon='INFO')
        else:
            layout.label(text='Detail camera is excluded from all base and full-scene passes.')
        layout.label(text='Uses the current prompt and cameras; keeps the source scene intact.')
        layout.label(text='Stop finishes the current pass before creating a partial comparison.')

    def execute(self, context):
        from .generator import ComfyUIGenerate
        # Capture the enum name before scene duplication changes its available items.
        self.detail_camera_name = '' if self.detail_camera == 'NONE' else self.detail_camera.removeprefix('CAMERA:')
        if self.detail_camera_name:
            camera = context.scene.objects.get(self.detail_camera_name)
            if not camera or camera.type != 'CAMERA':
                self.report({'ERROR'}, 'Choose a detail camera in the current scene')
                return {'CANCELLED'}
            if not self.detail_edit_prompt.strip():
                self.report({'ERROR'}, 'Enter a Detail Edit Prompt for the local passes')
                return {'CANCELLED'}
        source_targets = {obj.name for obj in targets(context)}
        if not source_targets or not any(o.type == 'CAMERA' and o.name != self.detail_camera_name
                                         for o in context.scene.objects):
            self.report({'ERROR'}, 'The scene needs visible target meshes and at least one base camera besides the detail camera')
            return {'CANCELLED'}
        _, cache = model_cache(context)
        if cache['loading']:
            self.report({'ERROR'}, 'Wait for the model lists to finish loading')
            return {'CANCELLED'}
        self.model_selections = {'sdxl': self.sdxl_model, 'qwen_image_edit': self.qwen_model}
        for architecture, model in self.model_selections.items():
            if model == _SELECT_MODEL or model not in {item[0] for item in regression_model_items(context, architecture)}:
                self.report({'ERROR'}, 'Choose both an SDXL model and a Qwen model from the dropdowns')
                return {'CANCELLED'}
        if not ComfyUIGenerate.poll(context):
            self.report({'ERROR'}, 'Connect ComfyUI and configure the StableGen output directory first')
            return {'CANCELLED'}
        self.source = context.scene
        self.window = context.window
        self.area = context.area
        self.entries = []
        self.index = -1
        self.pass_index = 0
        self.phase = 'next'
        self.stop_requested = False
        self.scene = None
        self.bake_restore = None
        self.timer = None
        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        self.root = Path(bpy.path.abspath(context.preferences.addons[ADDON_PKG].preferences.output_dir)) / 'regression' / stamp
        try:
            self.root.mkdir(parents=True, exist_ok=False)
            self.template = copy_scene(context, self.source, 'Regression Source ' + stamp)
            for obj in self.template.objects:
                obj['_sg_regression_target'] = obj.get('_sg_regression_source_name') in source_targets
                obj['_sg_regression_detail_camera'] = (obj.type == 'CAMERA'
                                                       and obj.get('_sg_regression_source_name') == self.detail_camera_name)
            self.template.use_fake_user = True
            save_scene(self.template, self.root / 'source.blend')
            self.source_settings = {key: getattr(self.template, key) for key in
                                    ('comfyui_prompt', 'comfyui_negative_prompt', 'texture_prompt',
                                     'use_separate_texture_prompt', 'seed', 'texture_objects', 'use_camera_prompts',
                                     'sg_use_custom_camera_order', 'sequential_custom_camera_order')}
            self.timer = context.window_manager.event_timer_add(.5, window=self.window)
            self._next_tick = time.monotonic()
            type(self)._active = self
            type(self).status = 'Preparing regression'
            context.window_manager.modal_handler_add(self)
            self.write_report('running')
            return {'RUNNING_MODAL'}
        except Exception as exc:
            context.window.scene = self.source
            self.cleanup(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def write_report(self, status):
        data = {'status': status, 'source_scene': self.source.name,
                'phase': self.phase,
                'source_file': bpy.data.filepath, 'seed': self.source.seed,
                'bake_resolution': self.bake_resolution, 'cases': self.entries,
                'detail_camera': self.detail_camera_name,
                'detail_edit_prompt': self.detail_edit_prompt.strip(),
                'planned_cases': [label for label, _ in CASES]}
        path = self.root / 'report.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary.replace(path)

    def apply_pass(self, context):
        from ..ui.presets import PRESETS
        preset = CASES[self.index][1][self.pass_index]
        arch = PRESETS[preset]['model_architecture']
        self.scene.architecture_mode = arch
        self.preset = preset
        self.deadline = time.monotonic() + 60
        self.phase = 'refresh'
        with internal_operators():
            bpy.ops.stablegen.refresh_checkpoint_list()
            bpy.ops.stablegen.refresh_lora_list()

    def configure_pass(self, context):
        self.scene.stablegen_preset = self.preset
        with internal_operators():
            if bpy.ops.stablegen.apply_preset() != {'FINISHED'}:
                raise RuntimeError(f'Cannot apply {self.preset}; check its required models and LoRAs')
        for key, value in self.source_settings.items():
            setattr(self.scene, key, value)
        self.scene.generation_mode = 'standard'
        self.scene.ask_object_prompts = False
        self.scene.control_after_generate = 'fixed'
        configure_camera_scope(self.scene, self.preset, self.detail_edit_prompt.strip())
        self.phase = 'ready'
        self.ready_ticks = 0

    def start_generation(self, context):
        from .view_plan import generation_cameras
        # Every pass gets an isolated output path and an explicit camera scope.
        self.scene['_sg_regression_output'] = str(self.root / f'{self.index + 1:02d}' / f'pass-{self.pass_index + 1}')
        cameras = generation_cameras(self.scene)
        if not cameras:
            raise RuntimeError('No cameras available for this regression pass')
        # Selection changes during generation/baking; always restore the captured targets.
        for obj in context.view_layer.objects:
            obj.select_set(bool(obj.get('_sg_regression_target')) or obj in cameras)
        self.scene.texture_objects = 'selected'
        model = self.model_selections[self.scene.model_architecture]
        if model:
            self.scene.model_name = model
        if not self.scene.model_name or self.scene.model_name.startswith(('NO_', 'NONE_')):
            raise RuntimeError('No usable model selected')
        self.entry['passes'].append({'preset': self.preset, 'model': self.scene.model_name,
                                     'method': self.scene.generation_method,
                                     'qwen_method': self.scene.qwen_generation_method,
                                     'cameras': [obj.name for obj in cameras],
                                     'camera_scope': self.scene['_sg_regression_camera_scope'],
                                     'prompt': self.scene.texture_prompt if self.scene.use_separate_texture_prompt else self.scene.comfyui_prompt})
        self.scene.sg_last_gen_error = False
        self.phase = 'generation'
        self.write_report('running')
        with internal_operators():
            result = bpy.ops.object.test_stable('INVOKE_DEFAULT')
        if result != {'RUNNING_MODAL'}:
            raise RuntimeError(f'Generation did not start: {result}')
        self.phase = 'generation'

    def start_bake(self, context):
        from .rendering import BakeTextures
        self.bake_objects = [o for o in context.view_layer.objects if o.get('_sg_regression_target')]
        if not self.bake_objects:
            raise RuntimeError('No targets remain for baking')
        BakeTextures._objects = self.bake_objects
        BakeTextures._current_index = 0
        BakeTextures._phase = 'unwrap'
        BakeTextures._total_objects = len(self.bake_objects)
        self.bake_restore = {
            'engine': self.scene.render.engine,
            'persistent': self.scene.render.use_persistent_data,
            'cycles': {key: getattr(self.scene.cycles, key) for key in
                       ('device', 'samples', 'max_bounces', 'use_denoising')},
            'compute': context.preferences.addons['cycles'].preferences.compute_device_type,
        }
        with internal_operators():
            result = bpy.ops.object.bake_textures('EXEC_DEFAULT', texture_resolution=self.bake_resolution,
                                                bake_pbr=False, add_material=True, flatten_for_refine=False,
                                                try_unwrap='smart', overlap_only=True)
        if result != {'RUNNING_MODAL'}:
            raise RuntimeError(f'Bake did not start: {result}')
        self.phase = 'bake'
        self.settle_ticks = 0

    def restore_bake(self, context):
        if self.bake_restore:
            saved = self.bake_restore
            self.scene.render.engine = saved['engine']
            self.scene.render.use_persistent_data = saved['persistent']
            for key, value in saved['cycles'].items():
                setattr(self.scene.cycles, key, value)
            context.preferences.addons['cycles'].preferences.compute_device_type = saved['compute']
            self.bake_restore = None

    def tick(self, context):
        from .generator import ComfyUIGenerate
        from ..core import state
        from ..utils import get_dir_path
        if self.scene and self.window.scene != self.scene:
            self.window.scene = self.scene
            self.stop_requested = True
        if self.phase in {'generation', 'bake'}:
            if ComfyUIGenerate._is_running or any(
                    op.bl_idname in {'OBJECT_OT_test_stable', 'OBJECT_OT_bake_textures'}
                    for win in context.window_manager.windows for op in win.modal_operators):
                return
            if self.phase == 'generation':
                if self.scene.sg_last_gen_error:
                    raise RuntimeError(self.scene.sg_last_gen_error_message or 'Generation failed or was cancelled')
                self.pass_index += 1
                if self.stop_requested:
                    self.entry['status'] = 'stopped'
                    self.phase = 'next'
                elif self.pass_index < len(CASES[self.index][1]):
                    self.apply_pass(context)
                else:
                    self.start_bake(context)
            else:
                # Bake restores Cycles settings in a deferred callback.
                self.settle_ticks += 1
                if self.settle_ticks < 2:
                    return
                self.restore_bake(context)
                verify_baked(self.bake_objects, get_dir_path(context, 'baked'))
                self.entry['objects'] = [obj.name for obj in self.bake_objects]
                self.entry['output'] = self.scene.get('_sg_regression_output', '')
                if '_sg_regression_output' in self.scene:
                    del self.scene['_sg_regression_output']
                self.entry['file'] = str(self.root / f'{self.index + 1:02d}.blend')
                save_scene(self.scene, self.entry['file'])
                self.entry['status'] = 'passed'
                self.phase = 'next'
            self.write_report('running')
            return
        if self.stop_requested:
            if state._pending_refreshes:
                return
            if self.entries and self.entries[-1]['status'] == 'running':
                self.entries[-1]['status'] = 'stopped'
            return self.finish(context)
        if self.phase == 'next':
            self.index += 1
            if self.index == len(CASES):
                return self.finish(context)
            label = CASES[self.index][0]
            self.entry = {'label': label, 'status': 'running', 'passes': []}
            self.entries.append(self.entry)
            self.write_report('running')
            if not self.detail_camera_name and any(preset in DETAIL_PRESETS for preset in CASES[self.index][1]):
                self.entry['status'] = 'skipped'
                self.entry['skip_reason'] = 'No detail camera assigned'
                self.write_report('running')
                return
            self.scene = copy_scene(context, self.template, f'Regression {self.index + 1:02d} {label}')
            park_detail_camera(self.scene)
            self.scene.use_fake_user = True
            self.entry['scene'] = self.scene.name
            self.pass_index = 0
            self.apply_pass(context)
        elif self.phase in {'refresh', 'ready'}:
            if state._pending_refreshes:
                if time.monotonic() > self.deadline:
                    # Never switch scenes with outstanding context-dependent refresh callbacks.
                    self.stop_requested = True
                    type(self).status = 'Model refresh timed out; waiting for cleanup'
                return
            if self.phase == 'refresh':
                self.configure_pass(context)
            else:
                self.ready_ticks += 1
                if self.ready_ticks >= 2:
                    self.start_generation(context)

    def finish(self, context):
        self.phase = 'finishing'
        if self.scene and '_sg_regression_output' in self.scene:
            del self.scene['_sg_regression_output']
        result = build_comparison(context, self.entries, 'StableGen Regression Comparison')
        result['regression_report'] = str(self.root / 'report.json')
        save_scene(result, self.root / 'comparison.blend')
        self.write_report('stopped' if self.stop_requested else 'complete')
        self.cleanup(context)
        passed = sum(entry['status'] == 'passed' for entry in self.entries)
        level = {'INFO'} if passed == len(CASES) else {'WARNING'}
        self.report(level, f'Regression: {passed}/{len(CASES)} passed. Results: {self.root}')
        return {'FINISHED'}

    def cleanup(self, context):
        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None
        type(self)._active = None
        type(self).status = ''

    def modal(self, context, event):
        if event.type == 'ESC':
            self.stop_requested = True
            return {'RUNNING_MODAL'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        # Blender's Event does not expose the originating timer. Generation and
        # baking also emit TIMER events, so throttle our state machine by time.
        now = time.monotonic()
        if now < self._next_tick:
            return {'PASS_THROUGH'}
        self._next_tick = now + .5
        try:
            with context.temp_override(window=self.window, area=self.area):
                result = self.tick(bpy.context)
            if result:
                return result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if self.phase == 'finishing' or self.index >= len(CASES) or not self.entries:
                self.window.scene = self.source
                self.cleanup(context)
                self.report({'ERROR'}, f'Regression could not finish: {exc}')
                return {'CANCELLED'}
            self.entry['status'] = 'failed'
            self.entry['error'] = str(exc)
            self.restore_bake(context)
            if self.scene and '_sg_regression_output' in self.scene:
                del self.scene['_sg_regression_output']
            self.phase = 'next'
            self.write_report('running')
        type(self).status = (('Stopping after current pass: ' if self.stop_requested else '')
                             + f'{self.index + 1}/{len(CASES)} {self.phase}: '
                             + (CASES[self.index][0] if 0 <= self.index < len(CASES) else 'comparison'))
        for area in self.window.screen.areas:
            area.tag_redraw()
        return {'PASS_THROUGH'}


class SG_OT_StopTexturingRegression(bpy.types.Operator):
    bl_idname = 'stablegen.stop_texturing_regression'
    bl_label = 'Stop After Current Pass'
    bl_description = 'Finish the active generation or bake, then build a partial comparison'

    @classmethod
    def poll(cls, context):
        return SG_OT_TexturingRegression._active is not None

    def execute(self, context):
        SG_OT_TexturingRegression._active.stop_requested = True
        return {'FINISHED'}


REGRESSION_CLASSES = [SG_OT_RefreshRegressionModels, SG_OT_TexturingRegression, SG_OT_StopTexturingRegression]
