"""Mesh tools that run on the ComfyUI server: LATO.2 low-poly remeshing through the ComfyUI-LATO2 node
(github.com/sakalond/ComfyUI-LATO2, installed by the ComfyUI setup's LATO.2 package).

The operator exports what it needs, runs one node in a background thread (WorkflowManager.generate_meshtools_file)
and imports the GLB that comes back as new objects, keeping the source.
"""

import os
import tempfile
import threading
import traceback
from datetime import datetime

import bpy  # pylint: disable=import-error

from ..core import get_addon_prefs
from ..utils import get_generation_dirs, sg_modal_active


def _save_dir(context):
    gen_dirs = get_generation_dirs(context)
    prefs = get_addon_prefs(context)
    folder = bpy.path.abspath(gen_dirs.get("revision", "") or (prefs.output_dir if prefs else "") or "")
    if not os.path.isabs(folder):           # no output folder set: not next to wherever Blender runs from
        folder = tempfile.gettempdir()
    folder = os.path.join(folder, "meshtools")
    os.makedirs(folder, exist_ok=True)
    return folder


def _selected_meshes(context):
    return [obj for obj in context.selected_objects if obj.type == 'MESH']


def lato2_missing():
    """(why the server has no LATO2Remesh node, an operator that helps, its label, its properties). A ComfyUI loads
    custom nodes only when it starts: one running since before LATO.2 was installed needs a restart."""
    from ..setup import packages, runtime
    p = runtime.prefs()
    if not runtime.local_mode(p):
        return ("LATO.2 isn't on the server: install it there with installer.py --packages lato2",
                "stablegen.comfy_copy_install_command", "Copy the install command", {"tags": "lato2"})
    folder = runtime.comfy_folder(p)
    state = packages.row_state(("lato2",), runtime.report(folder))["state"] if folder is not None else "missing"
    if state != "installed":
        return ("LATO.2 isn't installed in your ComfyUI", "stablegen.open_setup", "Install it in the ComfyUI setup", {})
    if runtime.comfy().owned:
        return ("LATO.2 is installed; ComfyUI started before it, so restart ComfyUI to load it",
                "stablegen.comfy_restart", "Restart ComfyUI", {})
    return ("LATO.2 is installed; your ComfyUI started before it, so restart it to load LATO.2 (StableGen didn't "
            "start it, so it won't restart it)", "stablegen.check_server_status", "Check again", {})


def _export_glb(context, objects, path, materials=True, geometry_only=False):
    """Export the objects (world space, modifiers applied) to one GLB, then restore the selection. geometry_only
    leaves out normals, UVs and colors, so the exporter splits no vertex at a seam or sharp edge and the mesh stays
    welded (edge-collapse decimation would open cracks there)."""
    previous_active = context.view_layer.objects.active
    previous = list(context.selected_objects)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    context.view_layer.objects.active = objects[0]
    options = dict(
        filepath=path, export_format='GLB', use_selection=True, export_apply=True,
        export_materials='EXPORT' if materials else 'NONE', export_animations=False,
        export_skins=False, export_morph=False,
    )
    if geometry_only:
        available = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
        options.update((name, value) for name, value in (
            ('export_normals', False), ('export_texcoords', False), ('export_vertex_color', 'NONE'),
            ('export_colors', False), ('export_attributes', False)) if name in available)
    try:
        bpy.ops.export_scene.gltf(**options)
    finally:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in previous:
            if context.scene.objects.get(obj.name):
                obj.select_set(True)
        if previous_active and context.scene.objects.get(previous_active.name):
            context.view_layer.objects.active = previous_active
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError("the GLB export wrote nothing")


def _import_glb(context, path, collection_name):
    """Import a GLB into a new collection; returns the imported objects."""
    collection = bpy.data.collections.new(collection_name)
    context.scene.collection.children.link(collection)
    layer_collection = context.view_layer.layer_collection.children[collection.name]
    previous = context.view_layer.active_layer_collection
    context.view_layer.active_layer_collection = layer_collection
    try:
        bpy.ops.import_scene.gltf(filepath=path)
    finally:
        context.view_layer.active_layer_collection = previous
    imported = list(collection.all_objects)
    for obj in imported:
        obj["sg_generated"] = True
    return imported


class _MeshToolOperator(bpy.types.Operator):
    """Shared modal flow: prepare inputs, run the node in a thread, import the result."""

    _is_running = False
    _cancelled = False
    _active_ws = None
    _PHASE_WEIGHTS = (10.0, 80.0, 10.0)
    _total_phases = 3
    _available_prop = ""
    _node_label = ""
    _class_type = ""

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return True
        prefs = get_addon_prefs(context)
        if prefs is None or not prefs.server_address or not prefs.server_online:
            cls.poll_message_set("ComfyUI server is not connected")
            return False
        if not getattr(context.scene, cls._available_prop, False):
            cls.poll_message_set(lato2_missing()[0])
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return cls._poll_inputs(context)

    @classmethod
    def _poll_inputs(cls, context):
        return True

    def _set_pipeline_phase(self, phase, label, percent, detail=None):
        phase = max(1, min(int(phase), self._total_phases))
        percent = max(0.0, min(float(percent), 100.0))
        overall = sum(self._PHASE_WEIGHTS[:phase - 1]) + self._PHASE_WEIGHTS[phase - 1] * percent / 100.0
        self._overall_progress = max(self._overall_progress, overall)
        self._overall_stage = f"{self._node_label}: {label}"
        self._phase_progress = percent
        self._phase_stage = detail or label
        self._progress = self._overall_progress
        self._stage = self._phase_stage

    def _prepare(self, context, work_dir):
        """(inputs, uploads) for the node; raises RuntimeError with a message for the user."""
        raise NotImplementedError

    def _finish(self, context, glb_path):
        """Import the result; returns a short report."""
        raise NotImplementedError

    def execute(self, context):
        cls = type(self)
        if cls._is_running:
            cls._cancelled = True
            ws = cls._active_ws
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            self.report({'INFO'}, f"Cancelling {self._node_label}...")
            return {'FINISHED'}
        try:
            self._work_dir = _save_dir(context)
            inputs, uploads = self._prepare(context, self._work_dir)
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, f"{self._node_label}: {exc}")
            return {'CANCELLED'}

        cls._is_running = True
        cls._cancelled = False
        self._error = None
        self._result = None
        self._overall_progress = 0.0
        self._overall_stage = f"{self._node_label}: Queued"
        self._phase_progress = 0.0
        self._phase_stage = "Preparing"
        self._progress = 0.0
        self._stage = "Preparing"
        from ..workflows import WorkflowManager
        self.workflow_manager = WorkflowManager(self)
        self._thread = threading.Thread(target=self._run, args=(context, inputs, uploads), daemon=True)
        self._thread.start()
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        self.report({'INFO'}, f"{self._node_label} started")
        return {'RUNNING_MODAL'}

    def _run(self, context, inputs, uploads):
        try:
            result = self.workflow_manager.generate_meshtools_file(
                context, self._class_type, inputs, uploads=uploads, label=self._node_label)
            if isinstance(result, dict) and "error" in result:
                self._error = result["error"]
            else:
                self._result = result
        except Exception as exc:
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
        cls = type(self)
        was_cancelled = cls._cancelled
        cls._is_running = False
        cls._cancelled = False
        cls._active_ws = None
        if was_cancelled or self._error == "cancelled":
            self.report({'INFO'}, f"{self._node_label} cancelled")
            return {'FINISHED'}
        if self._error or not self._result:
            self.report({'ERROR'}, f"{self._node_label} failed: {self._error or 'no GLB was returned'}")
            return {'CANCELLED'}
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            glb_path = os.path.join(self._work_dir, f"{self._class_type}_{stamp}.glb")
            with open(glb_path, 'wb') as handle:
                handle.write(self._result)
            message = self._finish(context, glb_path)
            self.report({'INFO'}, message)
            return {'FINISHED'}
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, f"{self._node_label}: importing the result failed: {exc}")
            return {'CANCELLED'}


class StableGenLato2Remesh(_MeshToolOperator):
    """Re-mesh the selected mesh into a low-poly mesh with generated topology (LATO.2 on the ComfyUI server)."""

    bl_idname = "object.stablegen_lato2_remesh"
    bl_label = "Low-Poly Remesh (LATO.2)"
    bl_description = (
        "Export the selected mesh, rebuild it on the ComfyUI server as a low-poly mesh with generated topology "
        "(LATO.2, 200 to 5000 vertices) and import the result next to the source"
    )
    bl_options = {'REGISTER', 'UNDO'}
    _is_running = False
    _cancelled = False
    _active_ws = None
    _available_prop = "meshtools_lato2_available"
    _node_label = "LATO.2 Remesh"
    _class_type = "LATO2Remesh"

    @classmethod
    def _poll_inputs(cls, context):
        if not _selected_meshes(context):
            cls.poll_message_set("Select a mesh object")
            return False
        return True

    def _prepare(self, context, work_dir):
        scene = context.scene
        self._sources = _selected_meshes(context)
        path = os.path.join(work_dir, f"lato2_source_{datetime.now():%H%M%S}.glb")
        _export_glb(context, self._sources, path, materials=False)
        inputs = {
            "vertex_count": int(scene.meshtools_lato2_vertices),
            "seed": int(scene.meshtools_lato2_seed),
            "filename_prefix": "lato2",
        }
        return inputs, {"input_path": path}

    def _finish(self, context, glb_path):
        name = self._sources[0].name
        imported = _import_glb(context, glb_path, f"{name}_LATO2")
        meshes = [obj for obj in imported if obj.type == 'MESH']
        for obj in meshes:
            obj.name = f"{name}_LATO2"
            obj.data.materials.clear()
        if context.scene.meshtools_hide_source:
            for obj in self._sources:
                obj.hide_set(True)
        faces = sum(len(obj.data.polygons) for obj in meshes)
        verts = sum(len(obj.data.vertices) for obj in meshes)
        return f"LATO.2: {verts} vertices, {faces} faces"
