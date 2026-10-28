"""Viewport-only camera display options and reversible surface diagnostics."""
import bpy

from ..utils import sg_modal_active
from . import prompt_overlay


def _redraw(self, context):
    from . import overlays
    overlays.display_settings_changed(context.scene)


SCOPE = [('ALL', 'All', 'Display all visible cameras'),
         ('SELECTED', 'Selected', 'Display only selected cameras'),
         ('NONE', 'None', 'Hide this overlay')]


class SGCameraDisplay(bpy.types.PropertyGroup):
    labels: bpy.props.EnumProperty(name='Labels', items=SCOPE, default='ALL', update=_redraw,
        description='Camera labels numbered in generation order. Click a saved-prompt icon to show or hide its editable view prompt')
    images: bpy.props.EnumProperty(name='Generated images', items=SCOPE, default='ALL', update=_redraw,
        description='Show existing projection images inside camera frames, preserving aspect ratio and alignment with the camera’s projection from either side. This viewport overlay does not change renders')
    crops: bpy.props.BoolProperty(name='Actual image frames', default=True, update=_redraw,
        description='Outline each camera’s actual rectangular projection frame inside Blender’s camera display')
    states: bpy.props.BoolProperty(name='Camera state indicators', default=True, update=_redraw,
        description='Orange marks selection; a camera icon marks the scene camera; a play icon marks the view currently generating')


class SG_OT_CameraDisplay(bpy.types.Operator):
    bl_idname = 'stablegen.camera_display'
    bl_label = 'Camera Display'
    bl_description = 'Configure numbered camera labels, generated image previews and rectangular image frames'

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=320)

    def execute(self, context):
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.sg_camera_display
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(settings, 'labels')
        layout.prop(settings, 'images')
        layout.prop(settings, 'crops')
        layout.prop(settings, 'states')
        layout.separator()
        row = layout.row(align=True)
        row.operator('stablegen.camera_surface_display', text='Visibility', icon='HIDE_OFF').mode = 'VISIBILITY'
        row.operator('stablegen.camera_surface_display', text='View weight', icon='SHADING_RENDERED').mode = 'WEIGHT'
        if context.scene.sg_studio_diagnostics.token:
            layout.operator('stablegen.camera_surface_display', text='Restore textures', icon='LOOP_BACK').mode = 'TEXTURES'


class SG_OT_CameraSurfaceDisplay(bpy.types.Operator):
    bl_idname = 'stablegen.camera_surface_display'
    bl_label = 'Camera Surface Diagnostic'
    bl_description = 'Inspect the selected camera using reversible diagnostic materials; requires Debug enabled in add-on preferences'
    bl_options = {'REGISTER', 'UNDO'}
    mode: bpy.props.EnumProperty(items=[('VISIBILITY', 'Visibility', 'Surfaces visible to this camera'),
                                       ('WEIGHT', 'View weight', 'Unnormalized projection visibility weight'),
                                       ('TEXTURES', 'Restore textures', 'Restore original meshes and materials')])

    @classmethod
    def poll(cls, context):
        if context.mode != 'OBJECT' or sg_modal_active(context):
            return False
        from ..core import ADDON_PKG
        prefs = context.preferences.addons.get(ADDON_PKG)
        if (not prefs or not prefs.preferences.enable_debug) and not context.scene.sg_studio_diagnostics.token:
            cls.poll_message_set('Enable Debug in StableGen add-on preferences to use surface diagnostics')
            return False
        return any(o.type == 'CAMERA' for o in context.scene.objects)

    def execute(self, context):
        from ..ui import studio_diagnostics as diagnostics
        state = context.scene.sg_studio_diagnostics
        if self.mode == 'TEXTURES':
            diagnostics.restore(context.scene)
        else:
            camera = context.active_object
            if camera is None or camera.type != 'CAMERA':
                camera = next((o for o in context.selected_objects if o.type == 'CAMERA'), context.scene.camera)
            if camera is None:
                self.report({'WARNING'}, 'Select a camera to inspect')
                return {'CANCELLED'}
            state.camera, state.mode = camera, self.mode
            try:
                diagnostics.apply(context)
            except Exception as exc:
                diagnostics.restore(context.scene)
                self.report({'WARNING'}, str(exc).split('\n')[0])
                return {'FINISHED'}  # Restoration must remain undoable.
        _redraw(None, context)
        return {'FINISHED'}


CLASSES = [SGCameraDisplay, SG_OT_CameraDisplay, SG_OT_CameraSurfaceDisplay,
           prompt_overlay.SG_GT_CameraPrompt, prompt_overlay.SG_GGT_CameraPrompts,
           prompt_overlay.SG_OT_CameraPromptInput]


def register():
    bpy.types.Scene.sg_camera_display = bpy.props.PointerProperty(type=SGCameraDisplay)
    prompt_overlay.register()
    from . import overlays
    overlays.register_overlays()


def unregister():
    from . import overlays
    overlays.unregister_overlays()
    prompt_overlay.unregister()
    del bpy.types.Scene.sg_camera_display
