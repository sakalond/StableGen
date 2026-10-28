"""Presentation metadata for Studio actions; operator behavior stays native."""

OPERATOR_ICONS = {
    'object.bake_textures': 'grid',
    'object.switch_material': 'material',
    'object.export_game_engine': 'download',
    'object.stablegen_preview_quantization': 'eye',
    'object.stablegen_preview_sliced': 'layers',
    'object.stablegen_clear_quantization_preview': 'x',
    'object.stablegen_preview_sliced_cancel': 'stop',
    'object.export_3mf_cancel': 'stop',
    'stablegen.prepare_mask_paint': 'settings',
    'stablegen.paint_edit_mask': 'settings',
    'stablegen.finish_edit_mask': 'check',
    'stablegen.clear_edit_mask': 'x',
    'stablegen.check_server_status': 'reset',
    'stablegen.refresh_checkpoint_list': 'reset',
    'stablegen.refresh_lora_list': 'reset',
    'stablegen.reset_qwen_prompt': 'reset',
    'stablegen.reset_texturing_experiments': 'reset',
    'stablegen.apply_animation_to_rig': 'play',
    'stablegen.apply_rig_pose_preset': 'rig',
    'stablegen.characterize_rig': 'rig',
    'stablegen.edit_rig_mapping': 'rig',
    'stablegen.match_source_reference_pose': 'rig',
    'stablegen.set_retarget_reference_pose': 'rig',
    'stablegen.apply_autorig_bone_display': 'rig',
    'stablegen.toggle_rig_mapping_overlay': 'eye',
    'stablegen.toggle_collision_proxy_overlay': 'eye',
    'stablegen.toggle_retarget_source_preview': 'eye',
    'stablegen.sync_camera_order': 'reset',
    'stablegen.queue_move_up': 'up',
    'stablegen.queue_move_down': 'down',
    'stablegen.queue_open_result': 'folder',
    'stablegen.queue_invalidate': 'reset',
    'stablegen.queue_process': 'play',
    'stablegen.queue_clear': 'x',
}
NATIVE_ICONS = {
    'ADD': 'plus', 'REMOVE': 'minus', 'X': 'x', 'CANCEL': 'stop',
    'FILE_REFRESH': 'reset', 'EXPORT': 'download', 'IMPORT': 'folder',
    'FILE_FOLDER': 'folder', 'FILEBROWSER': 'folder', 'PLAY': 'play',
    'TRIA_UP': 'up', 'TRIA_DOWN': 'down', 'COLOR': 'material',
    'MATERIAL': 'material', 'TEXTURE': 'layers', 'IMAGE_DATA': 'eye',
    'CAMERA_DATA': 'camera', 'ARMATURE_DATA': 'rig', 'OUTLINER_OB_ARMATURE': 'rig',
    'BONE_DATA': 'rig', 'MODIFIER': 'modifier', 'MOD_DECIM': 'layers',
    'MOD_REMESH': 'modifier', 'SETTINGS': 'settings', 'PRESET': 'settings',
    'CHECKMARK': 'check', 'HIDE_OFF': 'eye', 'LIGHT': 'light',
}
SELECTION_PATHS = {
    'scene.stablegen_print_palette_index': 'Filament',
    'scene.lora_units_index': 'Adapter',
}
PRINT_PREVIEW = {
    'object.stablegen_preview_quantization': 'Colors',
    'object.stablegen_preview_sliced': 'Slices',
    'object.stablegen_clear_quantization_preview': 'Clear',
}
PRINT_EXPORT = {'object.export_stl': 'STL', 'object.export_3mf': '3MF'}


def icon(control):
    """Use a relevant native/semantic icon, or text alone for unknown commands."""
    return OPERATOR_ICONS.get(control.args[0], NATIVE_ICONS.get(control.kwargs.get('icon')))


def selection(control):
    if control.method == 'operator' and control.args[0] == 'wm.context_set_int':
        path = control.properties.get('data_path', '')
        if path in SELECTION_PATHS:
            return path.removeprefix('scene.'), control.properties.get('value', 0), SELECTION_PATHS[path]
    return None


def print_group(controls, section):
    if section == 'print' and controls:
        names = {c.args[0] for c in controls}
        if names <= PRINT_PREVIEW.keys():
            return 'Preview', PRINT_PREVIEW
        if names <= PRINT_EXPORT.keys():
            return 'Export', PRINT_EXPORT
    return '', {}
