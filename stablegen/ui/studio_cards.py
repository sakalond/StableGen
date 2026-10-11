"""Card presentation for Studio, using the captured native controls.

The native panel decides which controls exist and whether they are enabled.
This module partitions that capture without replacing RNA bindings, command
arguments or native-editor indices. Unclassified controls get a fallback card,
so adding a setting to the source panel cannot silently hide it in Studio.
"""
from dataclasses import dataclass

import bpy

from . import studio_model as model, studio_hierarchy as hierarchy
from .presentation import Control, Layout, STRUCTURAL_PROPERTIES
from .studio_text import wrap
from .studio_widgets import FONT_SIZE, CONTROL_HEIGHT, CONTROL_GAP

GAP = 8
PAD = 12
# Content width at which cards and action rows sit side by side.
SIDE_BY_SIDE_WIDTH = 440
RESOLUTION_KEYS = {'auto_rescale', 'auto_rescale_target_mp', 'qwen_rescale_alignment'}


@dataclass(frozen=True)
class Spec:
    key: str
    title: str
    icon: str = 'settings'
    keys: tuple = ()
    feature: str = ''
    rows: tuple = ()
    description: str = ''
    beside: str = ''
    dividers: tuple = ()
    parent: str = ''


def feature(key, title, icon, **kwargs):
    """A feature owns the controls that are already conditional in the capture."""
    return Spec(key, title, icon, (key,), feature=key, **kwargs)


SCHEMAS = {
    'show_core_settings': (
        Spec('sampling', 'Sampling', 'curve', ('steps', 'cfg', 'sampler', 'scheduler',
             'comfyui_negative_prompt', 'clip_skip'), rows=(('steps', 'cfg'), ('sampler', 'scheduler'))),
        Spec('seed', 'Seed & repeatability', 'reset', ('seed', 'control_after_generate')),
        feature('qwen_image_21_native_rgba', 'Native RGBA output', 'layers',
                description='Preserve transparency in generated images.'),
    ),
    'resolution': (feature('auto_rescale', 'Fit camera resolution', 'expand'),),
    'show_scene_understanding_settings': (
        Spec('angle', 'Angle rejection', 'shield', ('discard_factor', 'discard_factor_generation_only'), beside='angle'),
        Spec('exponent', 'View preference', 'curve', ('weight_exponent', 'weight_exponent_generation_only'), beside='angle'),
        feature('early_priority', 'Prioritize earlier views', 'arrow'),
        Spec('visibility', 'Visibility & source quality', 'eye', ('sg_allow_backface_projection',
             'sg_rework_acquisition_quality', 'bake_visibility_weights', 'sg_texturing_frame'),
             rows=(('sg_allow_backface_projection',), ('sg_rework_acquisition_quality',),
                   ('bake_visibility_weights',), ('sg_texturing_frame',))),
        feature('sg_rework_projection_guard', 'Depth / part projection guard', 'shield',
                rows=(('sg_silhouette_margin',), ('sg_guard_directional',),
                      ('sg_silhouette_depth', 'sg_guard_coverage_rescue'),
                      ('sg_silhouette_feather',), ('sg_projection_guard_scale',), ('sg_guard_feather_method',)),
                dividers=(('sg_silhouette_feather', 'Boundary transition'),)),
    ),
    'show_image_guidance_settings': (
        Spec('structure', 'Structure & reference colors', 'layers', ('qwen_guidance_map_type',
             'qwen_timestep_zero_ref', 'qwen_refine_use_prev_ref', 'qwen_refine_use_depth')),
        Spec('style', 'Style reference', 'eye', ('qwen_use_external_style_image',
             'qwen_use_trellis2_style', 'sequential_ipadapter')),
        Spec('context', 'Sequential context', 'shield', ('qwen_context_render_mode', 'qwen_voronoi_mode')),
        Spec('wording', 'Guidance wording', 'labels', ('qwen_prompt_template', 'qwen_use_custom_prompts',
             'op:stablegen.reset_qwen_prompt')),
        feature('use_ipadapter', 'External image reference', 'eye'),
        feature('use_flux_lora', 'FLUX depth LoRA', 'layers'),
    ),
    'show_mode_specific_settings': (
        Spec('edit', 'Edit strength', 'curve', ('denoise',)),
        Spec('visibility', 'Visibility map', 'eye', ('sequential_smooth', 'weight_exponent_mask'),
             rows=(('sequential_factor_smooth', 'sequential_factor_smooth_2'),)),
        feature('sequential_ipadapter', 'Previous-view reference', 'eye'),
        feature('refine_angle_ramp_active', 'Angle-based blending', 'curve'),
        feature('visibility_vignette', 'Image-frame fade', 'layers'),
        feature('refine_edge_feather_projection', 'Projection silhouette fade', 'curve'),
        feature('view_blend_use_color_match', 'Match viewport colors', 'material'),
        feature('refine_images', 'Refine generated views', 'curve'),
        Spec('uv', 'UV inpainting', 'layers', ('allow_modify_existing_textures', 'ask_object_prompts')),
    ),
    'show_masking_inpainting_settings': (
        feature('differential_diffusion', 'Differential diffusion', 'curve'),
        Spec('mask', 'Mask preparation', 'shield', ('mask_blocky', 'blur_mask', 'grow_mask_by')),
    ),
    'show_output_material_settings': (
        # PBR map ownership is explicit: material assignment stays editable even
        # when the albedo map is off, as it does in the native panel.
        Spec('pbr', 'PBR map extraction', 'material', ('pbr_decomposition', 'op:stablegen.check_server_status'), feature='pbr_decomposition',
             description='Extract material maps from generated views.'),
        Spec('albedo', 'Albedo', 'material', ('pbr_map_albedo', 'pbr_albedo_source', 'pbr_delight_strength',
             'pbr_albedo_auto_saturation', 'pbr_albedo_saturation_mode', 'pbr_replace_color_with_albedo'),
             feature='pbr_map_albedo', description='Base color', parent='pbr',
             rows=(('pbr_albedo_source',), ('pbr_albedo_auto_saturation',), ('pbr_replace_color_with_albedo',))),
        Spec('roughness', 'Roughness', 'curve', ('pbr_map_roughness',), feature='pbr_map_roughness',
             description='Reflection softness', beside='material', parent='pbr'),
        Spec('metallic', 'Metallic', 'material', ('pbr_map_metallic',), feature='pbr_map_metallic',
             description='Metal or dielectric', beside='material', parent='pbr'),
        Spec('normal', 'Normal', 'layers', ('pbr_map_normal', 'pbr_normal_strength'), feature='pbr_map_normal',
             description='Surface detail', beside='detail', parent='pbr'),
        Spec('height', 'Height', 'modifier', ('pbr_map_height', 'pbr_height_scale'), feature='pbr_map_height',
             description='Displacement', beside='detail', parent='pbr'),
        Spec('ao', 'Ambient occlusion', 'eye', ('pbr_map_ao', 'pbr_ao_samples', 'pbr_ao_distance'),
             feature='pbr_map_ao', description='Geometry-based AO shader.', parent='pbr',
             rows=(('pbr_ao_samples', 'pbr_ao_distance'),)),
        Spec('emission', 'Emission', 'light', ('pbr_map_emission', 'pbr_emission_method',
             'pbr_emission_threshold', 'pbr_emission_strength', 'pbr_emission_saturation_min',
             'pbr_emission_value_min', 'pbr_emission_bloom'), feature='pbr_map_emission',
             description='Self-illumination', parent='pbr',
             rows=(('pbr_emission_saturation_min', 'pbr_emission_value_min'),)),
        Spec('processing', 'Shared processing', 'settings', ('pbr_use_native_resolution',
             'pbr_processing_resolution', 'pbr_tiling', 'pbr_tile_grid', 'pbr_tile_superres',
             'pbr_tile_albedo', 'pbr_tile_material', 'pbr_tile_normal', 'pbr_tile_height',
             'pbr_tile_emission', 'pbr_denoise_steps', 'pbr_ensemble_size'), parent='pbr',
             rows=(('pbr_use_native_resolution',), ('pbr_tiling',), ('pbr_denoise_steps', 'pbr_ensemble_size'))),
        Spec('material', 'Material & lighting', 'material', ('fallback_color', 'overwrite_material', 'pbr_auto_lighting')),
    ),
    'show_trellis2_mesh_settings': (
        Spec('mesh_seed', 'Mesh repeatability', 'reset', ('trellis2_seed',)),
        Spec('shape', 'Shape generation', 'layers', ('trellis2_ss_guidance', 'trellis2_ss_steps',
             'trellis2_shape_guidance', 'trellis2_shape_steps', 'trellis2_max_tokens')),
        feature('trellis2_post_processing_enabled', 'Mesh post-processing', 'modifier'),
        Spec('model', 'Mesh model', 'settings', ('trellis2_resolution', 'trellis2_precision', 'trellis2_attn_backend')),
        Spec('conditioning', 'Image conditioning', 'eye', ('trellis2_background_color', 'trellis2_bg_removal', 'trellis2_shade_mode')),
    ),
    'show_trellis2_autorig_settings': (
        Spec('autorig', 'Auto-rig', 'rig', ('trellis2_autorig_enabled', 'trellis2_autorig_use_transfer',
             'trellis2_autorig_use_postprocess', 'trellis2_autorig_use_skeleton',
             'trellis2_autorig_cleanup_cross_limb_weights', 'trellis2_autorig_garment_mode',
             'op:stablegen.check_server_status'), feature='trellis2_autorig_enabled'),
        Spec('mapping', 'Rig mapping & display', 'rig', ('trellis2_autorig_bone_heuristic',
             'trellis2_autorig_bone_display_type', 'op:stablegen.characterize_rig',
             'op:stablegen.edit_rig_mapping', 'op:stablegen.toggle_rig_mapping_overlay',
             'op:stablegen.toggle_collision_proxy_overlay', 'op:stablegen.apply_autorig_bone_display')),
        Spec('pose', 'Reference poses & animation', 'play', ('op:stablegen.apply_rig_pose_preset',
             'op:stablegen.match_source_reference_pose', 'op:stablegen.set_retarget_reference_pose',
             'op:stablegen.apply_animation_to_rig', 'op:stablegen.toggle_retarget_source_preview')),
        Spec('decode', 'Rig sampling', 'curve', ('trellis2_autorig_top_k', 'trellis2_autorig_top_p',
             'trellis2_autorig_temperature', 'trellis2_autorig_repetition_penalty', 'trellis2_autorig_num_beams')),
        Spec('checkpoint', 'Rig model', 'settings', ('trellis2_autorig_model_ckpt', 'trellis2_autorig_hf_path')),
    ),
    'show_trellis2_texture_settings': (
        Spec('texture', 'Native texture generation', 'layers', ('trellis2_tex_guidance',
             'trellis2_tex_steps', 'trellis2_texture_size', 'trellis2_auto_lighting')),
    ),
    'show_trellis2_camera_settings': (
        Spec('placement', 'View placement', 'camera', ('trellis2_import_scale', 'trellis2_placement_mode',
             'trellis2_camera_count', 'trellis2_coverage_target', 'trellis2_max_auto_cameras', 'trellis2_fan_angle')),
        Spec('coverage', 'Coverage & framing', 'eye', ('trellis2_auto_aspect', 'trellis2_occlusion_mode',
             'trellis2_exclude_bottom', 'trellis2_clamp_elevation')),
        Spec('behavior', 'Camera lifecycle', 'camera', ('trellis2_auto_prompts',
             'trellis2_consider_existing', 'trellis2_delete_cameras')),
    ),
    'show_camera_options': (
        Spec('order', 'Generation order', 'camera', ('sg_use_custom_camera_order',
             'op:stablegen.apply_camera_order_preset', 'op:stablegen.sync_camera_order',
             'op:stablegen.move_camera_order', 'native:template_list')),
    ),
    'show_texturing_experiments': (
        Spec('resolution', 'Resolution & display', 'expand', ('sg_rework_aligned_resolution',
             'sg_rework_full_resolution_scale', 'sg_rework_final_display')),
        feature('sg_auto_view_sequence', 'Automatic view sequence', 'camera'),
        Spec('context', 'Experimental Qwen context', 'shield', ('qwen_global_style_vlm_only',
             'qwen_context_vlm_only', 'qwen_latent_locked_context', 'qwen_regularize_context_mask',
             'qwen_use_projection_write_masks', 'qwen_reject_residual_fallback')),
    ),
    'queue': (
        Spec('scenes', 'Queued scenes', 'layers', ('op:stablegen.queue_move_up',
             'op:stablegen.queue_move_down', 'op:stablegen.queue_remove', 'op:stablegen.queue_add',
             'op:stablegen.queue_open_result', 'op:stablegen.queue_invalidate',
             'op:stablegen.queue_clear', 'op:stablegen.queue_process', 'native:template_list')),
        feature('sg_queue_gif_export', 'Orbit export', 'camera'),
    ),
    'print': (
        Spec('palette', 'Filament palette', 'material', ('stablegen_print_preset', 'name', 'color',
             'op:wm.context_set_int', 'op:stablegen.add_palette_color', 'op:stablegen.remove_palette_color')),
        feature('stablegen_print_dithered', 'Full-spectrum dithering', 'grid'),
        Spec('cleanup', 'Color cleanup', 'curve', ('stablegen_print_smoothing', 'stablegen_print_chroma_threshold')),
        feature('stablegen_print_make_solid', 'Solid print preparation', 'modifier'),
        feature('stablegen_print_show_advanced', 'Advanced solver', 'settings'),
        Spec('export', 'Print preview & export', 'download', ('op:object.stablegen_preview_quantization',
             'op:object.stablegen_preview_sliced', 'op:object.stablegen_clear_quantization_preview',
             'op:object.export_stl', 'op:object.export_3mf', 'op:object.export_textured_print_obj')),
    ),
}

# Captions can be shortened in Studio without changing their tooltips or values.
LABELS = {
    'cfg': 'Guidance (CFG)',
    'control_after_generate': 'After generation',
    'qwen_guidance_map_type': 'Guidance image',
    'qwen_timestep_zero_ref': 'Timestep-zero references',
    'qwen_custom_prompt_initial': 'Initial view',
    'qwen_custom_prompt_seq_none': 'Subsequent views',
    'qwen_custom_prompt_seq_replace': 'Subsequent views',
    'qwen_custom_prompt_seq_additional': 'Subsequent views',
    'discard_factor_generation_only': 'Resets',
    'weight_exponent_generation_only': 'Resets',
    'weight_exponent_after_generation': 'Reset exponent',
    'auto_rescale_target_mp': 'Target megapixels',
    'sg_texturing_frame': 'Texturing frame',
    'bake_visibility_weights': 'Bake transform-stable visibility',
    'sg_allow_backface_projection': 'Allow backface projection',
    'sg_rework_acquisition_quality': 'Weight by view quality',
    'sg_guard_coverage_rescue': 'Coverage rescue',
    'sg_guard_directional': 'Depth-aware margins',
    'sg_silhouette_depth': 'Depth threshold',
    'sg_projection_guard_scale': 'Guard render scale',
    'sg_guard_feather_method': 'Guard feather method',
    'sg_silhouette_feather': 'Guard feather width (px)',
    'discard_factor_after_generation': 'Reset angle (°)',
    'trellis2_resolution': 'Mesh generation resolution',
    'trellis2_background_color': 'Background',
    'trellis2_bg_removal': 'Background removal',
    'pbr_albedo_source': 'Extraction model',
    'pbr_delight_strength': 'Highlight removal',
    'pbr_albedo_auto_saturation': 'Correct saturation',
    'pbr_albedo_saturation_mode': 'Correction across views',
    'pbr_replace_color_with_albedo': 'Use as material base color',
    'pbr_use_native_resolution': 'Use source size for inference',
    'pbr_tiling': 'Tiling',
    'pbr_tile_superres': 'Keep tiled output resolution',
    'pbr_height_scale': 'Material displacement scale',
    'pbr_emission_method': 'Detection method',
    'pbr_emission_threshold': 'Emission detection cutoff',
    'pbr_emission_saturation_min': 'Minimum saturation',
    'pbr_emission_value_min': 'Minimum brightness',
    'pbr_ao_samples': 'AO samples',
    'pbr_ao_distance': 'Ray distance (BU)',
    'pbr_normal_strength': 'Normal strength',
    'pbr_auto_lighting': 'Add studio lighting after projection',
    'early_priority': 'Prioritize earlier views',
    'sequential_factor': 'Full-visibility tolerance',
    'visibility_vignette_softness': 'Frame fade exponent',
    'ipadapter_start': 'Start fraction',
    'start_percent': 'Start fraction',
    'ipadapter_end': 'End fraction',
    'end_percent': 'End fraction',
    'generation_method': 'Texture workflow',
    'qwen_generation_method': 'Texture workflow',
    'use_camera_prompts': 'Use view prompts',
    'denoise': 'Edit strength',
    'sg_masked_strength': 'Edit strength',
    'scheduler': 'Noise schedule',
    'refine_scheduler': 'Noise schedule',
    'refine_cfg': 'Prompt guidance (CFG)',
    'steps': 'Sampling steps',
    'refine_steps': 'Sampling steps',
    'clip_skip': 'CLIP layer skip',
    'auto_rescale': 'Fit camera resolution',
    'weight_exponent': 'View weight exponent',
    'weight_exponent_mask': 'Apply view exponent to visibility mask',
    'sg_guard_foreground_margin': 'Foreground margin (px)',
    'sequential_smooth': 'Smooth previous-view coverage',
    'sequential_factor_smooth': 'Coverage low point',
    'sequential_factor_smooth_2': 'Coverage high point',
    'allow_modify_existing_textures': 'Allow updating textured areas',
    'sequential_ipadapter': 'Use previous view as reference',
    'sequential_ipadapter_mode': 'Reference source',
    'sequential_desaturate_factor': 'Reference desaturation',
    'sequential_contrast_factor': 'Reference contrast reduction',
    'qwen_use_confidence_context': 'Protect reliable context',
    'qwen_trust_threshold': 'Minimum trusted view quality',
    'qwen_guidance_fallback_color': 'Missing-context color',
    'qwen_guidance_background_color': 'Image background',
    'qwen_voronoi_mode': 'Hard view segmentation',
    'qwen_context_fallback_dilation': 'Legacy context expansion (px)',
    'qwen_context_cleanup': 'Legacy fallback cleanup',
    'qwen_context_cleanup_hue_tolerance': 'Hue tolerance (°)',
    'qwen_context_cleanup_value_adjust': 'Brightness adjustment',
    'blur_mask_radius': 'Mask blur radius (px)',
    'blur_mask_sigma': 'Mask blur sigma',
    'grow_mask_by': 'Mask expansion (px)',
    'mask_blocky': 'Align mask to latent grid',
    'differential_noise': 'Use latent noise mask',
    'visibility_vignette': 'Image-frame fade',
    'visibility_vignette_width': 'Frame fade width',
    'visibility_vignette_blur': 'Blur frame mask',
    'refine_edge_feather_projection': 'Projection silhouette fade',
    'refine_edge_feather_width': 'Silhouette fade width (px)',
    'refine_edge_feather_softness': 'Silhouette ramp smoothing',
    'refine_angle_ramp_pos_0': 'Angle ramp low point',
    'refine_angle_ramp_pos_1': 'Angle ramp high point',
    'refine_feather_ramp_pos_0': 'Frame ramp low point',
    'refine_feather_ramp_pos_1': 'Frame ramp high point',
    'pbr_decomposition': 'PBR map extraction',
    'pbr_tile_grid': 'Tile grid',
    'pbr_processing_resolution': 'Inference resolution (px)',
    'pbr_denoise_steps': 'Inference steps',
    'pbr_ensemble_size': 'Predictions per map',
    'pbr_emission_bloom': 'Detection blur (px)',
    'trellis2_ss_guidance': 'Structure guidance',
    'trellis2_ss_steps': 'Structure steps',
    'trellis2_shape_guidance': 'Shape guidance',
    'trellis2_shape_steps': 'Shape steps',
    'trellis2_tex_guidance': 'Texture guidance',
    'trellis2_tex_steps': 'Texture steps',
    'trellis2_max_tokens': 'Sparse voxel budget',
    'trellis2_texture_size': 'Texture size (px)',
    'trellis2_coverage_target': 'Camera coverage target',
    'trellis2_autorig_top_k': 'Rig candidate limit (top-k)',
    'trellis2_autorig_top_p': 'Rig probability cutoff (top-p)',
    'trellis2_autorig_temperature': 'Rig sampling temperature',
    'trellis2_autorig_num_beams': 'Rig beam count',
    'trellis2_autorig_hf_path': 'Hugging Face model path',
    'trellis2_autorig_model_ckpt': 'Rig checkpoint path',
    'stablegen_print_solver_steps': 'Solver iterations',
    'stablegen_print_solver_init': 'Solver initialization',
    'stablegen_print_solver_formula': 'Color mixing model',
    'stablegen_print_chroma_threshold': 'Saturation cutoff',
    'stablegen_print_smoothing': 'Island cleanup passes',
    'stablegen_print_single_filament_internal': 'Single filament for unseen faces',
    'sg_rework_aligned_resolution': 'Joint resolution fitting',
    'sg_rework_full_resolution_scale': 'Use full guide resolution',
    'sg_rework_final_display': 'Rebuild final projection',
    'sg_auto_view_sequence': 'Automatic view sequence',
    'qwen_latent_locked_context': 'Lock reliable context during sampling',
    'qwen_regularize_context_mask': 'Clean repair-mask boundaries',
    'qwen_mask_min_island_area': 'Minimum trusted island (px²)',
    'qwen_mask_min_hole_area': 'Minimum repair hole (px²)',
    'qwen_mask_min_strip_width': 'Minimum trusted strip (px)',
    'qwen_mask_uncertainty_dilation': 'Repair expansion (px)',
    'refine_images': 'Refine generated views',
}
DECORATIVE = {
    'Maps to Extract:', 'PBR Decomposition', 'Post-Processing:', 'Shape Generation:',
    'Model:', 'Conditioning:', 'Misc:', 'LoRA Units', 'ControlNet Units',
    'Qwen Local Edit Parameters', 'Qwen Refine Parameters', 'Grid Mode Parameters',
    'Separate Mode Parameters', 'Refine Mode Parameters', 'Local Edit Parameters',
    'Sequential Mode Parameters', 'UV Inpainting Parameters', 'Masked Edit Parameters',
    'Multi-Color 3D Print Palette',
}


def property_name(control):
    return control.args[1] if control.method == 'prop' and len(control.args) > 1 else ''


def token(control):
    if control.method in {'operator', 'operator_menu_enum'}:
        return 'op:' + control.args[0]
    return property_name(control) or 'native:' + control.method


def family(keys):
    result = set(keys)
    pending = list(keys)
    while pending:
        for child in hierarchy.FAMILIES.get(pending.pop(), ()):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def select(node, ids):
    """Keep original controls and all inherited layout states, including alerts."""
    if isinstance(node, Control):
        return node if id(node) in ids else None
    result = Layout(node.method, node.args, dict(node.kwargs))
    result.attributes.update(node.attributes)
    result.children.extend(c for child in node.children if (c := select(child, ids)) is not None)
    return result if result.children else None


@dataclass
class Card:
    spec: Spec
    content: Layout


def partition(source, specs, omit=(), title='Additional settings', *, _tokens=None):
    """Assign each source control to exactly one card; never alter the capture."""
    controls = list(source.controls())
    excluded = {id(c) for c in controls if property_name(c) in omit or token(c) in omit
                or property_name(c) in STRUCTURAL_PROPERTIES}
    explicit = {key: spec.key for spec in specs for key in spec.keys}
    expanded = {key: spec.key for spec in reversed(specs) for key in family(spec.keys)}
    expanded.update(explicit)
    def key(c):
        override = (_tokens or {}).get(id(c))
        return override if override in expanded else token(c)
    assignments = {id(c): expanded[key(c)] for c in controls if key(c) in expanded and id(c) not in excluded}
    fallback = '__other__'

    def assign(node, inherited=None):
        groups = [{assignments[id(c)] for c in child.controls() if id(c) in assignments}
                  if isinstance(child, Layout) else {assignments[id(child)]} if id(child) in assignments else set()
                  for child in node.children]
        known = set.union(set(), *groups)
        owner = next(iter(known)) if len(known) == 1 else inherited
        for i, child in enumerate(node.children):
            if isinstance(child, Layout):
                assign(child, owner)
                continue
            if id(child) in assignments or id(child) in excluded:
                continue
            # Labels, refresh/reset commands and native editors accompany their
            # closest field. A heterogeneous row keeps its source annotations.
            nearby = next((g for g in groups[i + 1:] if g), None) if child.method in {'label', 'separator'} else None
            if not nearby:
                nearby = next((g for g in reversed(groups[:i]) if g), None)
            group = owner or (next(iter(nearby)) if nearby and len(nearby) == 1 else fallback)
            assignments[id(child)] = group

    assign(source)
    result = []
    for spec in (*specs, Spec(fallback, title)):
        ids = {id(c) for c in controls if assignments.get(id(c), fallback) == spec.key and id(c) not in excluded}
        node = select(source, ids)
        if node and any(c.method not in {'label', 'separator'} for c in node.controls()):
            result.append(Card(spec, node))
        elif node and any(c.method == 'label' and c.kwargs.get('icon') in {'ERROR', 'INFO', 'WARNING'} for c in node.controls()):
            result.append(Card(spec, node))
    return result


def note(p, text, x, y, width, tint='secondary'):
    for _, _, line in wrap(text, width, lambda value: p.measure(value, 10)):
        p.text(line, x, y, 10, tint)
        y += 14
    return y + 4


def heading(p, text, x, y, width, icon='settings'):
    p.icon(icon, x, y + 1, 12, 'secondary')
    p.text(text, x + 19, y, FONT_SIZE, 'text', width - 19)
    return y + 23


def captions(source):
    """Keep native captions when rearranging their fields, without orphan labels."""
    labels, ids = {}, set()
    def visit(node):
        for child in node.children:
            if isinstance(child, Layout):
                visit(child)
        fields = [c for c in node.controls() if c.method == 'prop']
        texts = [c for c in node.children if isinstance(c, Control) and c.method == 'label'
                 and c.kwargs.get('text') and not c.kwargs['text'].startswith('→')]
        if node.method in {'row', 'split'} and len(fields) == len(texts) == 1:
            labels.setdefault(id(fields[0]), texts[0].kwargs['text'].rstrip(':'))
            ids.add(id(texts[0]))
        for i, child in enumerate(node.children[:-1]):
            following = node.children[i + 1]
            if (isinstance(child, Control) and child.method == 'label' and isinstance(following, Layout)
                    and child.kwargs.get('text', '').endswith(':')):
                fields = [c for c in following.controls() if c.method == 'prop']
                if len(fields) == 1:
                    labels.setdefault(id(fields[0]), child.kwargs['text'].rstrip(':'))
                    ids.add(id(child))
    visit(source)
    return labels, ids


def group(p, title, x, y, width, draw, icon='settings'):
    start = y
    p.begin_group()
    y = heading(p, title, x + PAD, y + 11, width - PAD * 2, icon)
    p.rect(x + PAD, y - 3, width - PAD * 2, 1, 'divider')
    y = draw(x + PAD, y + 6, width - PAD * 2) + 8
    p.end_group((x, start, width, y - start), 'subbox', 'section_border', 3)
    return y + GAP


def _body(s, p, node, tree, section, spec, x, y, width):
    """Reorder selected root rows; conditional descendants retain their rails."""
    used = set()
    root_controls = list(node.controls())
    for row in spec.rows:
        parts = []
        for name in row:
            # A descendant belongs to its existing owner, not to a second row.
            ids = {id(c) for c in root_controls if property_name(c) == name}
            if ids:
                part = select(node, ids)
                parts.append(part)
                used.update(ids)
        if not parts:
            continue
        if row[0] in dict(spec.dividers):
            p.rect(x, y + 2, width, 1, 'divider')
            p.text(dict(spec.dividers)[row[0]], x, y + 13, FONT_SIZE, 'secondary', width)
            y += 34
        columns = len(parts) if width >= 320 else 1
        cw = (width - 14 * (columns - 1)) / columns
        for first in range(0, len(parts), columns):
            y = max(s.draw_node(p, part, x + i * (cw + 14), y, cw, section,
                                hierarchy=tree, card_body=True)
                    for i, part in enumerate(parts[first:first + columns]))
    remaining = select(node, {id(c) for c in root_controls if id(c) not in used})
    if remaining:
        y = s.draw_node(p, remaining, x, y, width, section, hierarchy=tree, card_body=True)
    return y


def content(s, p, card, section, x, y, width):
    spec = card.spec
    tree = hierarchy.build(card.content)
    tree.card_labels, caption_ids = captions(card.content)
    tree.redundant_labels.update(caption_ids)
    tree.redundant_labels.update(id(c) for c in card.content.controls()
                                 if c.method == 'label' and c.kwargs.get('text') in DECORATIVE)
    root = tree.root
    toggle = next((live for live in model.controls(root) if property_name(live.control) == spec.feature), None) if spec.feature else None
    cx, cw = x + PAD, width - PAD * 2
    if toggle:
        p.icon(spec.icon, cx, y + 13, 13, 'secondary')
        y = s.draw_plain_control(p, toggle, cx + 21, y + 8, cw - 21, section,
                                 s.control_index(section, toggle.control), spec.title)
        ids = {id(c) for c in root.controls() if c is not toggle.control}
        root = select(root, ids) or Layout()
        children = tree.children.get(id(toggle.control))
        if children:
            root.children.append(children)
    else:
        y = heading(p, spec.title, cx, y + 11, cw, spec.icon)
    if spec.description:
        y = note(p, spec.description, cx, y, cw)
    visible = [c for c in root.controls() if c.method not in {'label', 'separator'}]
    if visible or any(c.method == 'label' and c.kwargs.get('icon') in {'ERROR', 'INFO', 'WARNING'} for c in root.controls()):
        p.rect(cx, y - (3 if not toggle else 0), cw, 1, 'divider')
        y = _body(s, p, root, tree, section, spec, cx, y + 7, cw)
    return y + 8


def draw_family(s, p, parent, children, section, x, y, width):
    """Keep the parent header and its child cards inside one measured boundary."""
    start = y
    p.begin_group()
    y = content(s, p, parent, section, x, y, width)
    s.card_bounds[section + ':' + parent.spec.key + ':header'] = (x, start, width, y - start)
    p.rect(x + PAD, y - 1, width - PAD * 2, 1, 'divider')
    y = draw(s, p, children, section, x + 10, y + 9, width - 20) + 4
    rect = (x, start, width, y - start)
    s.card_bounds[section + ':' + parent.spec.key] = rect
    p.end_group(rect, 'group_body', 'group_border', 3)
    return y + 14


def draw(s, p, cards, section, x, y, width):
    # Relationships affect presentation only. Visibility, bindings and state
    # still come from the original native panel capture.
    keys = {card.spec.key for card in cards}
    children, roots = {}, []
    for card in cards:
        if card.spec.parent and card.spec.parent in keys:
            children.setdefault(card.spec.parent, []).append(card)
        else:
            roots.append(card)
    index = 0
    while index < len(roots):
        card = roots[index]
        if card.spec.key in children:
            y = draw_family(s, p, card, children[card.spec.key], section, x, y, width)
            index += 1
            continue
        row = [card]
        if (width >= SIDE_BY_SIDE_WIDTH and card.spec.beside and index + 1 < len(roots)
                and roots[index + 1].spec.beside == card.spec.beside
                and roots[index + 1].spec.key not in children):
            row.append(roots[index + 1])
        cw = (width - GAP * (len(row) - 1)) / len(row)
        p.begin_group()
        end = max(content(s, p, c, section, x + i * (cw + GAP), y, cw) for i, c in enumerate(row))
        rects = [(x + i * (cw + GAP), y, cw, end - y) for i in range(len(row))]
        for c, rect in zip(row, rects):
            s.card_bounds[section + ':' + c.spec.key] = rect
        p.end_group_row(rects, 'subbox', 'section_border', 3)
        y = end + GAP
        index += len(row)
    return y


def sections(s, p, section, x, y, width, omit=()):
    specs = SCHEMAS.get(section.key, ())
    # Collection cards follow the actual RNA items, including their selection
    # and removal operators. No cached pointer is held after undo or refresh.
    if section.key in {'show_lora_settings', 'show_image_guidance_settings'}:
        collection = 'lora_units' if section.key == 'show_lora_settings' else 'controlnet_units'
        prefix = 'LoRA' if collection == 'lora_units' else 'ControlNet'
        groups = []
        for i, unit in enumerate(getattr(bpy.context.scene, collection)):
            keys = tuple('data:' + str(unit.as_pointer()) + ':' + prop.identifier for prop in unit.bl_rna.properties)
            title = f'{prefix} {i + 1}'
            if collection == 'controlnet_units':
                title += ' · ' + unit.unit_type.replace('_', ' ').title()
            groups.append(Spec(f'unit:{i}', title, 'layers', keys))
        if groups:
            # Resolve names locally to avoid confusing multiple model_name RNA
            # fields with the scene's checkpoint.
            source = section.content
            explicit = {id(c): 'data:' + str(c.args[0].as_pointer()) + ':' + c.args[1]
                        for c in source.controls() if c.method == 'prop'}
            specs = (*specs, *groups)
            cards = partition_data(source, specs, explicit, omit,
                                   'Adapter management' if prefix == 'LoRA' else 'ControlNet management')
        else:
            cards = partition(section.content, specs, omit,
                              'Adapter management' if prefix == 'LoRA' else 'ControlNet management')
    else:
        cards = partition(section.content, specs, omit, section.title.replace(' Settings', ''))
    return draw(s, p, cards, section.key, x, y, width)


def partition_data(source, specs, data_tokens, omit, title):
    """Use per-item tokens while keeping original Controls in the final cards."""
    # Token overrides are scoped to one partition operation, never stored on RNA
    # or on the source controls.
    return partition(source, specs, omit, title, _tokens=data_tokens)


ACTION_DESCRIPTIONS = {
    'object.switch_material': 'Change the material shown on the model.',
    'object.stablegen_mirror_reproject': 'Mirror the last texture projection.',
    'object.stablegen_local_postprocess': 'Decimate, remesh or solidify the active mesh using local methods.',
    'object.add_hdri': 'Add environment lighting to the scene.',
    'object.apply_all_mesh_modifiers': 'Apply modifiers on the scene meshes.',
    'object.curves_to_mesh': 'Convert curves into mesh geometry.',
    'stablegen.import_dae': 'Load a single model.',
    'stablegen.batch_import_dae': 'Import multiple DAE files.',
    'stablegen.batch_texture_dae': 'Process a folder of DAE / GLB models.',
    'stablegen.batch_texture_dae_cancel': 'Stop processing after the current model.',
    'object.export_orbit_gif': 'Render an orbit as a GIF or video.',
    'object.stablegen_make_solid': 'Close mesh boundaries while preserving the rig and projection inputs.',
    'object.stablegen_autorig_selected_mesh': 'Generate a rig for the selected character.',
    'stablegen.debug_projection_line': 'Compare view contributions along a surface line.',
    'stablegen.debug_open_projection_line': 'Open the latest diagnostic graphs.',
    'stablegen.interactive_discard_check': 'Adjust a camera’s context with a live preview.',
    'stablegen.texturing_regression': 'Run the existing workflow checks.',
    'stablegen.stop_texturing_regression': 'Stop the running workflow check.',
    'stablegen.export_retarget_debug_bundle': 'Save a reproducible rig diagnostic package.',
    'stablegen.debug_skinning_weights': 'Inspect how bones influence the mesh.',
    'stablegen.debug_clear_skinning_weights': 'Restore the original vertex colors.',
    'stablegen.debug_convert_fortune_rig': 'Convert the selected rig to Fortune.',
    'stablegen.debug_make_solid': 'Inspect solid-mesh preparation.',
    'stablegen.debug_restore_material': 'Remove the selected meshes’ materials.',
    'stablegen.debug_force_generation_idle': 'Clear a stalled generation state.',
}


def actions(s, p, rows, x, y, width, *, max_columns=2):
    columns = min(max_columns, 2 if width >= SIDE_BY_SIDE_WIDTH else 1)
    cw = (width - GAP * (columns - 1)) / columns
    for first in range(0, len(rows), columns):
        p.begin_group()
        end = y
        rects = []
        for i, (name, label, icon, properties, enabled, hint) in enumerate(rows[first:first + columns]):
            cx = x + i * (cw + GAP)
            p.icon(icon, cx + 12, y + 13, 15, 'secondary' if enabled else 'disabled')
            lines = wrap(label, cw - 68, p.measure)
            cy = y + 12
            for _, _, line in lines:
                p.text(line, cx + 36, cy, FONT_SIZE, 'text' if enabled else 'disabled')
                cy += 15
            cy = note(p, ACTION_DESCRIPTIONS.get(name, hint), cx + 12, cy + 7, cw - 24)
            p.icon('right', cx + cw - 21, y + 16, 10, 'secondary' if enabled else 'disabled')
            end = max(end, cy + 8)
            rects.append((cx, y, cw, 0))
        rects = [(cx, y, cw, end - y) for cx, _, cw, _ in rects]
        for i, (name, _, _, properties, enabled, hint) in enumerate(rows[first:first + columns]):
            key = 'utility:' + name
            p.target(key, rects[i], ('operator', name, properties), enabled, hint)
        # A hovered card changes only its existing boundary and background.
        # Draw each measured backdrop behind its content in one deferred layer.
        styles = [('hover_soft', 'hover_border') if enabled and s.hovered('utility:' + name)
                  else ('subbox', 'section_border')
                  for name, _, _, _, enabled, _ in rows[first:first + columns]]
        p.end_group_row(rects, 'subbox', 'section_border', 3, styles=styles)
        y = end + GAP
    return y + 4
