"""Presentation ownership for conditional controls in the Studio sidebar.

The captured legacy layout remains intact for native popups and command indices.
This adapter moves only visible controls into their owner's presentation scope;
it retains every original Control and every inherited layout enable/alert state.
"""
from dataclasses import dataclass

from .presentation import Control, Layout


# A child may have different owners in different workflows. The closest preceding
# owner on the same RNA data wins, so repeated style controls keep their own scope.
FAMILIES = {
    'discard_factor_generation_only': ('discard_factor_after_generation',),
    'weight_exponent_generation_only': ('weight_exponent_after_generation',),
    'early_priority': ('early_priority_strength',),
    'sg_rework_projection_guard': ('sg_guard_directional', 'sg_silhouette_margin',
        'sg_silhouette_depth', 'sg_guard_coverage_rescue', 'sg_silhouette_feather',
        'sg_projection_guard_scale', 'sg_guard_feather_method'),
    'sg_guard_directional': ('sg_guard_foreground_margin',),
    'sequential_smooth': ('sequential_factor', 'sequential_factor_smooth', 'sequential_factor_smooth_2'),
    'sequential_ipadapter': ('sequential_ipadapter_mode', 'ipadapter_weight_type',
        'ipadapter_strength', 'ipadapter_start', 'ipadapter_end', 'sequential_ipadapter_regenerate'),
    'sequential_ipadapter_regenerate': ('sequential_ipadapter_regenerate_wo_controlnet',),
    'use_ipadapter': ('ipadapter_image', 'ipadapter_strength', 'ipadapter_start',
        'ipadapter_end', 'ipadapter_weight_type'),
    'sequential_ipadapter_mode': ('sequential_desaturate_factor', 'sequential_contrast_factor'),
    'qwen_use_external_style_image': ('qwen_external_style_image', 'qwen_external_style_initial_only'),
    'qwen_external_style_initial_only': ('sequential_ipadapter_mode',),
    'qwen_use_trellis2_style': ('qwen_trellis2_style_initial_only',),
    'qwen_trellis2_style_initial_only': ('sequential_ipadapter_mode',),
    'qwen_use_custom_prompts': ('qwen_custom_prompt_initial', 'qwen_custom_prompt_seq_none',
        'qwen_custom_prompt_seq_replace', 'qwen_custom_prompt_seq_additional'),
    'qwen_prompt_template': ('qwen_custom_prompt_initial', 'qwen_custom_prompt_seq_none',
        'qwen_custom_prompt_seq_replace', 'qwen_custom_prompt_seq_additional'),
    'qwen_context_render_mode': ('qwen_prompt_gray_background', 'qwen_guidance_background_color',
        'qwen_guidance_fallback_color', 'qwen_use_confidence_context'),
    'qwen_use_confidence_context': ('qwen_trust_threshold', 'qwen_context_fallback_dilation',
        'qwen_context_cleanup'),
    'qwen_context_cleanup': ('qwen_context_cleanup_hue_tolerance', 'qwen_context_cleanup_value_adjust'),
    'qwen_regularize_context_mask': ('qwen_mask_min_island_area', 'qwen_mask_min_hole_area',
        'qwen_mask_min_strip_width', 'qwen_mask_uncertainty_dilation', 'qwen_mask_keep_largest_components'),
    'qwen_reject_residual_fallback': ('qwen_residual_fallback_tolerance',),
    'qwen_latent_locked_context': ('qwen_latent_lock_neutral_fill',),
    'sg_auto_view_sequence': ('sg_anchor_camera',),
    'auto_rescale': ('auto_rescale_target_mp', 'qwen_rescale_alignment'),
    'use_separate_texture_prompt': ('texture_prompt',),
    'use_flux_lora': ('canny_threshold_low', 'canny_threshold_high'),
    'differential_diffusion': ('differential_noise',),
    'differential_noise': ('mask_blocky', 'blur_mask', 'grow_mask_by'),
    'blur_mask': ('blur_mask_radius', 'blur_mask_sigma'),
    'refine_angle_ramp_active': ('refine_angle_ramp_pos_0', 'refine_angle_ramp_pos_1'),
    'visibility_vignette': ('refine_feather_ramp_pos_0', 'refine_feather_ramp_pos_1',
        'visibility_vignette_width', 'visibility_vignette_softness', 'visibility_vignette_blur'),
    'refine_edge_feather_projection': ('refine_edge_feather_width', 'refine_edge_feather_softness'),
    'view_blend_use_color_match': ('view_blend_color_match_method', 'view_blend_color_match_strength'),
    'refine_images': ('refine_sampler', 'refine_scheduler', 'denoise', 'refine_cfg',
        'refine_steps', 'refine_prompt', 'refine_upscale_method'),
    'pbr_decomposition': ('pbr_map_albedo', 'pbr_map_roughness', 'pbr_map_metallic',
        'pbr_map_normal', 'pbr_map_height', 'pbr_map_ao', 'pbr_map_emission',
        'pbr_use_native_resolution', 'pbr_tiling', 'pbr_denoise_steps',
        'pbr_ensemble_size', 'pbr_replace_color_with_albedo', 'pbr_auto_lighting'),
    'pbr_map_normal': ('pbr_normal_strength',),
    'pbr_map_height': ('pbr_height_scale',),
    'pbr_map_ao': ('pbr_ao_samples', 'pbr_ao_distance'),
    'pbr_map_emission': ('pbr_emission_method', 'pbr_emission_threshold', 'pbr_emission_strength'),
    'pbr_emission_method': ('pbr_emission_saturation_min', 'pbr_emission_value_min', 'pbr_emission_bloom'),
    'pbr_map_albedo': ('pbr_albedo_source', 'pbr_albedo_auto_saturation'),
    'pbr_albedo_source': ('pbr_delight_strength',),
    'pbr_albedo_auto_saturation': ('pbr_albedo_saturation_mode',),
    'pbr_tiling': ('pbr_tile_grid', 'pbr_tile_inference', 'pbr_tile_superres', 'pbr_tile_albedo',
        'pbr_tile_material', 'pbr_tile_normal', 'pbr_tile_height', 'pbr_tile_emission'),
    'pbr_use_native_resolution': ('pbr_processing_resolution',),
    'trellis2_post_processing_enabled': ('trellis2_decimate_method', 'trellis2_decimation',
        'trellis2_remesh_method', 'trellis2_solidify'),
    'trellis2_exclude_bottom': ('trellis2_exclude_bottom_angle',),
    'trellis2_clamp_elevation': ('trellis2_min_elevation', 'trellis2_max_elevation'),
    'trellis2_preview_gallery_enabled': ('trellis2_preview_gallery_count',),
    'sg_queue_gif_export': ('sg_queue_gif_duration', 'sg_queue_gif_fps',
        'sg_queue_gif_resolution', 'sg_queue_gif_samples', 'sg_queue_gif_engine',
        'sg_queue_gif_interpolation', 'sg_queue_gif_use_hdri', 'sg_queue_gif_denoiser',
        'sg_queue_gif_use_gpu', 'sg_queue_gif_also_no_pbr'),
    'sg_queue_gif_use_hdri': ('sg_queue_gif_hdri_path', 'sg_queue_gif_hdri_strength',
        'sg_queue_gif_hdri_rotation', 'sg_queue_gif_env_mode'),
    'stablegen_print_dithered': ('stablegen_print_dither_method', 'stablegen_print_model_height',
        'stablegen_print_layer_height'),
    'stablegen_print_make_solid': ('stablegen_print_raycast_count', 'stablegen_print_keep_largest_island',
        'stablegen_print_single_filament_internal'),
    'stablegen_print_single_filament_internal': ('stablegen_print_internal_visibility_threshold',),
    'stablegen_print_show_advanced': ('stablegen_print_solver_formula', 'stablegen_print_solver_init',
        'stablegen_print_solver_steps', 'stablegen_print_saturation_penalty'),
    'stablegen_print_solver_formula': ('stablegen_print_scattering_weight',),
}
PARENTS = {}
for parent, children in FAMILIES.items():
    for child in children:
        PARENTS.setdefault(child, []).append(parent)

# Technical notes that are already available in the corresponding field tooltip.
TOOLTIP_LABELS = {'auto_rescale': {'Qwen 2.1 aligns to 32px'}}


def _property(control):
    if control.method != 'prop' or len(control.args) < 2:
        return None
    data, name = control.args[:2]
    return data.as_pointer(), name


@dataclass
class Hierarchy:
    source: Layout
    root: Layout
    children: dict
    redundant_labels: set


def build(source, omit=()):
    """Extract owner scopes without changing the source layout or control order."""
    controls = list(source.controls())
    positions = {id(c): i for i, c in enumerate(controls)}
    properties, by_property = {}, {}
    for c in controls:
        key = _property(c)
        if key:
            properties[id(c)] = key
            if key[1] not in omit:
                by_property.setdefault(key, []).append(id(c))
    owners = {}
    for child, (data, name) in properties.items():
        candidates = [parent for parent_name in PARENTS.get(name, ())
                      for parent in by_property.get((data, parent_name), ())
                      if positions[parent] < positions[child]]
        if candidates:
            owners[child] = max(candidates, key=positions.get)

    def ancestors(cid):
        seen = set()
        while cid in owners:
            cid = owners[cid]
            if cid in seen:
                raise ValueError('Cyclic Studio control ownership')
            seen.add(cid)
        return seen

    boxes = {}
    def claim_boxes(node):
        for c in node.children:
            if isinstance(c, Layout):
                claim_boxes(c)
        if node is source or node.method != 'box':
            return
        members = {id(c) for c in node.controls()}
        chains = [ancestors(cid) - members for cid in members if cid in owners]
        if not chains:
            return
        common = set.intersection(*chains)
        if common:
            parent = max(common, key=lambda cid: len(ancestors(cid)))
            boxes[id(node)] = parent
            for cid in members:
                if cid not in owners:
                    owners[cid] = parent
    claim_boxes(source)

    def title(text):
        words = text.casefold().rstrip(':').split()
        while words and words[-1] in {'options', 'settings', 'parameters'}:
            words.pop()
        return ' '.join(words)

    redundant_labels = set()
    by_id = {id(c): c for c in controls}
    for c in controls:
        parent = by_id.get(owners.get(id(c)))
        if c.method != 'label' or parent is None or parent.method != 'prop':
            continue
        data, name = parent.args[:2]
        caption = title(c.kwargs.get('text', ''))
        if (caption and caption in {title(parent.kwargs.get('text', '')),
                                   title(data.bl_rna.properties[name].name)} or
                c.kwargs.get('text') in TOOLTIP_LABELS.get(name, ())):
            # The controlling field and its rail already name this scope. Keep
            # the native label in the source for native popups and diagnostics.
            redundant_labels.add(id(c))

    def omitted(control):
        key = _property(control)
        return (key is not None and key[1] in omit or
                control.method == 'operator' and 'op:' + control.args[0] in omit)

    def extract(node, owner):
        if isinstance(node, Control):
            if omitted(node):
                return None
            return node if owners.get(id(node)) == owner else None
        if omit and node is not source:
            native_controls = list(node.controls())
            # Custom Studio controls replace some native labeled rows. Keep the
            # renderer's original omission rule before extracting their fields,
            # so their now-empty labels cannot leak back into the presentation.
            if (any(omitted(c) for c in native_controls) and
                    all(omitted(c) or c.method in {'label', 'separator'} for c in native_controls)):
                return None
        result = Layout('column' if id(node) in boxes else node.method, node.args, dict(node.kwargs))
        result.attributes.update(node.attributes)
        result.children.extend(part for c in node.children if (part := extract(c, owner)) is not None)
        if not result.children:
            return None
        # Keep the background margin beside the outer guard, before the nested
        # depth-aware controls, rather than splitting the two margins apart.
        if owner is not None and properties.get(owner, (None, ''))[1] == 'sg_rework_projection_guard' and id(node) in boxes:
            result.children.sort(key=lambda part: 0 if (
                isinstance(part, Control) and properties.get(id(part), (None, ''))[1] == 'sg_silhouette_margin'
                or isinstance(part, Layout) and part.has_property('sg_silhouette_margin')) else 1)
        return result

    root = extract(source, None) or Layout()
    children = {owner: extract(source, owner) for owner in set(owners.values())}
    return Hierarchy(source, root, {key: value for key, value in children.items() if value is not None},
                     redundant_labels)
