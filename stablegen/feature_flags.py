"""Regular texturing defaults and preference-gated research options."""

GRADUATED_FEATURE_DEFAULTS = {
    "qwen_use_confidence_context": False,
    "sg_rework_acquisition_quality": False,
    "sg_rework_projection_guard": False,
    "sg_allow_backface_projection": True,
}
EXPERIMENTAL_FEATURE_DEFAULTS = dict.fromkeys((
    "sg_rework_aligned_resolution", "sg_rework_full_resolution_scale",
    "sg_rework_final_display", "qwen_global_style_vlm_only",
    "qwen_context_vlm_only", "qwen_latent_locked_context",
    "qwen_latent_lock_neutral_fill", "qwen_regularize_context_mask",
    "qwen_use_projection_write_masks", "qwen_reject_residual_fallback",
    "sg_auto_view_sequence",
), False)
TEXTURING_DEFAULTS = {
    **GRADUATED_FEATURE_DEFAULTS, **EXPERIMENTAL_FEATURE_DEFAULTS,
    'sg_silhouette_feather': 12,
    'sg_projection_guard_scale': '2',
    'sg_guard_feather_method': 'ORIGINAL',
    'sg_guard_directional': True,
    'sg_guard_foreground_margin': 0,
    'sg_guard_coverage_rescue': 0.25,
}


def experiments_enabled(context=None):
    """Fail closed when the addon is absent, including background helpers."""
    try:
        if context is None:
            import bpy
            context = bpy.context
        addon = context.preferences.addons.get(__package__)
        return bool(addon and addon.preferences.enable_texturing_experiments)
    except (AttributeError, ImportError, KeyError, ReferenceError):
        return False


def experimental_enabled(scene, name, context=None):
    """Saved switches are effective only while the application gate permits."""
    if name not in EXPERIMENTAL_FEATURE_DEFAULTS:
        raise ValueError(f"Not a texturing experiment: {name}")
    return experiments_enabled(context) and bool(getattr(scene, name, False))


def active_experiments(scene, context=None):
    if not experiments_enabled(context):
        return []
    return [name for name in EXPERIMENTAL_FEATURE_DEFAULTS if getattr(scene, name, False)]
