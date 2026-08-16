"""StableGen texturing experiment switches and legacy-compatible defaults."""

# This mapping is deliberately Blender-independent so tests and preset code can
# verify the baseline without importing the add-on registration graph.
TEXTURING_REWORK_DEFAULTS = {
    "sg_rework_camera_plan": False,
    "sg_rework_aligned_resolution": False,
    "sg_rework_full_resolution_scale": False,
    "sg_rework_safe_render_state": False,
    "sg_rework_reset_controls": False,
    "sg_rework_sdxl_registration": False,
    "sg_rework_acquisition_quality": False,
    "sg_rework_initial_view_priority": False,
    "sg_rework_projection_guard": False,
    "sg_rework_projection_rebuild": False,
    "sg_rework_projection_geometry": False,
    "sg_rework_consistent_frame": False,
    "sg_rework_workflow_validation": False,
    "sg_rework_qwen_workflow": False,
    "qwen_global_style_vlm_only": False,
    "qwen_latent_locked_context": False,
    "sg_rework_final_display": False,
    "qwen_use_confidence_context": False,
    "qwen_regularize_context_mask": False,
    "qwen_use_projection_write_masks": False,
    "qwen_reject_residual_fallback": False,
    "sg_texel_ownership": False,
    "sg_auto_view_sequence": False,
    "sg_freeze_ownership_after_generation": False,
    # Legacy projection used abs(dot), i.e. two-sided/backface contribution.
    "sg_allow_backface_projection": True,
}


def all_rework_features_disabled(scene):
    """Return True when *scene* is configured for the legacy baseline."""
    return all(
        getattr(scene, name, default) == default
        for name, default in TEXTURING_REWORK_DEFAULTS.items()
    )
