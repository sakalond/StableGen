"""Core infrastructure for the StableGen addon."""

# The top-level addon package name, used for bpy preferences lookup.
# Modules in sub-packages must use this instead of their own __package__.
ADDON_PKG = __name__.rsplit('.', 1)[0]  # "stablegen" or "bl_ext.user_default.stablegen"


def get_addon_prefs(context=None):
    """Return the StableGenAddonPreferences instance.

    Convenience wrapper so every module doesn't need to know
    the addon package name or handle KeyError.
    """
    import bpy  # noqa: delayed import – safe for Blender
    ctx = context or bpy.context
    wrapper = ctx.preferences.addons.get(ADDON_PKG)
    return wrapper.preferences if wrapper else None


def legacy_panel_enabled(context=None):
    """Whether the sidebar shows the legacy panel instead of the new one."""
    prefs = get_addon_prefs(context)
    return bool(prefs and prefs.use_legacy_panel)
