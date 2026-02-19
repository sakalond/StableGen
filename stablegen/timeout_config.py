"""Timeout configuration helper for StableGen."""

import bpy  # pylint: disable=import-error

_DEFAULTS = {'ping': 1.0, 'api': 10.0, 'transfer': 120.0, 'reboot': 120.0, 'mesh_gen': 600.0}


def get_timeout(category, default=None):
    """Return a timeout value (seconds) from the addon preferences.

    Args:
        category: One of ``'ping'``, ``'api'``, ``'transfer'``, ``'reboot'``.
        default:  Fallback value if the preference cannot be read.
                  When *None*, built-in defaults are:
                  ping=1.0, api=10.0, transfer=120.0, reboot=120.0.
    """
    if default is None:
        default = _DEFAULTS.get(category, 10.0)
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        return getattr(prefs, f'timeout_{category}', default)
    except Exception:
        return default
