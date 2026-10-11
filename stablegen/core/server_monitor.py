"""Keep the ComfyUI online status current while a StableGen sidebar is open.

The status is otherwise only checked at startup and on demand, so the Generate
button kept trusting it after ComfyUI stopped, and needed a manual refresh
after ComfyUI started.
"""
import logging

import bpy  # pylint: disable=import-error

from . import ADDON_PKG
from .server_api import check_server_availability
from .state import _run_async, show_offline_lists
from ..timeout_config import get_timeout
from ..utils import sg_modal_active

INTERVAL = 10.0
SIDEBAR_TABS = {'StableGen', 'StableGen Print'}

_log = logging.getLogger(__name__)
_in_flight = False


def _sidebar_open(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D' or not area.spaces.active.show_region_ui:
                continue
            region = next((r for r in area.regions if r.type == 'UI'), None)
            if region and region.active_panel_category in SIDEBAR_TABS:
                return True
    return False


def _busy(context):
    """Generation keeps its own connection; don't ping alongside it or another check."""
    from .preferences import CheckServerStatus
    scene = context.scene
    return (CheckServerStatus._checking or sg_modal_active(context)
            or (scene is not None and getattr(scene, 'generation_status', 'idle') == 'waiting'))


def _redraw(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _apply(address, online):
    global _in_flight
    _in_flight = False
    context = bpy.context
    wrapper = context.preferences.addons.get(ADDON_PKG)
    # None means the check failed or a newer server check superseded it.
    if online is None or wrapper is None or wrapper.preferences.server_address != address:
        return
    prefs = wrapper.preferences
    if online and not prefs.server_online:
        # The full check also refreshes model lists and node availability.
        if bpy.ops.stablegen.check_server_status.poll():
            bpy.ops.stablegen.check_server_status('INVOKE_DEFAULT')
    elif not online and prefs.server_online:
        print(f"[StableGen] ComfyUI went offline at {address}.")
        prefs.server_online = False
        show_offline_lists(address)
        _redraw(context)


def _tick():
    global _in_flight
    try:
        context = bpy.context
        wrapper = context.preferences.addons.get(ADDON_PKG)
        if (_in_flight or wrapper is None or not wrapper.preferences.server_address
                or not bpy.app.online_access or not _sidebar_open(context) or _busy(context)):
            return INTERVAL
        address = wrapper.preferences.server_address
        _in_flight = True
        _run_async(lambda: check_server_availability(address, timeout=get_timeout('ping')),
                   lambda online: _apply(address, online))
    except Exception:
        _in_flight = False
        _log.exception('ComfyUI status check failed')
    return INTERVAL


def register():
    if not bpy.app.background and not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=INTERVAL, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
