"""Studio tips whose dismissal lasts across files, sessions and add-on updates.

Dismissals live in a small file in StableGen's user folder instead of the
add-on preferences, so they persist even when preference auto-save is off.
"""
import json
import logging
from pathlib import Path

import bpy

from ..core import ADDON_PKG

WIDEN_CALLOUT = 'widen_callout'
WIDEN_HINT = 'widen_hint'

_log = logging.getLogger(__name__)
_dismissed = None


def _path():
    try:
        folder = bpy.utils.extension_path_user(ADDON_PKG, create=True)
    except ValueError:
        # Legacy add-on installs have no extension user folder.
        folder = bpy.utils.user_resource('CONFIG', path=ADDON_PKG, create=True)
    return Path(folder) / 'tips.json'


def _load():
    global _dismissed
    if _dismissed is None:
        try:
            _dismissed = set(json.loads(_path().read_text(encoding='utf-8')).get('dismissed', ()))
        except FileNotFoundError:
            _dismissed = set()
        except (OSError, ValueError, AttributeError):
            _log.warning('Could not read StableGen tips; showing them again', exc_info=True)
            _dismissed = set()
    return _dismissed


def dismissed(name):
    return name in _load()


def dismiss(name):
    tips = _load()
    if name in tips:
        return
    tips.add(name)
    try:
        _path().write_text(json.dumps({'dismissed': sorted(tips)}), encoding='utf-8')
    except OSError:
        # The tip stays hidden for this session even if the file is unwritable.
        _log.warning('Could not save dismissed StableGen tip', exc_info=True)


def reset():
    global _dismissed
    _dismissed = set()
    try:
        _path().unlink(missing_ok=True)
    except OSError:
        _log.warning('Could not reset StableGen tips', exc_info=True)


class SG_OT_ResetTips(bpy.types.Operator):
    bl_idname = 'stablegen.reset_tips'
    bl_label = 'Reset Tips'
    bl_description = 'Show dismissed StableGen tips again, such as the hint to widen the sidebar'

    def execute(self, context):
        reset()
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        self.report({'INFO'}, 'StableGen tips will show again')
        return {'FINISHED'}
