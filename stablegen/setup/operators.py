"""Operators behind the ComfyUI section of StableGen's preferences."""
import bpy  # pylint: disable=import-error

from . import runtime


class STABLEGEN_OT_comfy_start(bpy.types.Operator):
    """Start ComfyUI hidden in the background (or connect to the one already running for this folder)"""
    bl_idname = "stablegen.comfy_start"
    bl_label = "Start ComfyUI"

    @classmethod
    def poll(cls, context):
        if runtime.comfy().state in ('starting', 'running'):
            cls.poll_message_set("ComfyUI is already running")
            return False
        return True

    def execute(self, context):
        if runtime.comfy().state == 'unresponsive':
            runtime.stop()
        error = runtime.start()
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Starting ComfyUI at {runtime.comfy().address}")
        return {'FINISHED'}


class STABLEGEN_OT_comfy_stop(bpy.types.Operator):
    """Stop the ComfyUI StableGen started (one started outside StableGen keeps running)"""
    bl_idname = "stablegen.comfy_stop"
    bl_label = "Stop ComfyUI"

    @classmethod
    def poll(cls, context):
        return runtime.comfy().state in ('starting', 'running', 'unresponsive')

    def execute(self, context):
        owned = runtime.comfy().owned
        runtime.stop()
        if owned:
            context.preferences.addons[runtime.ADDON_PKG].preferences.server_online = False
        self.report({'INFO'}, "ComfyUI stopped" if owned else "Disconnected; that ComfyUI was not started by StableGen")
        return {'FINISHED'}


class STABLEGEN_OT_comfy_restart(bpy.types.Operator):
    """Restart the ComfyUI StableGen started, so it loads nodes installed since it started"""
    bl_idname = "stablegen.comfy_restart"
    bl_label = "Restart ComfyUI"

    @classmethod
    def poll(cls, context):
        if not runtime.comfy().owned:
            cls.poll_message_set("StableGen didn't start this ComfyUI, so it won't restart it: restart it yourself")
            return False
        return True

    def execute(self, context):
        runtime.stop()
        context.preferences.addons[runtime.ADDON_PKG].preferences.server_online = False
        error = runtime.start()
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        runtime.restart_needed = None
        self.report({'INFO'}, f"Restarting ComfyUI at {runtime.comfy().address}")
        return {'FINISHED'}


class STABLEGEN_OT_comfy_open_log(bpy.types.Operator):
    """Open the log of the last installer run or ComfyUI start"""
    bl_idname = "stablegen.comfy_open_log"
    bl_label = "Open Log"

    which: bpy.props.EnumProperty(items=[('COMFYUI', "ComfyUI", ""), ('INSTALL', "Install", "")],
                                  default='COMFYUI')  # type: ignore

    def execute(self, context):
        if self.which == 'INSTALL' and runtime.job:
            path = runtime.job.log_path
        else:
            path = runtime.comfy().log_path
        if not path.is_file():
            self.report({'WARNING'}, "No log yet")
            return {'CANCELLED'}
        bpy.ops.wm.path_open(filepath=str(path))
        return {'FINISHED'}


class STABLEGEN_OT_comfy_refresh(bpy.types.Operator):
    """Read again which models are installed and which Python and GPU ComfyUI uses"""
    bl_idname = "stablegen.comfy_refresh"
    bl_label = "Refresh"

    def execute(self, context):
        if runtime.comfy_folder() is None:
            self.report({'WARNING'}, "Choose the ComfyUI folder first")
            return {'CANCELLED'}
        runtime.refresh_status()
        return {'FINISHED'}


class STABLEGEN_OT_comfy_install(bpy.types.Operator):
    """Download and install these models and the nodes they need into ComfyUI; while another install runs, it waits
    in the queue"""
    bl_idname = "stablegen.comfy_install"
    bl_label = "Install"

    tags: bpy.props.StringProperty(options={'HIDDEN'})  # type: ignore

    @classmethod
    def poll(cls, context):
        if runtime.job and runtime.job.running and runtime.job.kind == "create":
            cls.poll_message_set("StableGen's own ComfyUI is still installing")
            return False
        if runtime.comfy_folder() is None:
            cls.poll_message_set("Choose the ComfyUI folder first")
            return False
        return True

    def execute(self, context):
        if not bpy.app.online_access:
            self.report({'ERROR'}, "Allow online access in Blender's preferences (System) to download models")
            return {'CANCELLED'}
        runtime.begin_install([t for t in self.tags.split(",") if t])
        return {'FINISHED'}


class STABLEGEN_OT_comfy_remove(bpy.types.Operator):
    """Delete it from ComfyUI: its models and unfinished downloads, and in StableGen's own ComfyUI its nodes.
    What another installed package uses stays; in your own ComfyUI folder the nodes stay too"""
    bl_idname = "stablegen.comfy_remove"
    bl_label = "Remove"

    tags: bpy.props.StringProperty(options={'HIDDEN'})  # type: ignore
    title: bpy.props.StringProperty(options={'HIDDEN'})  # type: ignore

    @classmethod
    def poll(cls, context):
        if runtime.job and runtime.job.running and runtime.job.kind == "create":
            cls.poll_message_set("StableGen's own ComfyUI is still installing")
            return False
        return True

    def invoke(self, context, event):
        if runtime.busy_tags() & set(self._tags()):
            self.report({'WARNING'}, "It is installing or in the queue; cancel that first")
            return {'CANCELLED'}
        paths = runtime.removable(self._tags())
        if not paths:
            self.report({'INFO'}, "Nothing of it to remove: another installed package uses its files")
            return {'CANCELLED'}
        from .packages import size_text
        size = size_text(sum(runtime.path_size(p) for p in paths) / 1024**2)
        nodes = sum(p.parent.name == "custom_nodes" for p in paths)
        files = len(paths) - nodes
        unfinished = sum(p.suffix == ".part" for p in paths)
        parts = ([f"{files} model file{'s' if files != 1 else ''}" + (f" ({unfinished} unfinished)" if unfinished
                                                                    else "")] if files else [])
        parts += [f"{nodes} node{'s' if nodes != 1 else ''}"] if nodes else []
        kept = runtime.kept_nodes(self._tags(), paths)
        stays = " Its other nodes stay." if kept and nodes else " Its nodes stay." if kept else ""
        return context.window_manager.invoke_confirm(
            self, event, title=f"Remove {self.title or 'it'}?",
            message=f"Deletes {' and '.join(parts)}, {size}.{stays}", confirm_text="Remove", icon='WARNING')

    def execute(self, context):
        removed, failed = runtime.remove(self._tags())
        from .packages import size_text
        if failed:
            self.report({'WARNING'}, f"Removed {size_text(removed / 1024**2)}; {len(failed)} in use by ComfyUI "
                                     f"stayed (stop it and remove again)")
        else:
            self.report({'INFO'}, f"Removed {size_text(removed / 1024**2)}")
        return {'FINISHED'}

    def _tags(self):
        return [t for t in self.tags.split(",") if t]


class STABLEGEN_OT_comfy_unqueue(bpy.types.Operator):
    """Take it out of the install queue"""
    bl_idname = "stablegen.comfy_unqueue"
    bl_label = "Remove from Queue"

    tags: bpy.props.StringProperty(options={'HIDDEN'})  # type: ignore

    def execute(self, context):
        runtime.unqueue([t for t in self.tags.split(",") if t])
        return {'FINISHED'}


class STABLEGEN_OT_comfy_confirm_install(bpy.types.Operator):
    """Accept the license notices and changes listed above and start installing"""
    bl_idname = "stablegen.comfy_confirm_install"
    bl_label = "Install"

    def execute(self, context):
        error = runtime.confirm_install()
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        return {'FINISHED'}


class STABLEGEN_OT_comfy_cancel_review(bpy.types.Operator):
    """Don't install"""
    bl_idname = "stablegen.comfy_cancel_review"
    bl_label = "Cancel"

    def execute(self, context):
        runtime.cancel_review()
        return {'FINISHED'}


class STABLEGEN_OT_comfy_create(bpy.types.Operator):
    """Download and set up StableGen's own ComfyUI in the install folder: uv, Python 3.12, torch for your GPU and a
    tested ComfyUI release. Running it again repairs or updates it"""
    bl_idname = "stablegen.comfy_create"
    bl_label = "Install ComfyUI"

    @classmethod
    def poll(cls, context):
        if runtime.job and runtime.job.running:
            cls.poll_message_set("An install is running")
            return False
        if not context.preferences.addons[runtime.ADDON_PKG].preferences.comfy_root:
            cls.poll_message_set("Choose the install folder first")
            return False
        return True

    def execute(self, context):
        error = runtime.create_own()
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        return {'FINISHED'}


class STABLEGEN_OT_comfy_choose_folder(bpy.types.Operator):
    """Choose the folder"""
    bl_idname = "stablegen.comfy_choose_folder"
    bl_label = "Choose Folder"
    bl_options = {'INTERNAL'}

    target: bpy.props.EnumProperty(items=[('ROOT', "Install folder", "Where StableGen keeps its own ComfyUI"),
                                          ('LOCAL', "ComfyUI folder", "Your ComfyUI folder")],
                                   default='ROOT', options={'HIDDEN'})  # type: ignore
    directory: bpy.props.StringProperty(subtype='DIR_PATH')  # type: ignore

    def invoke(self, context, event):
        p = context.preferences.addons[runtime.ADDON_PKG].preferences
        self.directory = p.comfy_root if self.target == 'ROOT' else p.comfy_local_path
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        p = context.preferences.addons[runtime.ADDON_PKG].preferences
        setattr(p, "comfy_root" if self.target == 'ROOT' else "comfy_local_path", self.directory)
        runtime.redraw()
        return {'FINISHED'}


class STABLEGEN_OT_comfy_set_address(bpy.types.Operator):
    """Set the address of the ComfyUI server"""
    bl_idname = "stablegen.comfy_set_address"
    bl_label = "ComfyUI Server"
    bl_options = {'INTERNAL'}

    address: bpy.props.StringProperty(name="Address", description="Host and port, for example 127.0.0.1:8188")  # type: ignore

    def invoke(self, context, event):
        self.address = context.preferences.addons[runtime.ADDON_PKG].preferences.server_address
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        context.preferences.addons[runtime.ADDON_PKG].preferences.server_address = self.address.strip()
        context.preferences.is_dirty = True         # saved on quit, like a change in the preferences
        runtime.redraw()
        return {'FINISHED'}


class STABLEGEN_OT_dismiss_setup_card(bpy.types.Operator):
    """Hide the setup card; the server address stays where it is (change it in StableGen's preferences)"""
    bl_idname = "stablegen.dismiss_setup_card"
    bl_label = "I have a server"

    def execute(self, context):
        from ..ui import studio_tips
        from .prefs_ui import SETUP_CARD
        studio_tips.dismiss(SETUP_CARD)
        runtime.redraw()
        return {'FINISHED'}


class STABLEGEN_OT_comfy_copy_install_command(bpy.types.Operator):
    """Copy the command that installs this model on the server: run it there with installer.py from the StableGen
    repository"""
    bl_idname = "stablegen.comfy_copy_install_command"
    bl_label = "Copy Install Command"

    tags: bpy.props.StringProperty(options={'HIDDEN'})  # type: ignore

    def execute(self, context):
        command = f'python installer.py "<ComfyUI folder>" --packages {self.tags}'
        context.window_manager.clipboard = command
        self.report({'INFO'}, f"Copied: {command}  (run it on the server)")
        return {'FINISHED'}


class STABLEGEN_OT_comfy_use_server_folder(bpy.types.Operator):
    """Let StableGen install models into the ComfyUI running on this computer: use its folder (Use my ComfyUI
    folder) and open the setup"""
    bl_idname = "stablegen.comfy_use_server_folder"
    bl_label = "Use This ComfyUI's Folder"

    def execute(self, context):
        from .comfy_process import server_folder
        p = context.preferences.addons[runtime.ADDON_PKG].preferences
        folder = server_folder(p.server_address)
        if folder is not None:
            p.comfy_local_path = str(folder)
        p.comfy_mode = 'LOCAL'
        from ..ui import studio
        for session in studio._sessions.values():
            session.setup_change = True     # the choices stay open, to confirm the folder with Use
        bpy.ops.stablegen.open_setup()
        if folder is None:
            self.report({'WARNING'}, "Couldn't ask this ComfyUI for its folder: choose it")
        return {'FINISHED'}


class STABLEGEN_OT_comfy_cancel_install(bpy.types.Operator):
    """Stop this install; finished files stay, downloads resume the next time, and the queue goes on"""
    bl_idname = "stablegen.comfy_cancel_install"
    bl_label = "Cancel Install"

    @classmethod
    def poll(cls, context):
        return bool(runtime.job and runtime.job.running)

    def execute(self, context):
        runtime.cancel_job()
        return {'FINISHED'}


SETUP_CLASSES = (
    STABLEGEN_OT_comfy_start,
    STABLEGEN_OT_comfy_stop,
    STABLEGEN_OT_comfy_restart,
    STABLEGEN_OT_comfy_open_log,
    STABLEGEN_OT_comfy_refresh,
    STABLEGEN_OT_comfy_install,
    STABLEGEN_OT_comfy_remove,
    STABLEGEN_OT_comfy_unqueue,
    STABLEGEN_OT_comfy_confirm_install,
    STABLEGEN_OT_comfy_cancel_review,
    STABLEGEN_OT_comfy_create,
    STABLEGEN_OT_comfy_choose_folder,
    STABLEGEN_OT_comfy_set_address,
    STABLEGEN_OT_dismiss_setup_card,
    STABLEGEN_OT_comfy_copy_install_command,
    STABLEGEN_OT_comfy_use_server_folder,
    STABLEGEN_OT_comfy_cancel_install,
)
