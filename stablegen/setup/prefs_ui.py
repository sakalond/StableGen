"""The ComfyUI and Models sections of StableGen's preferences (layout A of ref/setup-ui-20261103).

Where ComfyUI comes from (StableGen's own, a folder of the user's, or a remote server), its status with Start and
Stop, and the models to install with their progress. The install waiting for review lists its license notices
and, for a ComfyUI of the user's, what it changes in that ComfyUI's Python.

Drawn with Blender's own widgets the way its preference pages are: collapsible sub-panels, a label column on the
left, secondary text greyed, compact buttons on the right, and the models in two columns.
"""
from . import installer, packages, runtime

STATE_TEXT = {'stopped': ("Stopped", 'RADIOBUT_OFF'), 'starting': ("Starting…", 'SORTTIME'),
              'running': ("Running", 'CHECKMARK'), 'exited': ("Stopped unexpectedly", 'ERROR'),
              'unresponsive': ("Not answering", 'ERROR')}
GROUP_ICONS = {"Texturing": 'IMAGE_DATA', "Experimental": 'EXPERIMENTAL', "Deprecated": 'IMAGE_DATA',
               "3D, maps and rigging": 'MESH_CUBE'}
ROW_ICONS = {"trellis2": 'MESH_CUBE', "pbr_marigold": 'NODE_MATERIAL', "pbr_marigold_v2": 'NODE_MATERIAL',
             "skintokens": 'ARMATURE_DATA', "lato2": 'MOD_REMESH'}
LEFT_GROUPS, RIGHT_GROUPS = ("Texturing", "Experimental", "Deprecated"), ("3D, maps and rigging",)
SPLIT = 0.22                # the label column's share of the section
WIDTH = 36                  # the section's width in UI units: a wide Preferences window must not spread it out


def draw(p, layout, context):
    header, body = layout.panel("stablegen_setup_comfyui", default_closed=False)
    header.label(text="ComfyUI", icon='NETWORK_DRIVE')
    if body is None:
        return
    col = _container(body)
    _value(col, "Use").prop(p, "comfy_mode", expand=True)

    if p.comfy_mode == 'REMOTE':
        row = _value(col, "Server address")
        row.prop(p, "server_address", text="")
        row.operator("stablegen.check_server_status", text="", icon='FILE_REFRESH')
        _note(col, "Install models on that machine with installer.py from the StableGen repository.")
        return

    if p.comfy_mode == 'LOCAL':
        _value(col, "ComfyUI folder").prop(p, "comfy_local_path", text="")
        folder = runtime.comfy_folder(p)
        if folder is None:
            if p.comfy_local_path:
                _note(col, "No ComfyUI here: choose the folder with 'models' and 'custom_nodes'.", icon='ERROR',
                      alert=True)
            return
    else:
        _value(col, "Install folder").prop(p, "comfy_root", text="")
        folder = runtime.comfy_folder(p)
        job = runtime.job if runtime.job is not None and runtime.job.kind == "create" else None
        if folder is None or (job is not None and (job.running or not job.ok)):
            _draw_own_install(col, p, job, ready=folder is not None)
            if folder is None:
                return
        elif runtime.own_outdated(p):
            right = _value(col)
            right.label(text=f"StableGen now uses ComfyUI {installer.COMFYUI_TAG}", icon='INFO')
            right.operator("stablegen.comfy_create", text="Update", icon='FILE_REFRESH')

    _draw_environment(col, folder)
    _draw_server(col, p)
    if p.use_legacy_panel:          # the StableGen panel's Setup page has the models otherwise
        _draw_models(layout, p, folder)
    else:
        row = _value(col, "Models")
        row.alignment = 'LEFT'
        row.operator("stablegen.open_setup", text="Install models in the StableGen panel", icon='PACKAGE')


# --- layout helpers ---
def _container(layout):
    """A fixed-width column at the left of a sub-panel."""
    row = layout.row()
    row.alignment = 'LEFT'
    col = row.column()
    col.ui_units_x = WIDTH
    return col


def _value(col, label=""):
    """A row in the value column, with an optional label in the label column (like a property)."""
    split = col.split(factor=SPLIT)
    left = split.row()
    left.alignment = 'RIGHT'
    left.label(text=label)
    return split.row()


def _note(col, text, icon='NONE', alert=False):
    """Secondary text under the field it belongs to."""
    row = _value(col)
    row.active = alert
    row.alert = alert
    row.label(text=text, icon=icon)


def _right(layout):
    row = layout.row(align=True)
    row.alignment = 'RIGHT'
    return row


# --- ComfyUI ---
def _free_space_text(path_text):
    """'79 GB free on E:' for the nearest existing folder of path_text."""
    import shutil
    from pathlib import Path
    import bpy
    path = Path(bpy.path.abspath(path_text)).expanduser()
    while not path.exists() and path.parent != path:
        path = path.parent
    try:
        free = shutil.disk_usage(path).free / 1024**3
    except OSError:
        return ""
    return f"{free:.0f} GB free on {path.anchor.rstrip(chr(92)) or path.anchor}"


def _draw_own_install(col, p, job, ready):
    """Installing StableGen's own ComfyUI: what it takes, then the steps while it runs."""
    if job is not None and job.running:
        right = _value(col, "Installing")
        right.progress(factor=job.progress(), text=job.status_line())
        right.operator("stablegen.comfy_cancel_install", text="", icon='X')
        steps = _value(col).column(align=True)
        for step_id in job.order:
            item = job.items[step_id]
            done = item.get("status") in ('installed', 'present')
            row = steps.row()
            row.active = done or step_id == job.current
            row.label(text=item["name"], icon='CHECKMARK' if done else 'ERROR' if item.get("status") == 'failed'
                      else 'SORTTIME' if step_id == job.current else 'RADIOBUT_OFF')
        return
    if job is not None and not job.ok:
        right = _value(col)
        right.alert = True
        right.label(text=job.status_line(), icon='ERROR')
        right.operator("stablegen.comfy_open_log", text="Log", icon='TEXT').which = 'INSTALL'
    if ready:
        return
    space = _free_space_text(p.comfy_root) if p.comfy_root else ""
    _note(col, f"ComfyUI {installer.COMFYUI_TAG}, Python {installer.PYTHON_VERSION} and torch for your GPU: about "
               f"5 GB, plus the models you add below." + (f"  {space}." if space else ""))
    row = _value(col)
    row.alignment = 'LEFT'
    row.scale_y = 1.3
    row.operator("stablegen.comfy_create", text="Retry" if job is not None else "Install ComfyUI", icon='IMPORT')


def _draw_environment(col, folder):
    report = runtime.ensure_report(folder)
    if report is None or report.get("loading"):
        _note(col, "Reading ComfyUI's Python and installed models…", icon='SORTTIME')
    elif "error" in report:
        _note(col, report.get("message", "Could not read this ComfyUI"), icon='ERROR', alert=True)
    else:
        py = report.get("python", {})
        parts = [f"Python {py.get('version', '?')}", f"torch {py.get('torch') or 'not installed'}"]
        if py.get("gpu"):
            parts.append(f"{py['gpu']}" + (f" · {round(py['vram_mb'] / 1024)} GB" if py.get("vram_mb") else ""))
        _note(col, "  ·  ".join(parts))
        if py.get("default"):
            _note(col, f"The default python on PATH (this folder has none of its own): {py.get('path')}")


def _draw_server(col, p):
    c = runtime.comfy()
    text, icon = STATE_TEXT.get(c.state, (c.state, 'QUESTION'))
    if c.state in ('running', 'starting'):
        text += f" on {c.address}"
        if not c.owned:
            text += " (started outside StableGen)"
    elif c.message:
        text += f": {c.message}"
    box = _value(col, "Status").box()
    row = box.row()
    row.alert = c.state in ('exited', 'unresponsive')
    row.label(text=text, icon=icon)
    buttons = _right(row)
    buttons.operator("stablegen.comfy_open_log", text="Log", icon='TEXT').which = 'COMFYUI'
    if c.state in ('running', 'starting', 'unresponsive'):
        buttons.operator("stablegen.comfy_stop", text="Stop", icon='PAUSE')
    else:
        buttons.operator("stablegen.comfy_start", text="Restart" if c.state == 'exited' else "Start", icon='PLAY')
    folder = runtime.comfy_folder(p)
    if runtime.restart_needed and folder is not None and runtime.restart_needed == str(folder) and not c.owned:
        note = box.row()
        note.alert = True
        note.label(text="Restart ComfyUI so it loads the nodes just installed (StableGen didn't start it)",
                   icon='ERROR')
    options = _value(col, "Options").column(align=True)
    options.prop(p, "comfy_auto_start")
    options.prop(p, "comfy_keep_running")
    header, advanced = col.panel("stablegen_setup_advanced", default_closed=True)
    header.label(text="Port and arguments")
    if advanced is not None:
        _value(advanced, "Port").prop(p, "comfy_port", text="")
        _value(advanced, "Extra arguments").prop(p, "comfy_extra_args", text="")


# --- Models ---
def _draw_models(layout, p, folder):
    report = runtime.report(folder)
    rows = list(packages.visible_rows(p.enable_deprecated_architectures, p.enable_experimental_architectures))
    states = {tags: packages.row_state(tags, report) for _, tags, _, _ in rows}
    header, body = layout.panel("stablegen_setup_models", default_closed=False)
    header.label(text="Models", icon='PACKAGE')
    summary = _right(header)
    if report and "tags" in report:
        installed = sum(s["size_mb"] for s in states.values() if s["state"] == "installed")
        info = summary.row()
        info.active = False
        free = f"{report['free_mb'] / 1024:.0f} GB free"
        info.label(text=f"{packages.size_text(installed)} installed  ·  {free}" if installed else free)
    summary.operator("stablegen.comfy_refresh", text="", icon='FILE_REFRESH', emboss=False)
    if body is None:
        return

    job = runtime.job
    if job is not None and job.kind == "install" and job.folder == str(folder) and not job.running and not job.ok:
        row = body.row()
        row.alert = True
        row.label(text=f"{job.title}: {job.status_line()}", icon='ERROR')
        _right(row).operator("stablegen.comfy_open_log", text="Log", icon='TEXT').which = 'INSTALL'
    if runtime.review is not None and runtime.review["folder"] == str(folder):
        _draw_review(body, runtime.review)

    installing = set(job.tags) if job is not None and job.running else set()
    columns = body.split(factor=0.5)
    for column, groups in ((columns.column(), LEFT_GROUPS), (columns.column(), RIGHT_GROUPS)):
        for group in groups:
            members = [r for r in rows if r[0] == group]
            if not members:
                continue
            heading = column.row()
            heading.active = False
            heading.label(text=group)
            box = column.box()
            for i, (_, tags, title, purpose) in enumerate(members):
                if i:
                    box.separator(factor=0.4)
                _model_row(box, tags, title, purpose, states[tags], job if installing & set(tags) else None)
            column.separator()


def _model_row(box, tags, title, purpose, state, job):
    split = box.split(factor=0.64)
    text = split.column(align=True)
    text.label(text=title, icon=ROW_ICONS.get(tags[0], 'IMAGE_DATA'))
    right = _right(split)
    if job is not None:
        rate = job.rate_text()
        text.progress(factor=job.progress(), text=f"{round(job.progress() * 100)}%  ·  {job.size_text()}" + (
            f"  ·  {rate}" if rate else ""))
        right.operator("stablegen.comfy_cancel_install", text="", icon='X')
        return
    place = runtime.queued(list(tags))
    if place is not None:
        meta = text.row()
        meta.active = False
        meta.label(text=f"Queued  ·  {packages.queue_text(place)}", icon='SORTTIME')
        right.operator("stablegen.comfy_unqueue", text="", icon='X').tags = ",".join(tags)
        return
    size = state["size_mb"] if state["state"] == "installed" else state["download_mb"]
    meta = text.row()
    meta.active = False
    meta.label(text=packages.meta_text(purpose, tags, size), icon='BLANK1')
    if state["missing_core_nodes"]:
        warn = text.row()
        warn.alert = True
        warn.label(text="Needs a newer ComfyUI", icon='ERROR')
    if packages.removable_state(tags, state, runtime.node_policy()):
        op = right.operator("stablegen.comfy_remove", text="", icon='TRASH')
        op.tags, op.title = ",".join(tags), title
    if state["state"] == "installed":
        done = right.row()
        done.active = False
        done.label(text="Installed", icon='CHECKMARK')
    elif state["state"] != "unknown":
        op = right.operator("stablegen.comfy_install", text="Finish" if state["state"] == "partial" else "Install",
                            icon='IMPORT')
        op.tags = ",".join(tags)


def _wrapped(layout, text, width=96):
    line = ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            layout.label(text=line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        layout.label(text=line)


def _draw_review(layout, review):
    inner = layout.box()
    titles = ", ".join(installer.package_info(t)["name"] for t in review["tags"])
    inner.label(text=f"Install {titles}", icon='IMPORT')
    plan = review.get("plan")
    if plan is None:
        inner.label(text="Checking what this changes in your ComfyUI…", icon='SORTTIME')
    elif "error" in plan:
        inner.label(text=plan.get("message", "Could not check this ComfyUI"), icon='ERROR')
    elif runtime.needs_review(plan):
        col = inner.column(align=True)
        col.label(text="Changes to your ComfyUI's Python:", icon='ERROR')
        for pip in plan.get("pip", []):
            installed = f" (you have {pip['installed']})" if pip.get("installed") else " (new)"
            col.label(text=f"{pip['spec']}{installed}", icon='BLANK1')
        for step in plan.get("steps", []):
            col.label(text=f"{step['node']}: {step['step']}", icon='BLANK1')
        for node in plan.get("nodes", []):
            if node["action"] in ("move to pin", "different commit"):
                col.label(text=f"{node['name']}: kept at your commit" if node["action"] == "different commit"
                          else f"{node['name']}: moved to the commit StableGen was tested with", icon='BLANK1')
        note = col.row()
        note.active = False
        note.label(text="Your package versions are saved first, to roll back from.", icon='INFO')
    for key in review["notices"]:
        col = inner.column(align=True)
        for line in installer.LICENSE_NOTICES.get(key, []):
            if line:
                _wrapped(col, line)
    row = _right(inner)
    row.enabled = plan is not None      # StableGen's own ComfyUI has an empty plan: nothing of the user's changes
    row.operator("stablegen.comfy_cancel_review", text="Cancel")
    row.operator("stablegen.comfy_confirm_install",
                 text=runtime.accept_label(review), icon='CHECKMARK')


# --- the sidebar ---
SETUP_CARD = 'setup_card'


HINT_OPERATORS = ("stablegen.comfy_install", "stablegen.comfy_cancel_install", "stablegen.comfy_copy_install_command",
                  "stablegen.comfy_use_server_folder", "stablegen.comfy_unqueue")
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


def model_hint(context):
    """When the chosen architecture's model isn't on the ComfyUI StableGen uses, what to say and offer, else None:
        kind 'missing'      a local ComfyUI without it: Install, with its download size
        kind 'installing'   its install is running: progress and detail ('5.1 of 14.3 GB · 42 MB/s · 4 min left')
        kind 'queued'       it waits for the install that runs: where in the queue ('next', '2nd in line')
        kind 'server'       a server without it: run installer.py there (copy the command), or for a ComfyUI on
                            this machine, use its folder so StableGen can install it
    A local ComfyUI is read from its status report; a server from the model list StableGen already fetched."""
    p = runtime.prefs()
    architecture = context.scene.architecture_mode
    if runtime.local_mode(p):
        folder = runtime.comfy_folder(p)
        report = runtime.ensure_report(folder)
        missing = packages.missing_for_architecture(architecture, report)
        if not missing:
            return None
        tags, title = missing
        job = runtime.job
        if job is not None and job.running and set(job.tags) & set(tags):
            rate = job.rate_text()
            return {'kind': 'installing', 'tags': tags, 'title': title, 'text': f"Installing {title}",
                    'progress': job.progress(), 'detail': job.size_text() + (f"  ·  {rate}" if rate else "")}
        place = runtime.queued(list(tags))
        if place is not None:
            return {'kind': 'queued', 'tags': tags, 'title': title, 'text': f"{title} is queued",
                    'detail': packages.queue_text(place)}
        size = packages.size_text(packages.row_state(tags, report)["download_mb"])
        return {'kind': 'missing', 'tags': tags, 'title': title, 'text': f"{title} isn't installed",
                'button': f"Install · {size}", 'size': size}
    tags = packages.ARCH_TAGS.get(architecture)
    if not p.server_online or not tags or architecture == 'trellis2':    # TRELLIS.2 has no model list to check
        return None
    from ..core import state
    if context.scene.model_name != 'NONE_FOUND' or state._cached_checkpoint_architecture != architecture:
        return None
    title = next((t for _, row_tags, t, _ in packages.ROWS if set(tags) <= set(row_tags)), architecture)
    local = p.server_address.split("//")[-1].rsplit(":", 1)[0].lower() in LOOPBACK
    return {'kind': 'server', 'tags': tags, 'title': title, 'local': local,
            'text': f"{title} isn't on your ComfyUI" if local else f"{title} isn't on the server",
            'button': "Set up" if local else "Copy command"}


def draw_panel_notices(layout, context):
    """The sidebar's part of the setup: a first-run card that opens these preferences, and a note with an Install
    button when the chosen model is not on the ComfyUI. Top-level boxes with an INFO or ERROR label, so the
    StableGen panel shows them on every page (it draws the model note itself, under Architecture)."""
    from ..ui import studio_tips
    p = runtime.prefs()
    if (p.comfy_mode == 'REMOTE' and not p.server_online and p.server_address == "127.0.0.1:8188"
            and not studio_tips.dismissed(SETUP_CARD)):
        box = layout.box()
        box.label(text="ComfyUI isn't set up yet", icon='INFO')
        box.label(text="StableGen can install its own, use yours, or connect to a server.")
        row = box.row(align=True)
        row.operator("stablegen.open_setup", text="Set up ComfyUI", icon='PREFERENCES')
        row.operator("stablegen.dismiss_setup_card", text="I have a server")
        return
    hint = model_hint(context)
    if hint is None:
        return
    box = layout.box()
    tags = ",".join(hint['tags'])
    if hint['kind'] == 'installing':
        # a label, not a progress bar: the panel's progress bars are generation progress
        row = box.row(align=True)
        row.label(text=f"{hint['text']} · {round(hint['progress'] * 100)}%  ({hint['detail']})", icon='INFO')
        row.operator("stablegen.comfy_cancel_install", text="", icon='X')
    elif hint['kind'] == 'queued':
        row = box.row(align=True)
        row.label(text=f"{hint['text']} ({hint['detail']})", icon='INFO')
        row.operator("stablegen.comfy_unqueue", text="", icon='X').tags = tags
    elif hint['kind'] == 'missing':
        box.label(text=hint['text'] + " in your ComfyUI", icon='ERROR')
        box.operator("stablegen.comfy_install", text=hint['button'], icon='IMPORT').tags = tags
    else:
        box.label(text=hint['text'], icon='ERROR')
        if hint['local']:
            box.operator("stablegen.comfy_use_server_folder", text="Let StableGen install it", icon='IMPORT')
        else:
            box.operator("stablegen.comfy_copy_install_command", text="Copy the install command",
                         icon='COPYDOWN').tags = tags
