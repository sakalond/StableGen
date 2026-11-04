"""The ComfyUI setup page of the StableGen panel (layout B of ref/setup-ui-20261103).

Where ComfyUI comes from, installing StableGen's own, its status with Start and Stop, and the models with their
install progress. It draws the state of stablegen/setup/runtime.py in the panel's own style and runs the same
operators as the preferences section, which stays as the place for the port and extra arguments.
"""
import bpy

from . import studio_cards as cards
from . import studio_tips as tips
from .studio_text import wrap
from .studio_widgets import CONTROL_GAP, CONTROL_HEIGHT, FONT_SIZE, font_size
from ..setup import installer, packages, prefs_ui, runtime
from ..setup.jobs import InstallerJob

OPENED = 'setup_page_opened'        # the page opens by itself once, on the first run
MODES = (
    ('OWN', "Install StableGen's own", "Recommended",
     "About 5 GB. Nothing else to install, and it starts and stops by itself."),
    ('LOCAL', "Use my ComfyUI folder", "", "Adds StableGen's nodes and models to your install."),
    ('REMOTE', "Connect to a server", "", "ComfyUI on another machine or a cloud GPU."),
)
MODE_NAMES = {'OWN': "StableGen's own", 'LOCAL': "Your ComfyUI folder", 'REMOTE': "Remote server"}
STATES = {'stopped': ("Stopped", 'muted'), 'starting': ("Starting…", 'accent'), 'running': ("Running", 'ok'),
          'exited': ("Stopped unexpectedly", 'warning'), 'unresponsive': ("Not answering", 'warning')}
ROW_ICONS = {"trellis2": 'cube', "pbr_marigold": 'layers', "pbr_marigold_v2": 'layers', "skintokens": 'rig',
             "lato2": 'modifier'}
# Shown at once; the rest sit under "More"
PRIMARY = {"core", "qwen_image_21", "qwen_core", "trellis2", "pbr_marigold", "skintokens", "lato2"}
ROW_HEIGHT = 40


def needed(prefs=None) -> bool:
    """ComfyUI is not set up: nothing configured, or the folder of a local mode is missing."""
    p = prefs or runtime.prefs()
    if p.comfy_mode == 'REMOTE':
        return not p.server_online and p.server_address == "127.0.0.1:8188"
    return runtime.comfy_folder(p) is None


def open_first_run(session):
    """Show this page once when ComfyUI is not set up yet."""
    if session.page != 'SETUP' and not tips.dismissed(OPENED) and needed():
        tips.dismiss(OPENED)
        session.page = 'SETUP'


def draw(session, p, x, y, width):
    prefs = runtime.prefs()
    folder = runtime.comfy_folder(prefs)
    y = _card(p, 'ComfyUI', 'server', x, y, width,
              lambda cx, cy, cw: _comfy(session, p, prefs, folder, cx, cy, cw))
    review = runtime.review
    if review is not None and folder is not None and review["folder"] == str(folder):
        y = _card(p, 'Install ' + ", ".join(installer.package_info(t)["name"] for t in review["tags"]), 'download',
                  x, y + 12, width, lambda cx, cy, cw: _review(session, p, review, cx, cy, cw))
    if folder is not None:
        report = runtime.ensure_report(folder)
        aside = f"{report['free_mb'] / 1024:.0f} GB free" if report and "free_mb" in report else ""
        y = _card(p, 'Models', 'layers', x, y + 12, width,
                  lambda cx, cy, cw: _models(session, p, prefs, folder, report, cx, cy, cw), aside=aside,
                  refresh=lambda: session.call('stablegen.comfy_refresh'))
    return y


# --- pieces ---
def _card(p, title, icon, x, y, width, body, aside='', refresh=None):
    start = y
    p.begin_group()
    y = cards.heading(p, title, x + 12, y + 11, width - 24, icon)
    right = x + width - 12
    if refresh:
        p.button('setup-refresh', (right - 22, y - 26, 22, CONTROL_HEIGHT), '', refresh, icon='reset',
                 hint='Read again which models are installed')
        right -= 28
    if aside:
        p.text(aside, right - p.measure(aside, 10), y - 22, 10, 'muted')
    p.rect(x + 12, y - 3, width - 24, 1, 'divider')
    y = body(x + 12, y + 6, width - 24) + 8
    p.end_group((x, start, width, y - start), 'subbox', 'section_border', 3)
    return y


def _note(p, text, x, y, width, tint='secondary'):
    for _, _, line in wrap(text, width, lambda value: p.measure(value, 10)):
        p.text(line, x, y, 10, tint)
        y += 14
    return y


def _field(session, p, key, label, value, x, y, width, action, *, icon='folder', hint=''):
    """A read-only field whose value changes through a dialog (a folder picker, an address)."""
    p.text(label, x, y, FONT_SIZE, 'text', width)
    y += 17
    hover = session.hovered(key)
    p.outline(x, y, width, CONTROL_HEIGHT, 'field', 'hover_border' if hover else 'line', 4)
    p.text(p.fit(value or 'Not set', width - 40), x + 8, y + 4, FONT_SIZE, 'field_text' if value else 'muted',
           width - 40)
    p.icon(icon, x + width - 22, y + 4, 14, 'secondary')
    p.target(key, (x, y, width, CONTROL_HEIGHT), action, True, hint)
    return y + CONTROL_HEIGHT + CONTROL_GAP + 2


def _option(session, p, key, title, tag, description, selected, x, y, width, action):
    """One way to get ComfyUI: a radio row with a title and a line of explanation."""
    lines = wrap(description, width - 44, lambda value: p.measure(value, 10))
    height = 28 + 14 * len(lines)
    hover = session.hovered(key)
    p.outline(x, y, width, height, 'selected' if selected else 'hover_soft' if hover else 'card',
              'accent' if selected else 'hover_border' if hover else 'line', 5)
    cy = y + 10
    p.rect(x + 11, cy + 1, 12, 12, 'accent' if selected else 'secondary', 6)
    p.rect(x + 12, cy + 2, 10, 10, 'selected' if selected else 'card', 5)
    if selected:
        p.rect(x + 14, cy + 4, 6, 6, 'accent', 3)
    p.text(title, x + 32, cy, FONT_SIZE, 'text', width - 44)
    if tag:
        p.text(tag, x + 32 + p.measure(title) + 8, cy + 1, 10, 'accent')
    for i, (_, _, line) in enumerate(lines):
        p.text(line, x + 32, cy + 17 + i * 14, 10, 'secondary')
    p.target(key, (x, y, width, height), action, True, '')
    return y + height + 6


def _status(session, p, prefs, x, y, width):
    c = runtime.comfy()
    text, tint = STATES.get(c.state, (c.state, 'muted'))
    detail = ""
    if c.state in ('running', 'starting'):
        detail = c.address + ("" if c.owned else " · started outside StableGen")
        report = runtime.report(runtime.comfy_folder(prefs))
        gpu = ((report or {}).get("python") or {}).get("gpu")
        if gpu and c.state == 'running':
            detail += " · " + gpu.replace("NVIDIA GeForce ", "")
    elif c.message:
        detail = c.message
    height = 30
    p.outline(x, y, width, height, 'card', 'line', 5)
    p.rect(x + 10, y + 12, 7, 7, tint, 3.5)
    p.text(text, x + 24, y + 8, FONT_SIZE, 'text')
    button_width = 58
    p.text(p.fit(detail, width - 40 - p.measure(text) - button_width - 30, 10), x + 30 + p.measure(text), y + 9, 10,
           'muted')
    bx = x + width - button_width - 4
    if c.state in ('running', 'starting', 'unresponsive'):
        p.button('setup-stop', (bx, y + 4, button_width, CONTROL_HEIGHT), 'Stop',
                 lambda: session.call('stablegen.comfy_stop'), icon='stop', style='outlined',
                 hint='Stop ComfyUI' if c.owned else 'Disconnect; StableGen did not start this ComfyUI')
    else:
        p.button('setup-start', (bx, y + 4, button_width, CONTROL_HEIGHT), 'Restart' if c.state == 'exited' else 'Start',
                 lambda: session.call('stablegen.comfy_start'), icon='play', style='outlined',
                 hint='Start ComfyUI in the background')
    y += height + 4
    folder = runtime.comfy_folder(prefs)
    if runtime.restart_needed and folder is not None and runtime.restart_needed == str(folder) and not c.owned:
        y = _note(p, "Restart ComfyUI so it loads the nodes just installed. StableGen didn't start it, so it won't "
                     "restart it.", x, y + 2, width, 'warning') + 6
    if c.state in ('exited', 'unresponsive'):
        p.button('setup-log', (x, y, 80, CONTROL_HEIGHT), 'Show log',
                 lambda: session.call('stablegen.comfy_open_log', {'which': 'COMFYUI'}), icon='document')
        y += CONTROL_HEIGHT + 2
    for key, prop, label in (('setup-auto', 'comfy_auto_start', 'Start when needed, stop with Blender'),
                             ('setup-keep', 'comfy_keep_running', 'Keep running after Blender closes')):
        p.switch(key, (x, y, width, CONTROL_HEIGHT), label, getattr(prefs, prop),
                 lambda prop=prop: _toggle(session, prefs, prop), hover=session.hovered(key))
        y += CONTROL_HEIGHT + CONTROL_GAP
    return y


def _toggle(session, prefs, prop):
    setattr(prefs, prop, not getattr(prefs, prop))
    session.area.tag_redraw()


def _set_mode(session, prefs, mode):
    prefs.comfy_mode = mode     # while changing, the choice stays open until Use, Install or Connect
    session.area.tag_redraw()


def _comfy(session, p, prefs, folder, x, y, width):
    job = runtime.job if runtime.job is not None and runtime.job.kind == "create" else None
    if job is not None and job.running:
        return _own_progress(session, p, job, x, y, width)
    configured = folder is not None or (prefs.comfy_mode == 'REMOTE' and not needed(prefs))
    if configured and not getattr(session, 'setup_change', False):
        where = (str(folder) if folder is not None else prefs.server_address)
        p.text(MODE_NAMES[prefs.comfy_mode], x, y, FONT_SIZE, 'text', width - 70)
        p.button('setup-change', (x + width - 64, y - 4, 64, CONTROL_HEIGHT), 'Change',
                 lambda: _change(session), style='quiet', hint='Choose where ComfyUI comes from')
        y = _note(p, where, x, y + 17, width - 70, 'muted') + 6
        python = (runtime.report(folder) or {}).get("python") or {} if folder is not None else {}
        if python.get("default"):       # no Python of its own: it runs with the python on PATH
            y = _note(p, f"Runs with the default python: {python.get('path')}", x, y - 4, width, 'muted') + 6
        if prefs.comfy_mode == 'OWN' and runtime.own_outdated(prefs):
            y = _note(p, f"This StableGen uses ComfyUI {installer.COMFYUI_TAG}.", x, y, width - 90, 'warning')
            p.button('setup-update', (x + width - 84, y - 18, 84, CONTROL_HEIGHT), 'Update',
                     lambda: session.call('stablegen.comfy_create'), icon='download', style='outlined')
            y += 6
        if prefs.comfy_mode == 'REMOTE':
            return _note(p, "Install models on that machine with installer.py from the StableGen repository.",
                         x, y, width)
        return _status(session, p, prefs, x, y, width)

    for mode, title, tag, description in MODES:
        y = _option(session, p, 'setup-mode:' + mode, title, tag, description, prefs.comfy_mode == mode, x, y, width,
                    lambda mode=mode: _set_mode(session, prefs, mode))
    y += 4
    if prefs.comfy_mode == 'OWN':
        y = _field(session, p, 'setup-root', 'Install folder', prefs.comfy_root, x, y, width,
                   lambda: session.call('stablegen.comfy_choose_folder', {'target': 'ROOT'}),
                   hint='Where StableGen keeps its own ComfyUI')
        if folder is not None:      # already installed there: use it, or repair it
            installed = f"Installed  ·  ComfyUI {runtime.own_marker(prefs).get('comfyui', '')}".rstrip()
            p.icon('check', x, y + 1, 12, 'ok')
            p.text(installed, x + 18, y, 10, 'ok', width - 110)
            p.button('setup-repair', (x + width - 84, y - 4, 84, CONTROL_HEIGHT), 'Repair',
                     lambda: session.call('stablegen.comfy_create'), icon='reset', style='quiet',
                     hint='Check every part and download what is missing or broken')
            return _use(session, p, "Use StableGen's own", x, y + 22, width)
        if prefs.comfy_root:
            space = _free_space(prefs.comfy_root)
            if space:
                y = _note(p, space, x, y - 2, width, 'muted') + 4
        if job is not None and not job.ok:
            y = _note(p, job.status_line(), x, y, width, 'warning') + 4
        p.button('setup-install', (x, y + 4, width, 28), 'Retry' if job is not None else 'Install ComfyUI',
                 lambda: session.call('stablegen.comfy_create'), icon='download', style='primary',
                 enabled=bool(prefs.comfy_root), hint='Download ComfyUI, Python and torch for your GPU')
        return y + 36
    if prefs.comfy_mode == 'LOCAL':
        y = _field(session, p, 'setup-local', 'ComfyUI folder', prefs.comfy_local_path, x, y, width,
                   lambda: session.call('stablegen.comfy_choose_folder', {'target': 'LOCAL'}),
                   hint="Your ComfyUI folder, with 'models' and 'custom_nodes'")
        if folder is not None:
            return _use(session, p, "Use this folder", x, y + 2, width)
        if prefs.comfy_local_path:
            y = _note(p, "No ComfyUI here: choose the folder with 'models' and 'custom_nodes'.", x, y, width,
                      'warning')
        return y
    y = _field(session, p, 'setup-address', 'Server address', prefs.server_address, x, y, width,
               lambda: session.call('stablegen.comfy_set_address'), icon='rename',
               hint='Address and port of the ComfyUI server')
    p.button('setup-connect', (x, y + 2, width, 26), 'Connect', lambda: _connect(session),
             icon='reset', style='outlined')
    return y + 32


def _use(session, p, label, x, y, width):
    """Leave the choices for the status of the ComfyUI just chosen."""
    p.button('setup-use', (x, y + 4, width, 28), label, lambda: _done(session), icon='check', style='primary')
    return y + 36


def _done(session):
    session.setup_change = False
    error = runtime.follow_folder()     # the running ComfyUI was for the folder chosen before
    if error:
        session.tell(error)
    session.area.tag_redraw()


def _connect(session):
    session.call('stablegen.check_server_status')
    _done(session)


def _change(session):
    session.setup_change = True         # trying the choices restarts nothing until Use (runtime.choosing)
    session.area.tag_redraw()


def _own_progress(session, p, job, x, y, width):
    p.text('Installing ComfyUI', x, y, FONT_SIZE, 'text')
    pct = f"{job.progress():.0%}"
    p.text(pct, x + width - p.measure(pct, 10), y + 1, 10, 'muted')
    y += 22
    for step_id in job.order:
        item = job.items[step_id]
        done = item.get("status") in ('installed', 'present')
        current = step_id == job.current
        failed = item.get("status") == 'failed'
        p.icon('check' if done else 'x' if failed else 'download' if current else 'minus', x, y, 13,
               'ok' if done else 'warning' if failed else 'accent' if current else 'muted')
        p.text(item["name"], x + 20, y, FONT_SIZE, 'secondary' if done else 'text' if current else 'muted',
               width - 110)
        if current and item.get("total"):
            size = InstallerJob.bytes_text(item['done'], item['total'])
        else:
            size = packages.size_text(item.get("size_mb", 0)) if item.get("size_mb") else ""
        p.text(size, x + width - p.measure(size, 10), y + 1, 10, 'muted')
        y += 18
        if current and item.get("total"):
            p.rect(x + 20, y - 1, width - 20, 3, 'line', 1)
            p.rect(x + 20, y - 1, (width - 20) * min(1, item["done"] / item["total"]), 3, 'blue', 1)
            y += 6
    y += 4
    y = _note(p, "You can keep working while it installs.", x, y + 4, width - 90, 'muted')
    p.button('setup-cancel', (x + width - 80, y - 18, 80, CONTROL_HEIGHT), 'Cancel',
             lambda: session.call('stablegen.comfy_cancel_install'), icon='x', style='outlined')
    return y + 8


def _models(session, p, prefs, folder, report, x, y, width):
    if report is None or report.get("loading"):
        return _note(p, "Reading which models are installed…", x, y, width, 'muted')
    if "error" in report:
        return _note(p, report.get("message", "Could not read this ComfyUI"), x, y, width, 'warning')
    job = runtime.job
    if job is not None and job.kind == "install" and job.folder == str(folder) and not job.running and not job.ok:
        y = _note(p, f"{job.title}: {job.status_line()}", x, y, width - 90, 'warning')
        p.button('setup-install-log', (x + width - 84, y - 18, 84, CONTROL_HEIGHT), 'Show log',
                 lambda: session.call('stablegen.comfy_open_log', {'which': 'INSTALL'}), icon='document')
        y += 8
    rows = list(packages.visible_rows(prefs.enable_deprecated_architectures, prefs.enable_experimental_architectures))
    more = getattr(session, 'setup_more', False)
    shown = [r for r in rows if more or PRIMARY & set(r[1])]
    hidden = [r for r in rows if not PRIMARY & set(r[1])]
    group = None
    for row_group, tags, title, purpose in shown:
        if row_group != group:
            group = row_group
            p.text(group, x, y + (4 if y else 0), 10, 'muted')
            y += 22
        y = _model_row(session, p, tags, title, purpose, packages.row_state(tags, report), job, x, y, width)
    if hidden:
        label = 'Fewer models' if more else f'More  ·  {len(hidden)} models'
        p.button('setup-more', (x, y + 4, width, CONTROL_HEIGHT), label,
                 lambda: _more(session), icon='up' if more else 'down', style='quiet')
        y += CONTROL_HEIGHT + 8
    return y


def _more(session):
    session.setup_more = not getattr(session, 'setup_more', False)
    session.area.tag_redraw()


def _model_row(session, p, tags, title, purpose, state, job, x, y, width):
    installing = job is not None and job.running and set(job.tags) & set(tags)
    place = runtime.queued(list(tags))
    p.rect(x, y - 4, width, 1, 'divider')
    p.icon(ROW_ICONS.get(tags[0], 'image'), x, y + 6, 14, 'secondary')
    tx, right = x + 24, 86
    p.text(p.fit(title, width - 24 - right), tx, y + 2, FONT_SIZE, 'text')
    joined = ",".join(tags)
    if installing:
        rate = job.rate_text()
        meta = f"{job.progress():.0%}  ·  {job.size_text()}" + (f"  ·  {rate}" if rate else "")
        p.text(p.fit(meta, width - 24 - 32, 10), tx, y + 18, 10, 'accent')
        p.rect(tx, y + 32, width - 24 - 32, 3, 'line', 1)
        p.rect(tx, y + 32, (width - 24 - 32) * job.progress(), 3, 'blue', 1)
        p.button('setup-cancel-install', (x + width - 26, y + 6, 26, CONTROL_HEIGHT), '',
                 lambda: session.call('stablegen.comfy_cancel_install'), icon='x',
                 hint='Cancel this install; finished files stay, downloads resume next time, and the queue goes on')
        return y + ROW_HEIGHT
    if place is not None:
        p.text(f"Queued  ·  {packages.queue_text(place)}", tx, y + 18, 10, 'accent')
        p.button('setup-unqueue:' + tags[0], (x + width - 26, y + 6, 26, CONTROL_HEIGHT), '',
                 lambda: session.call('stablegen.comfy_unqueue', {'tags': joined}), icon='x',
                 hint='Take it out of the queue')
        return y + ROW_HEIGHT
    size = state["size_mb"] if state["state"] == "installed" else state["download_mb"]
    creating = job is not None and job.running and job.kind == "create"
    queueing = job is not None and job.running and not creating
    trash = 28 if packages.removable_state(tags, state, runtime.node_policy()) else 0
    p.text(p.fit(packages.meta_text(purpose, tags, size), width - 24 - right - trash, 10), tx, y + 18, 10, 'muted')
    if state["missing_core_nodes"]:
        p.text("Needs a newer ComfyUI", tx, y + 31, 10, 'warning')
    if state["state"] == "installed":
        p.icon('check', x + width - 74 - trash, y + 7, 12, 'ok')
        p.text('Installed', x + width - 58 - trash, y + 6, 10, 'ok')
    elif state["state"] != "unknown":
        p.button('setup-install:' + tags[0], (x + width - 80, y + 4, 80, CONTROL_HEIGHT),
                 'Finish' if state["state"] == "partial" else 'Install',
                 lambda: session.call('stablegen.comfy_install', {'tags': joined}), icon='download',
                 style='outlined', enabled=not creating,
                 hint="StableGen's own ComfyUI is still installing" if creating else
                 f'Add it to the queue ({packages.size_text(state["download_mb"])})' if queueing else
                 f'Download {packages.size_text(state["download_mb"])} into ComfyUI')
    if trash:
        tx_trash = x + width - 24 if state["state"] == "installed" else x + width - 80 - trash
        p.button('setup-remove:' + tags[0], (tx_trash, y + 4, 24, CONTROL_HEIGHT), '',
                 lambda: session.call('stablegen.comfy_remove', {'tags': joined, 'title': title}), icon='trash',
                 enabled=not creating, hint="Remove it: delete its models and unfinished downloads" + (
                     " and its nodes" if runtime.node_policy() == "all" else
                     "; nodes from elsewhere stay, other workflows may use them")
                 + ". What another installed package uses stays")
    return y + ROW_HEIGHT + (10 if state["missing_core_nodes"] else 0)


def _review(session, p, review, x, y, width):
    plan = review.get("plan")
    if plan is None:
        y = _note(p, "Checking what this changes in your ComfyUI…", x, y, width, 'muted') + 4
    elif "error" in plan:
        y = _note(p, plan.get("message", "Could not check this ComfyUI"), x, y, width, 'warning') + 4
    elif runtime.needs_review(plan):
        p.text("Changes your ComfyUI's Python", x, y, 10, 'warning')
        y += 16
        lines = [f"{pip['spec']}" + (f"  (you have {pip['installed']})" if pip.get("installed") else "  (new)")
                 for pip in plan.get("pip", [])]
        lines += [f"{s['node']}: {s['step']}" for s in plan.get("steps", [])]
        lines += [f"{n['name']}: moved to the tested commit" for n in plan.get("nodes", [])
                  if n["action"] == "move to pin"]
        for line in lines:
            p.rect(x, y, 2, 14, 'warning', 1)
            p.text(p.fit(line, width - 12, 10), x + 10, y, 10, 'secondary')
            y += 16
        y = _note(p, "Your package versions are saved first, to roll back from.", x, y + 4, width, 'muted') + 6
    for key in review["notices"]:
        y = _notice(p, installer.LICENSE_NOTICES.get(key, []), x, y, width) + 6
    ready = plan is not None
    button = 132 if review["notices"] else 80
    p.button('setup-review-cancel', (x + width - button - 88, y + 4, 80, 26), 'Cancel',
             lambda: session.call('stablegen.comfy_cancel_review'), style='outlined')
    p.button('setup-review-accept', (x + width - button, y + 4, button, 26),
             runtime.accept_label(review),
             lambda: session.call('stablegen.comfy_confirm_install'), icon='check', style='primary', enabled=ready)
    return y + 34


def _notice(p, lines, x, y, width):
    """A license notice: the CLI wraps its lines by hand, so each paragraph is joined and wrapped to the card. The
    first line is its title, which the card already says."""
    paragraphs, current = [], []
    for line in lines[1:]:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    for text in paragraphs:
        y = _note(p, text, x, y, width, 'secondary') + 6
    return y


HINT_HEIGHT = 28        # the tab switcher's: the line is drawn like its selected tab
HINT_ACTIONS = {    # what the button of a missing model does: icon, operator, hint
    'missing': ('download', 'stablegen.comfy_install', 'Download it into your ComfyUI; you can keep working'),
    'server-local': ('download', 'stablegen.comfy_use_server_folder',
                     "Use this ComfyUI's folder, so StableGen can install models into it"),
    'server': ('clone', 'stablegen.comfy_copy_install_command',
               'Copy the installer.py command that installs it; run it on the server'),
}


def draw_model_hint(session, p, x, y, width):
    """The line under Architecture when the chosen model isn't on the ComfyUI, drawn like the tab switcher's
    selected tab: amber with Install as its right segment, then blue while it waits in the queue and with the
    download's progress along its bottom. The Setup page lists the models instead."""
    hint = prefs_ui.model_hint(bpy.context) if session.page != 'SETUP' else None
    if hint is None:
        return y
    h = HINT_HEIGHT
    text_y = y + (h - font_size(FONT_SIZE)) / 2 - 1
    if hint['kind'] in ('installing', 'queued'):
        installing = hint['kind'] == 'installing'
        p.outline(x, y, width, h, 'tab_selected', 'accent', 4)
        p.icon('download', x + 9, y + 7, 13, 'accent')
        detail = p.fit(hint['detail'], width * .55, 10)
        detail_x = x + width - 32 - p.measure(detail, 10)
        p.text(p.fit(hint['text'], detail_x - x - 40), x + 30, text_y, FONT_SIZE, 'accent')
        p.text(detail, detail_x, text_y + 1, 10, 'secondary')
        if installing:
            p.rect(x + 1, y + h - 3, (width - 2) * min(1, hint['progress']), 2, 'accent')
            p.button('hint-cancel', (x + width - 27, y + 3, 24, h - 6), '',
                     lambda: session.call('stablegen.comfy_cancel_install'), icon='x',
                     hint='Cancel the install; finished files stay, and downloads resume next time')
        else:
            p.button('hint-cancel', (x + width - 27, y + 3, 24, h - 6), '',
                     lambda: session.call('stablegen.comfy_unqueue', {'tags': ",".join(hint['tags'])}), icon='x',
                     hint='Take it out of the queue')
        return y + h + 8
    kind = 'server-local' if hint['kind'] == 'server' and hint['local'] else hint['kind']
    icon, operator, tip = HINT_ACTIONS[kind]
    properties = {} if kind == 'server-local' else {'tags': ",".join(hint['tags'])}
    running = runtime.job if runtime.job is not None and runtime.job.running else None
    enabled = not (kind == 'missing' and running is not None and running.kind == "create")
    if kind == 'missing' and running is not None and enabled:
        tip = 'Add it to the queue: it downloads after the install that runs'
    label = hint['button']
    segment = min(p.measure(label) + 40, width * .5)
    sx = x + width - segment
    p.outline(x, y, width, h, 'warning_tab', 'warning', 4)
    if enabled and session.hovered('hint-action'):
        p.rect(sx, y + 1, segment - 1, h - 2, 'warning_strong', 3, corners='RIGHT')
    p.rect(sx, y + 1, 1, h - 2, 'warning')         # divided like the switcher's tabs
    p.icon('warning', x + 9, y + 7, 13, 'warning')
    p.text(p.fit(hint['text'], sx - x - 40), x + 30, text_y, FONT_SIZE, 'warning')
    tint = 'warning' if enabled else 'disabled'
    label_x = sx + (segment - p.measure(label) - 17) / 2
    p.icon(icon, label_x, y + 7, 13, tint)
    p.text(label, label_x + 17, text_y, FONT_SIZE, tint)
    p.target('hint-action', (sx, y, segment, h), lambda: session.call(operator, properties), enabled,
             tip if enabled else "StableGen's own ComfyUI is still installing")
    return y + h + 8


def _free_space(path_text):
    from ..setup.prefs_ui import _free_space_text
    return _free_space_text(path_text)


class SG_OT_OpenSetup(bpy.types.Operator):
    """Open the ComfyUI setup: the Setup page of the StableGen panel, or the preferences with the legacy panel"""
    bl_idname = 'stablegen.open_setup'
    bl_label = 'Set up ComfyUI'
    bl_options = {'INTERNAL'}

    def execute(self, context):
        from . import studio
        sessions = [s for s in studio._sessions.values() if getattr(s, 'alive', True)]
        if not sessions:
            bpy.ops.preferences.addon_show('INVOKE_DEFAULT', module=runtime.ADDON_PKG)
            return {'FINISHED'}
        for session in sessions:
            session.navigate('SETUP')
        return {'FINISHED'}


CLASSES = (SG_OT_OpenSetup,)
