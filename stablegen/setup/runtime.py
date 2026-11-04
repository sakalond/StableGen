"""This Blender session's setup state: the local ComfyUI, the installer job, the last status report of each
ComfyUI folder and the install waiting for review. A timer redraws the preferences and sidebars while something
runs, notices ComfyUI coming up (then checks the server like the refresh button) or stopping, and refreshes the
package list after an install.

ComfyUI keeps running across add-on reloads (the next registration adopts it); Blender quitting stops it unless
"keep running" is on.
"""
import atexit
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import bpy  # pylint: disable=import-error

from ..core import ADDON_PKG
from . import installer
from .comfy_process import LocalComfyUI, python_for, same_folder
from .jobs import InstallerJob, run_report

TICK = 0.5
IDLE_TICK = 5.0
REPORT_TIMEOUT = 660        # jobs.run_report gives up after 600 s

_comfy: Optional[LocalComfyUI] = None
job: Optional[InstallerJob] = None
reports = {}                # str(folder) -> --status report, {"loading": True} while it runs
review: Optional[dict] = None   # the install waiting for its notices/changes to be accepted
queue: List[dict] = []      # accepted installs waiting for the running one, each as review had it
_restart_after_queue = False    # an install added nodes: restart ComfyUI once the queue is empty
restart_needed: Optional[str] = None    # the folder whose running ComfyUI (not StableGen's) must restart for new nodes
keep_running = False        # mirrors the preference, which atexit can no longer read
_last_state = "stopped"
_job_seen_done = True


def _run_async(work, done):
    """work() on a thread, then done(result) on the main thread. Not core.state's helper: that one drops results
    when the server address changes meanwhile, which starting ComfyUI does."""
    import threading
    box = {}

    def worker():
        try:
            box["result"] = work()
        except Exception as e:      # report it like a failed run
            box["result"] = {"error": "run", "message": str(e)}

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def poll():
        if thread.is_alive():
            return 0.25
        done(box.get("result"))
        return None

    # persistent: opening a .blend while it runs (the status read takes a while) must not drop the result
    bpy.app.timers.register(poll, first_interval=0.25, persistent=True)


# --- paths and preferences ---
def prefs():
    return bpy.context.preferences.addons[ADDON_PKG].preferences


def state_dir() -> Path:
    try:
        folder = bpy.utils.extension_path_user(ADDON_PKG, create=True)
    except ValueError:      # legacy add-on installs have no extension user folder
        folder = bpy.utils.user_resource('CONFIG', path=ADDON_PKG, create=True)
    return Path(folder) / "setup"


def resolve_folder(path: str) -> Optional[Path]:
    """The ComfyUI folder at path (also the outer folder of a portable build), or None."""
    if not path:
        return None
    base = Path(bpy.path.abspath(path)).expanduser()
    for candidate in (base, base / "ComfyUI"):
        if (candidate / "models").is_dir() and (candidate / "custom_nodes").is_dir():
            return candidate
    return None


def own_folder(p=None) -> Optional[Path]:
    p = p or prefs()
    return Path(bpy.path.abspath(p.comfy_root)).expanduser() / "ComfyUI" if p.comfy_root else None


def own_marker(p=None) -> dict:
    """What --create-comfyui recorded about StableGen's own ComfyUI (empty when there is none)."""
    folder = own_folder(p)
    try:
        return json.loads((folder.parent / installer.OWN_MARKER).read_text(encoding="utf-8")) if folder else {}
    except (OSError, ValueError):
        return {}


def own_ready(p=None) -> bool:
    """Every step of --create-comfyui finished (a half-made one has main.py but no torch yet)."""
    folder, marker = own_folder(p), own_marker(p)
    return bool(folder and all(marker.get(k) for k in ("uv", "python", "torch", "requirements"))
                and installer.own_python(folder).is_file())


def own_outdated(p=None) -> bool:
    """This StableGen pins a newer ComfyUI or torch than the own ComfyUI has."""
    marker = own_marker(p)
    return bool(marker) and (marker.get("comfyui") != installer.COMFYUI_TAG
                             or not str(marker.get("torch", "")).startswith(" ".join(installer.TORCH_PINS)))


def comfy_folder(p=None) -> Optional[Path]:
    """The ComfyUI StableGen runs and installs into, in the Own and Local modes."""
    p = p or prefs()
    if p.comfy_mode == 'LOCAL':
        return resolve_folder(p.comfy_local_path)
    if p.comfy_mode == 'OWN':
        return own_folder(p) if own_ready(p) else None
    return None


def local_mode(p=None) -> bool:
    return (p or prefs()).comfy_mode in ('LOCAL', 'OWN')


def comfy() -> LocalComfyUI:
    global _comfy
    if _comfy is None:
        _comfy = LocalComfyUI(state_dir())
        _comfy.adopt_recorded()
    return _comfy


# --- ComfyUI ---
def start(p=None) -> str:
    """Start (or adopt) the local ComfyUI. Returns an error message, or '' when it is starting."""
    p = p or prefs()
    folder = comfy_folder(p)
    if folder is None:
        return "Choose the ComfyUI folder in StableGen's preferences first"
    try:
        address = comfy().start(folder, port=p.comfy_port, extra_args=shlex.split(p.comfy_extra_args),
                                env=own_environment(p))
    except (RuntimeError, OSError, ValueError) as e:
        return str(e)
    if p.server_address != address:
        p.server_address = address
    ensure_timer()
    return ""


def stop():
    comfy().stop()
    redraw()


def own_git(p) -> Optional[Path]:
    """The Git that StableGen's own ComfyUI was set up with (Windows has none of its own)."""
    if p.comfy_mode != 'OWN' or not p.comfy_root:
        return None
    exe = Path(bpy.path.abspath(p.comfy_root)) / "tools" / "git" / "cmd" / (
        "git.exe" if sys.platform == "win32" else "git")
    return exe if exe.is_file() else None


def own_environment(p) -> Optional[dict]:
    """Git on PATH for StableGen's own ComfyUI (ComfyUI-Manager and node installs use it)."""
    git = own_git(p)
    return {"PATH": str(git.parent) + os.pathsep + os.environ.get("PATH", "")} if git else None


def choosing() -> bool:
    """The StableGen panel's Setup page is open on its choices (Change): trying them restarts nothing until Use."""
    from ..ui import studio
    return any(getattr(s, 'setup_change', False) and s.page == 'SETUP' for s in studio._sessions.values())


def follow_folder(p=None) -> str:
    """After the mode or folder changes: the ComfyUI StableGen runs (or adopted) for another folder is replaced by
    the chosen folder's, so the model lists come from the ComfyUI the setup shows. Nothing changes while it is
    stopped, or when no other local folder is chosen yet (a server, or a folder still to pick). Returns an error
    or ''."""
    p = p or prefs()
    c = comfy()
    folder = comfy_folder(p) if local_mode(p) else None
    if folder is None or c.folder is None or c.state not in ('starting', 'running', 'unresponsive') or choosing():
        return ""
    if same_folder(c.folder, folder):
        return ""
    print(f"[StableGen] Switching from the ComfyUI in {c.folder} to the one in {folder}.")
    p.server_online = False
    error = start(p)            # stops the other one (or lets it go, if StableGen didn't start it)
    redraw()
    return error


def maybe_autostart() -> bool:
    """Start the local ComfyUI because StableGen needs it. Only from 'stopped': after a crash the user restarts it."""
    p = prefs()
    follow_folder(p)
    if not (local_mode(p) and p.comfy_auto_start and comfy().state == 'stopped' and comfy_folder(p)):
        return False
    if job and job.running:
        return False
    return not start(p)


def offline_note() -> Optional[str]:
    """What to tell an operator that needs the server while the local ComfyUI is not up yet."""
    p = prefs()
    if not local_mode(p):
        return None
    state = comfy().state
    if state == 'starting':
        return "ComfyUI is still starting; try again in a moment"
    if maybe_autostart():
        return "Starting ComfyUI; try again when the status line says it is running"
    if state == 'exited':
        return f"ComfyUI stopped ({comfy().message}); restart it from the StableGen panel"
    return None


# --- reports and installs ---
def report(folder: Optional[Path]) -> Optional[dict]:
    return reports.get(str(folder)) if folder else None


def refresh_status(folder: Optional[Path] = None):
    """Read --status for the folder in the background (installed packages, Python, GPU)."""
    folder = folder or comfy_folder()
    if folder is None:
        return
    python, default = python_for(folder)
    if python is None:
        reports[str(folder)] = {"error": "python", "message": "No Python for this ComfyUI: it has none of its own "
                                                              "(python_embeded, venv or .venv) and there is no "
                                                              "python on PATH"}
        redraw()
        return
    reports[str(folder)] = {"loading": True, "since": time.time()}
    redraw()
    key = str(folder)

    def done(result):
        if result is not None and "python" in result:
            result["python"]["default"] = default       # the python on PATH, as the folder has none of its own
        _report_done(key, result)
    _run_async(lambda: run_report([key, "--status", "--python", python]), done)


def _report_done(key, result):
    reports[key] = result if result is not None else {"error": "run", "message": "the status check did not finish"}
    redraw()


def _unread(current: Optional[dict]) -> bool:
    """No report yet, or a read that never came back (it gives up after REPORT_TIMEOUT)."""
    return current is None or (current.get("loading", False) and time.time() - current.get("since", 0) > REPORT_TIMEOUT)


def ensure_report(folder: Optional[Path]) -> Optional[dict]:
    """The report for this folder, starting to read it when there is none yet: nothing may have read it since
    Blender started or the folder was chosen."""
    if folder is not None and _unread(report(folder)):
        refresh_status(folder)
    return report(folder)


def refresh_models(folder: Optional[Path] = None):
    """Read again which models are installed, from the files alone (in-process, without ComfyUI's Python), and
    keep the rest of the report. With no report yet, read all of it."""
    folder = folder or comfy_folder()
    if folder is None:
        return
    key = str(folder)
    current = reports.get(key)
    if current is None or "tags" not in current:
        if _unread(current) or not current.get("loading"):
            refresh_status(folder)
        return

    def done(result):
        if result is not None and "tags" in reports.get(key, {}):
            reports[key] = {**reports[key], **result}
            redraw()
    _run_async(lambda: installer.package_report(folder), done)


def notices_for(tags: List[str]) -> List[str]:
    return list(dict.fromkeys(installer.package_info(t)["notice"] for t in tags
                              if installer.package_info(t).get("notice")))


def begin_install(tags: List[str]):
    """Install these package tags, after the user accepts their license notices and, for a ComfyUI of their own,
    the changes to its Python (both shown in the preferences)."""
    global review
    p = prefs()
    folder = comfy_folder(p)
    python = python_for(folder)[0] if folder else None
    review = {"tags": list(tags), "folder": str(folder), "python": python, "notices": notices_for(tags),
              "plan": None if p.comfy_mode == 'LOCAL' else {}}
    if p.comfy_mode == 'LOCAL' and python:
        current = review
        _run_async(lambda: run_report([str(folder), "--plan", "--packages", ",".join(tags), "--python", python]),
                   lambda plan: _plan_done(current, plan))
    elif not review["notices"]:
        confirm_install()
    redraw()


def _plan_done(pending, plan):
    if review is not pending:
        return
    pending["plan"] = plan or {"error": "run", "message": "the check did not finish"}
    if not pending["notices"] and not needs_review(pending["plan"]):
        confirm_install()
    redraw()


def needs_review(plan: Optional[dict]) -> bool:
    """A plan worth showing: it changes ComfyUI's Python or moves a node someone may have updated."""
    if not plan or "error" in plan:
        return bool(plan)
    return bool(plan.get("changes_python") or any(n["action"] in ("move to pin", "different commit")
                                                  for n in plan.get("nodes", [])))


def cancel_review():
    global review
    review = None
    redraw()


def confirm_install() -> str:
    """Start the reviewed install, or queue it behind the running one. Returns an error message or ''."""
    global review
    pending, review = review, None
    if not pending:
        return ""
    if not bpy.app.online_access:
        return "Allow online access in Blender's preferences (System) to download models"
    if not pending["python"]:
        return "No Python for this ComfyUI: none of its own, and no python on PATH"
    if job and job.running:
        if job.kind == "create":
            return "StableGen's own ComfyUI is still installing"
        if queued(pending["tags"]) is None and not set(job.tags) & set(pending["tags"]):
            queue.append(pending)
        redraw()
        return ""
    return _start_install(pending)


def accept_label(pending: dict) -> str:
    """The review's button: 'Accept and install', or 'Queue' when another install runs."""
    verb = "queue" if job is not None and job.running else "install"
    return f"Accept and {verb}" if pending["notices"] else verb.capitalize()


def queued(tags: List[str]) -> Optional[int]:
    """Where these tags wait in the queue (0 is next), or None."""
    return next((i for i, entry in enumerate(queue) if set(entry["tags"]) & set(tags)), None)


def unqueue(tags: List[str]):
    queue[:] = [entry for entry in queue if not set(entry["tags"]) & set(tags)]
    redraw()


def _start_next():
    """The next queued install for the ComfyUI in use; ones for a folder no longer chosen are dropped."""
    folder = comfy_folder()
    while queue:
        pending = queue.pop(0)
        if folder is not None and pending["folder"] == str(folder) and not _start_install(pending):
            return True
    return False


def _start_install(pending: dict) -> str:
    global job, _job_seen_done
    folder = Path(pending["folder"])
    args = [str(folder), "--packages", ",".join(pending["tags"]), "--python", pending["python"]]
    if pending["notices"]:
        args += ["--accept-notices", ",".join(pending["notices"])]
    git = own_git(prefs())
    if git:
        args += ["--git", str(git)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if needs_review(pending.get("plan")):
        snapshot(pending["python"], pending.get("plan") or {}, state_dir() / "snapshots" / f"{stamp}.txt")
    titles = ", ".join(installer.package_info(t)["name"] for t in pending["tags"])
    job = InstallerJob(titles, args, state_dir() / "logs" / f"install-{stamp}-{pending['tags'][0]}.log",
                       folder=str(folder), tags=pending["tags"])
    job.start()
    _job_seen_done = False
    ensure_timer()
    redraw()
    return ""


def busy_tags() -> set:
    """The tags the running install or the queue will write."""
    tags = set(job.tags) if job is not None and job.running else set()
    for entry in queue:
        tags |= set(entry["tags"])
    return tags


def node_policy(p=None) -> str:
    """Which nodes removing a package takes (installer.removed_with_package): all of them in StableGen's own ComfyUI,
    only StableGen's own nodes in someone's own, where the others may predate StableGen."""
    return "all" if (p or prefs()).comfy_mode == 'OWN' else "stablegen"


def removable(tags: List[str]) -> List[Path]:
    """What removing these tags deletes (installer.removal_plan): their models, Hugging Face folders and unfinished
    downloads (and nodes, in StableGen's own ComfyUI), without the ones another package still uses."""
    folder = comfy_folder()
    return (installer.removal_plan(folder, list(tags), sorted(busy_tags()), nodes=node_policy())
            if folder is not None else [])


def kept_nodes(tags: List[str], removing: List[Path]) -> List[Path]:
    """The nodes of these tags on disk that removing them leaves (ones from elsewhere, in someone's own ComfyUI)."""
    folder = comfy_folder()
    if folder is None:
        return []
    targets = [installer.item_target(i, folder) for i in installer.DEPENDENCIES.values()
               if i["type"] in installer.NODE_TYPES and set(i["packages"]) & set(tags)]
    return [t for t in targets if t.exists() and t not in removing]


def path_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size


def remove(tags: List[str]):
    """Delete them; returns (bytes removed, paths that could not be deleted, such as a model ComfyUI has open).
    Not while an install or the queue includes them."""
    if busy_tags() & set(tags):
        return 0, []
    removed, failed = 0, []
    nodes = comfy_folder() / "custom_nodes"
    for path in removable(tags):
        try:
            size = path_size(path)
            if path.parent == nodes:
                # renamed first, so a node in use fails whole instead of half deleted; ComfyUI skips '.disabled'
                gone = path.with_name(f"{path.name}.removing-{int(time.time())}.disabled")
                path.rename(gone)
                shutil.rmtree(gone, ignore_errors=True)
                if gone.exists():
                    failed.append(str(gone))
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += size
        except OSError:
            failed.append(str(path))
    refresh_models()
    if prefs().server_online and bpy.ops.stablegen.refresh_checkpoint_list.poll():
        bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')     # the removed checkpoints leave the list
    return removed, failed


def create_own() -> str:
    """Install (or repair, or update) StableGen's own ComfyUI in the install folder. Returns an error or ''."""
    global job, _job_seen_done
    p = prefs()
    if not p.comfy_root:
        return "Choose the install folder first"
    if job and job.running:
        return "Another install is still running"
    if not bpy.app.online_access:
        return "Allow online access in Blender's preferences (System) to download ComfyUI"
    c = comfy()
    if c.owned and c.state in ('running', 'starting'):
        c.stop()                    # its Python is about to change
    root = Path(bpy.path.abspath(p.comfy_root)).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = InstallerJob("StableGen's own ComfyUI", ["--create-comfyui", str(root)],
                       state_dir() / "logs" / f"create-{stamp}.log", kind="create", folder=str(root / "ComfyUI"))
    job.start()
    _job_seen_done = False
    ensure_timer()
    redraw()
    return ""


def snapshot(python: str, plan: dict, path: Path):
    """Package versions and node commits before an install changes a ComfyUI of the user's, to roll back from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, timeout=120,
                                text=True, encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        freeze = f"# pip freeze failed: {e}\n"
    nodes = "".join(f"# {n['path']} at {n.get('head') or '?'}\n" for n in plan.get("nodes", []))
    path.write_text(f"# Before installing {', '.join(plan.get('packages', []))} into {plan.get('comfyui', '')}\n"
                    f"# Roll back with: \"{python}\" -m pip install -r \"{path}\"\n{nodes}{freeze}", encoding="utf-8")


def cancel_job():
    if job:
        job.cancel()
    redraw()


# --- timer and redraws ---
def redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in ('PREFERENCES', 'VIEW_3D'):
                area.tag_redraw()


def ensure_timer():
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=TICK, persistent=True)


def _tick():
    global _last_state, _job_seen_done, restart_needed
    try:
        c = comfy()
        state = c.poll() if c.state == 'running' else c.state
        if state != _last_state:
            if state != 'running':
                restart_needed = None       # it stopped or went away: the next start loads the new nodes
            if state == 'running':
                print(f"[StableGen] ComfyUI is running at {c.address}.")
                if bpy.ops.stablegen.check_server_status.poll():
                    bpy.ops.stablegen.check_server_status('INVOKE_DEFAULT')
            elif state == 'exited':
                print(f"[StableGen] {c.message}.")
            _last_state = state
            redraw()
        busy = state == 'starting' or (job is not None and job.running)
        if job is not None and not job.running and not _job_seen_done:
            _job_seen_done = True
            _install_finished(job)
        if busy:
            redraw()
            return TICK
        return IDLE_TICK if state == 'running' else None
    except Exception as e:      # a timer must not raise; it would be dropped
        print(f"[StableGen] Setup timer: {e}")
        return IDLE_TICK


def _install_finished(finished: InstallerJob):
    print(f"[StableGen] Install {'finished' if finished.ok else 'stopped'}: {finished.status_line()}")
    folder = Path(finished.folder)
    if finished.kind == "create":
        if finished.ok and comfy_folder() == folder:
            refresh_status(folder)
            if prefs().comfy_auto_start:
                start()
        redraw()
        return
    global _restart_after_queue, restart_needed
    _restart_after_queue |= bool(finished.summary and finished.summary.get("restart"))
    if _start_next():                   # cancelling one install leaves the rest of the queue to go on
        refresh_models(folder)          # the finished one shows as installed while the next downloads
        redraw()
        return
    refresh_status(folder)
    c = comfy()
    restart, _restart_after_queue = _restart_after_queue, False
    if restart and c.owned and c.state == 'running' and c.folder and str(c.folder) == str(folder):
        print("[StableGen] Restarting ComfyUI so it loads the new nodes.")
        c.stop()
        start()
    elif restart and c.state == 'running':
        # StableGen never stops a ComfyUI it didn't start: the Setup page asks for the restart instead
        restart_needed = str(folder)
        print("[StableGen] Restart ComfyUI so it loads the new nodes (StableGen did not start it).")
    if c.state == 'running' and bpy.ops.stablegen.check_server_status.poll():
        bpy.ops.stablegen.check_server_status('INVOKE_DEFAULT')     # new models in the lists
    redraw()


def _on_exit():
    c = _comfy
    if c is not None and c.owned and not keep_running:
        c.stop()
    if job is not None and job.running:
        job.cancel()


def _startup():
    global keep_running, _last_state
    try:
        p = prefs()
        keep_running = p.comfy_keep_running
        if local_mode(p):
            c = comfy()
            _last_state = c.state
            if c.state == 'running':
                if p.server_address != c.address:
                    p.server_address = c.address
                ensure_timer()
            folder = comfy_folder(p)
            if folder:
                refresh_status(folder)
            follow_folder(p)        # the recorded ComfyUI may be for the folder chosen before
    except Exception as e:
        print(f"[StableGen] Setup start-up: {e}")
    return None


def register():
    atexit.register(_on_exit)
    if not bpy.app.background:
        bpy.app.timers.register(_startup, first_interval=1.0, persistent=True)     # Blender's start-up file loads after this


def unregister():
    atexit.unregister(_on_exit)
    for timer in (_tick, _startup):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
