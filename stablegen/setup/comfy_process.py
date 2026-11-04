"""The local ComfyUI StableGen starts, adopts and stops (the Own and Local server modes).

One ComfyUI per Blender: started hidden on 127.0.0.1, logged to <state>/logs/comfyui.log and recorded in
<state>/comfyui.json, so the next Blender adopts it (after a crash, or when "keep running" is on) instead of
starting a second copy. A ComfyUI that answers for the same folder is adopted, a server for another folder is left
alone and the next free port is used. StableGen only ever stops a ComfyUI it started.

No bpy here: the add-on drives it from timers and operators, and the tests run it with a fake ComfyUI.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import installer

READY_TIMEOUT = 15 * 60     # the first start after installing nodes can take minutes
PORT_SPAN = 20              # ports tried after the preferred one
LOGS_KEPT = 3


# --- Looking at servers and processes ---
def _get_json(address: str, path: str, timeout: float):
    with urllib.request.urlopen(f"http://{address}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def answers(address: str, timeout: float = 2.0) -> bool:
    """A ComfyUI answers at this address."""
    try:
        return "system" in _get_json(address, "/system_stats", timeout)
    except Exception:
        return False


def server_folder(address: str, timeout: float = 2.0) -> Optional[Path]:
    """The ComfyUI folder a server runs from, or None when no ComfyUI answers there."""
    try:
        paths = _get_json(address, "/internal/folder_paths", timeout)
    except Exception:
        return None
    nodes = paths.get("custom_nodes") or []
    if nodes and isinstance(nodes[0], list):        # [[paths], [extensions]] in some versions
        nodes = nodes[0]
    return Path(nodes[0]).parent if nodes else None


def same_folder(a, b) -> bool:
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port(preferred: int, folder: Path) -> Tuple[int, bool]:
    """(port, adopt): the port a ComfyUI for this folder already answers on, else the first free one."""
    for port in range(preferred, preferred + PORT_SPAN):
        found = server_folder(f"127.0.0.1:{port}")
        if found is not None and same_folder(found, folder):
            return port, True
        if found is None and port_free(port):
            return port, False
    raise RuntimeError(f"no free port between {preferred} and {preferred + PORT_SPAN - 1}")


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))     # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_pid_tree(pid: int, timeout: float = 30.0):
    """Stop a process StableGen started in an earlier session, with its children."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(int(pid), sig)
            except OSError:
                return
            end = time.time() + timeout
            while time.time() < end and pid_alive(pid):
                time.sleep(0.2)
            if not pid_alive(pid):
                return


def comfyui_python(folder: Path) -> Optional[str]:
    """ComfyUI's own Python for this folder (portable build, venv, .venv), never Blender's."""
    for candidate in installer.python_candidates(Path(folder)):
        if candidate.is_file():
            return str(candidate)
    return None


_default_python: List[Optional[str]] = []      # looked up once: [path or None]


def default_python() -> Optional[str]:
    """The Python a terminal runs as 'python' (else 'python3', else the py launcher's), as its real executable, for
    a ComfyUI that has no Python of its own: a pyenv shim or a launcher resolves to the interpreter behind it. Never
    the Microsoft Store placeholder, and never Blender's own Python."""
    if _default_python:
        return _default_python[0]
    import shutil
    commands = [[found] for found in map(shutil.which, ("python", "python3"))
                if found and "windowsapps" not in found.lower()]
    if sys.platform == "win32" and shutil.which("py"):
        commands.append([shutil.which("py"), "-3"])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    hidden = {"creationflags": 0x08000000} if sys.platform == "win32" else {}       # no console window
    found = None
    for command in commands:
        try:
            out = subprocess.run(command + ["-c", "import sys; print(sys.executable)"], capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL, env=env, timeout=30, **hidden)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = out.stdout.strip().splitlines() if out.returncode == 0 else []
        blenders = "bpy" in sys.modules and same_folder(lines[-1], sys.executable) if lines else False
        if lines and Path(lines[-1]).is_file() and not blenders:
            found = lines[-1]
            break
    _default_python.append(found)
    return found


def python_for(folder: Path) -> Tuple[Optional[str], bool]:
    """(the Python to run this ComfyUI with, whether it is the default one rather than the folder's own)."""
    own = comfyui_python(folder)
    return (own, False) if own else (default_python(), True)


# --- The process ---
class LocalComfyUI:
    """State: 'stopped', 'starting', 'running', 'exited' (it stopped or never answered; see message) or
    'unresponsive' (a recorded process that is alive but no longer answers)."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.record_path = self.state_dir / "comfyui.json"
        self.log_path = self.state_dir / "logs" / "comfyui.log"
        self.proc: Optional[subprocess.Popen] = None     # started in this session
        self.pid: Optional[int] = None                   # started by StableGen (this session or an earlier one)
        self.port: Optional[int] = None
        self.folder: Optional[Path] = None
        self.state = "stopped"
        self.message = ""
        self.started_at = 0.0
        self._watch: Optional[threading.Thread] = None

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}" if self.port else ""

    @property
    def owned(self) -> bool:
        """StableGen started it, so StableGen may stop it."""
        return self.proc is not None or self.pid is not None

    def start(self, folder, port: int = 8188, extra_args: Sequence[str] = (), env: Optional[dict] = None,
              python: Optional[str] = None) -> str:
        """Start ComfyUI for folder, or adopt the one already answering for it. Returns the address at once;
        a thread watches until it answers (state 'running') or exits."""
        folder = Path(folder)
        if self.state in ("starting", "running") and self.folder and same_folder(self.folder, folder):
            return self.address
        if self.state in ("starting", "running", "unresponsive"):
            self.stop()     # the one for another folder: stopped if StableGen started it, else let go
        if not (folder / "main.py").is_file():
            raise RuntimeError(f"{folder} has no main.py, so StableGen cannot start it")
        python = python or python_for(folder)[0]
        if not python:
            raise RuntimeError(f"No Python for {folder}: it has none of its own (python_embeded, venv or .venv) and "
                               f"there is no python on PATH")
        port, adopt = choose_port(port, folder)
        if adopt:
            self._adopt(port, folder)
            return self.address
        cmd = [python] + (["-s"] if "python_embed" in python else []) + [
            str(folder / "main.py"), "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
            *extra_args]
        child_env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **(env or {}))
        for name in ("PYTHONHOME", "PYTHONPATH"):       # Blender's own Python must not leak into ComfyUI's
            child_env.pop(name, None)
        kwargs = ({"creationflags": 0x00000200 | 0x08000000} if sys.platform == "win32"   # new group, no window
                  else {"start_new_session": True})
        self._rotate_logs()
        with open(self.log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            log.flush()
            self.proc = subprocess.Popen(cmd, cwd=str(folder), stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=child_env, **kwargs)
        self.pid, self.port, self.folder = self.proc.pid, port, folder
        self.state, self.message, self.started_at = "starting", "", time.time()
        self._write_record()
        self._watch = threading.Thread(target=self._watch_ready, daemon=True)
        self._watch.start()
        return self.address

    def _watch_ready(self):
        proc = self.proc
        while self.proc is proc and time.time() - self.started_at < READY_TIMEOUT:
            if proc.poll() is not None:
                self._exited(f"ComfyUI stopped (exit code {proc.returncode})")
                return
            if answers(self.address):
                self.state = "running"
                return
            time.sleep(1.0)
        if self.proc is proc and self.state == "starting":
            self.stop()
            self.state, self.message = "exited", f"ComfyUI did not answer within {READY_TIMEOUT // 60} min"

    def _exited(self, message: str):
        self.state, self.message = "exited", message
        self.proc, self.pid = None, None
        self._clear_record()

    def _adopt(self, port: int, folder: Path):
        """A ComfyUI for this folder already answers on port. Ours if the record says so; one started by hand is
        used but never stopped."""
        record = self._read_record()
        ours = record and record.get("port") == port and same_folder(record.get("folder", ""), folder)
        self.proc = None
        self.pid = record["pid"] if ours and pid_alive(record.get("pid")) else None
        self.port, self.folder = port, folder
        self.state, self.message = "running", "" if self.pid else "started outside StableGen"

    def adopt_recorded(self) -> bool:
        """After Blender restarts: take over the ComfyUI an earlier session started, if it still runs."""
        record = self._read_record()
        if not record:
            return False
        if not pid_alive(record.get("pid")):
            self._clear_record()
            return False
        self.pid, self.port, self.folder = record["pid"], record["port"], Path(record["folder"])
        found = server_folder(self.address)
        if found is not None and same_folder(found, self.folder):
            self.state, self.message = "running", ""
            return True
        self.state, self.message = "unresponsive", "an earlier ComfyUI is still running but does not answer"
        return False

    def poll(self) -> str:
        """Notice a ComfyUI that stopped by itself (call from a timer)."""
        if self.state == "running":
            if self.proc is not None and self.proc.poll() is not None:
                self._exited(f"ComfyUI stopped (exit code {self.proc.returncode})")
            elif self.proc is None and self.pid and not pid_alive(self.pid):
                self._exited("ComfyUI stopped")
            elif self.proc is None and not self.pid and not answers(self.address):
                self.state, self.message = "stopped", ""        # the one started by hand went away
        return self.state

    def stop(self):
        """Stop the ComfyUI StableGen started; one started outside StableGen is only let go."""
        if self.proc is not None:
            installer._stop_process_tree(self.proc)
        elif self.pid and pid_alive(self.pid):
            stop_pid_tree(self.pid)
        self.proc, self.pid = None, None
        self.state, self.message = "stopped", ""
        self._clear_record()

    def tail_log(self, lines: int = 12) -> List[str]:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [line for line in text.splitlines() if line.strip()][-lines:]

    # --- the record and logs ---
    def _read_record(self) -> Optional[dict]:
        try:
            return json.loads(self.record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_record(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.record_path.write_text(json.dumps({"pid": self.pid, "port": self.port, "folder": str(self.folder),
                                                "started": self.started_at}), encoding="utf-8")

    def _clear_record(self):
        try:
            self.record_path.unlink()
        except OSError:
            pass

    def _rotate_logs(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(LOGS_KEPT - 1, 0, -1):
            older = self.log_path.with_name(f"comfyui.{i}.log")
            newer = self.log_path if i == 1 else self.log_path.with_name(f"comfyui.{i - 1}.log")
            if newer.exists():
                try:
                    os.replace(newer, older)
                except OSError:
                    pass
