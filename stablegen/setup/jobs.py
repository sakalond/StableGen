"""The installer, run in the background for the add-on: reports (--status, --plan) and installs (--events), one
install at a time, with the events turned into state the preferences draw. No bpy here.

Blender's own Python runs the installer (it only needs the standard library); --python always names ComfyUI's,
or the installer would fall back to Blender's.
"""
import json
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from . import installer

INSTALLER = Path(__file__).with_name("installer.py")
TEXT = dict(text=True, encoding="utf-8", errors="replace")
SPEED_WINDOW = 8.0          # seconds the download speed averages over


def installer_command(*args: str, python: Optional[str] = None) -> List[str]:
    return [python or sys.executable, "-I", "-u", str(INSTALLER), *args]


def _hidden() -> Dict[str, Any]:
    return ({"creationflags": 0x00000200 | 0x08000000} if sys.platform == "win32"     # new group, no window
            else {"start_new_session": True})


def run_report(args: List[str], timeout: float = 600, python: Optional[str] = None) -> Dict[str, Any]:
    """Run --status or --plan (blocking; call it on a thread) and return its JSON document. Failures come back
    as {"error": code, "message": text}."""
    try:
        result = subprocess.run(installer_command(*args, "--json", python=python), capture_output=True,
                                stdin=subprocess.DEVNULL, timeout=timeout, **TEXT, **_hidden())
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": "run", "message": str(e)}
    try:
        return json.loads(result.stdout)
    except ValueError:
        lines = (result.stderr or result.stdout).strip().splitlines()
        return {"error": "run", "message": lines[-1] if lines else f"exit code {result.returncode}"}


class InstallerJob:
    """One installer run with --events. start() returns at once; a thread reads the events."""

    def __init__(self, title: str, args: List[str], log_path, python: Optional[str] = None, kind: str = "install",
                 folder: str = "", tags: Optional[List[str]] = None):
        self.title = title
        self.kind = kind            # 'install' (packages) or 'create' (StableGen's own ComfyUI)
        self.folder = folder        # the ComfyUI folder it works on
        self.tags = list(tags or [])
        self.log_path = Path(log_path)
        self.cmd = installer_command(*args, "--events", "--log", str(self.log_path), python=python)
        self.items: Dict[str, Dict[str, Any]] = {}      # id -> name, type, size_mb, present, phase, status, ...
        self.order: List[str] = []
        self.current: Optional[str] = None
        self.errors: List[Dict[str, Any]] = []
        self.summary: Optional[Dict[str, Any]] = None
        self.verify: Optional[Dict[str, Any]] = None
        self.download_mb = 0
        self.returncode: Optional[int] = None
        self.cancelled = False
        self.started = 0.0
        self.finished = 0.0
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self.bytes_done = 0         # downloaded in this run (what a resumed .part already had doesn't count)
        self._last_done: Dict[tuple, int] = {}
        self._samples: Deque[tuple] = deque()      # (monotonic time, bytes_done) over the last SPEED_WINDOW

    @property
    def running(self) -> bool:
        return self.proc is not None and self.returncode is None

    @property
    def ok(self) -> bool:
        return (self.returncode == 0 and not self.errors and not self.cancelled
                and bool(self.summary and self.summary.get("ok")))

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, **TEXT, **_hidden())
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("@sg "):
                try:
                    self.handle(json.loads(line[4:]))
                except ValueError:
                    pass
        self.returncode = self.proc.wait()
        self.finished = time.time()

    def handle(self, event: Dict[str, Any]):
        kind = event.get("event")
        if kind == "plan":
            self.download_mb = event.get("download_mb", 0)
            for item in event.get("items", []):
                self.items[item["id"]] = dict(item, phase="waiting", status=None, done=0, total=0, file="")
                self.order.append(item["id"])
        elif kind == "item":
            item = self.items.setdefault(event["id"], {"name": event.get("name", event["id"]), "size_mb": 0,
                                                       "present": False, "done": 0, "total": 0, "file": ""})
            item["phase"] = event["phase"]
            if event["phase"] == "start":
                self.current = event["id"]
            else:
                item["status"], item["rows"] = event.get("status"), event.get("rows", [])
                if self.current == event["id"]:
                    self.current = None
        elif kind == "progress":
            item = self.items.get(event.get("id"))
            if item is not None:
                item["done"], item["total"], item["file"] = event["done"], event["total"], event.get("file", "")
                key = (event.get("id"), event.get("file", ""))
                last = self._last_done.get(key)
                if last is not None and event["done"] >= last:
                    self.bytes_done += event["done"] - last
                self._last_done[key] = event["done"]
                now = time.monotonic()
                self._samples.append((now, self.bytes_done))
                while self._samples and now - self._samples[0][0] > SPEED_WINDOW:
                    self._samples.popleft()
        elif kind == "error":
            self.errors.append(event)
        elif kind == "summary":
            self.summary = event
        elif kind == "verify":
            self.verify = event

    def cancel(self):
        if self.running:
            self.cancelled = True
            installer._stop_process_tree(self.proc)

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        if self._thread:
            self._thread.join(timeout)
        return self.returncode

    # --- for drawing ---
    def progress(self) -> float:
        """0..1 over what this run downloads: finished items plus the current file's share of its item."""
        sizes = {i: max(1, item.get("size_mb", 0)) for i, item in self.items.items() if not item.get("present")}
        total = sum(sizes.values())
        if not total:
            return 1.0 if self.returncode is not None else 0.0
        done = 0.0
        for i, size in sizes.items():
            item = self.items[i]
            if item["phase"] == "done":
                done += size
            elif i == self.current and item.get("total"):
                done += size * min(1.0, item["done"] / item["total"])
        return min(1.0, done / total)

    def speed(self) -> Optional[float]:
        """Bytes per second over the last SPEED_WINDOW seconds, falling while nothing arrives; None before a
        download has reported twice (git, pip and the checks don't report progress)."""
        if not self.running or len(self._samples) < 2 or time.monotonic() - self._samples[-1][0] > SPEED_WINDOW:
            return None
        start_time, start_bytes = self._samples[0]
        elapsed = max(time.monotonic(), self._samples[-1][0]) - start_time
        return (self._samples[-1][1] - start_bytes) / elapsed if elapsed > 0.5 else None

    @staticmethod
    def bytes_text(done, total) -> str:
        """'5.1 of 14.3 GB', or in MB below a gigabyte (uv, Git): not '0.0 of 0.0 GB'."""
        if total >= 1024**3:
            return f"{done / 1024**3:.1f} of {total / 1024**3:.1f} GB"
        return f"{done / 1024**2:.0f} of {total / 1024**2:.0f} MB"

    def total_mb(self) -> float:
        return sum(i.get("size_mb", 0) for i in self.items.values() if not i.get("present"))

    def size_text(self) -> str:
        """'5.1 of 14.3 GB' over the whole run, or 'Preparing…' before its plan arrives."""
        total = self.total_mb()
        return self.bytes_text(self.progress() * total * 1024**2, total * 1024**2) if total else "Preparing…"

    def rate_text(self) -> str:
        """'42 MB/s · 4 min left' while downloading, else ''."""
        speed = self.speed()
        if not speed:
            return ""
        left = self.total_mb() * 1024**2 * (1 - self.progress()) / speed
        rate = f"{speed / 1024**2:.0f} MB/s" if speed >= 10 * 1024**2 else f"{speed / 1024**2:.1f} MB/s"
        when = (f"{int(left // 3600)} h {int(left % 3600 // 60)} min left" if left >= 3600 else
                f"{left / 60:.0f} min left" if left >= 90 else "under a minute left" if left >= 5 else "")
        return f"{rate}  ·  {when}" if when else rate

    def status_line(self) -> str:
        if self.cancelled:
            return "Cancelled; downloads resume next time"
        if self.errors:
            return self.errors[0].get("message", "failed")
        if self.returncode is not None:
            failed = [r for r in (self.summary or {}).get("rows", []) if r["status"] == "failed"]
            if failed:
                return f"{failed[0]['name']}: {failed[0]['message']}"
            return "Done" if self.ok else f"Stopped (exit code {self.returncode})"
        item = self.items.get(self.current)
        if not item:
            return "Preparing…"
        if item.get("total"):
            return f"{item['name']}: {self.bytes_text(item['done'], item['total'])}"
        return f"{item['name']}…"
