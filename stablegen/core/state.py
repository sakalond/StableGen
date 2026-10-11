"""Shared mutable state and async helpers for the StableGen addon.

Global caches for model lists, pending-refresh counters, and the
``_run_async`` helper that runs blocking I/O on a background thread
and delivers results back on Blender's main thread.
"""

import threading as _threading
import time as _time_mod
import traceback as _traceback

import bpy  # pylint: disable=import-error


# ── Model-list caches (populated by refresh operators) ────────────────────
_cached_checkpoint_list = [("NONE_AVAILABLE", "None available", "Fetch models from server")]
_cached_lora_list = [("NONE_AVAILABLE", "None available", "Fetch models from server")]
_cached_checkpoint_architecture = None
_pending_checkpoint_refresh_architecture = None
_updating_print_preset = False


def no_server_item(server_address):
    """Model-list placeholder that says why the list is empty."""
    if not server_address:
        return ("NO_SERVER", "Set Server Address", "No ComfyUI server address is configured")
    return ("NO_SERVER", "No connection to ComfyUI", f"ComfyUI isn't reachable at {server_address}")


def show_offline_lists(server_address):
    """Replace empty placeholders with the offline one; keep lists fetched earlier."""
    global _cached_checkpoint_list, _cached_lora_list
    if _cached_checkpoint_list[0][0] in {"NONE_AVAILABLE", "NO_SERVER"}:
        _cached_checkpoint_list = [no_server_item(server_address)]
    if _cached_lora_list[0][0] in {"NONE_AVAILABLE", "NO_SERVER"}:
        _cached_lora_list = [no_server_item(server_address)]


# ── In-flight refresh counter ─────────────────────────────────────────────
_pending_refreshes = 0
_refresh_started_at = 0.0   # monotonic timestamp of first in-flight refresh
_REFRESH_TIMEOUT = 30.0     # seconds before we force-clear a stuck indicator


def _inc_pending_refreshes():
    """Increment the in-flight refresh counter (call from main thread)."""
    global _pending_refreshes, _refresh_started_at
    if _pending_refreshes <= 0:
        _refresh_started_at = _time_mod.monotonic()
    _pending_refreshes += 1


def _dec_pending_refreshes():
    """Decrement the in-flight refresh counter (call from main thread)."""
    global _pending_refreshes, _refresh_started_at
    _pending_refreshes = max(0, _pending_refreshes - 1)
    if _pending_refreshes <= 0:
        _refresh_started_at = 0.0


# ── Async network helper ─────────────────────────────────────────────────

# Monotonically incrementing token so we can discard stale results when the
# server address changes while a request is still in-flight.
_async_generation = 0


def _run_async(work_fn, done_fn, poll_interval=0.25, track_generation=False):
    """Run *work_fn* in a background thread; call *done_fn(result)* on the
    main thread via ``bpy.app.timers`` when finished.

    *work_fn* receives no arguments and should return a result dict/object.
    *done_fn* receives that result.  Both run without any lock — *done_fn*
    is guaranteed to execute on the main Blender thread.

    If *track_generation* is True the call increments the global
    ``_async_generation`` counter and the result is silently discarded if
    a newer tracked call was started before this one finishes.  Use this
    only for server-address-change callbacks where stale results must be
    dropped; refresh operators should leave it False so their results are
    never accidentally discarded.
    """
    global _async_generation
    if track_generation:
        _async_generation += 1
    gen = _async_generation

    container = {}  # mutable box for the thread to deposit its result

    def _worker():
        try:
            container['result'] = work_fn()
        except Exception:
            _traceback.print_exc()
            container['result'] = None

    t = _threading.Thread(target=_worker, daemon=True)
    t.start()

    def _poll():
        # Discard result if the server address changed after we started.
        if gen != _async_generation:
            # Still call done_fn so it can clean up (e.g. decrement
            # _pending_refreshes).  Pass None to signal stale/no result.
            try:
                done_fn(None)
            except Exception:
                _traceback.print_exc()
            return None  # stop polling — stale
        if t.is_alive():
            return poll_interval  # keep polling
        try:
            done_fn(container.get('result'))
        except Exception:
            _traceback.print_exc()
        return None  # done

    bpy.app.timers.register(_poll, first_interval=poll_interval)
