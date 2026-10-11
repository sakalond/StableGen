import bpy
import os
import json
import uuid
import websocket
import socket
import time
import urllib.request
import urllib.parse
import email.utils
from datetime import datetime, timezone, timedelta

from .timeout_config import get_timeout
from .prompt_archive import archive_prompt
from .generation_status import set_server_stage, node_stage

from io import BytesIO
import numpy as np
from PIL import Image

from .texturing.workflows import _TexturingWorkflowMixin
from .mesh_gen.workflows import _Trellis2WorkflowMixin


class _WorkflowBase:
    """Shared infrastructure for WorkflowManager."""

    def __init__(self, operator):
        """
        Initializes the WorkflowManager.

        Args:
            operator: The instance of the ComfyUIGenerate operator.
        """
        self.operator = operator
        self._clock_offset = timedelta(0)
        self._clock_offset_utc = timedelta(0)

    def _check_server_alive(self, server_address, timeout=None):
        """Return True if the ComfyUI server responds to a lightweight request."""
        if timeout is None:
            timeout = get_timeout('ping')
        try:
            req = urllib.request.Request(
                f"http://{server_address}/system_stats",
                method='GET'
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                date_header = response.headers.get('Date')
                if date_header:
                    try:
                        server_time = email.utils.parsedate_to_datetime(date_header)
                        if server_time.tzinfo is None:
                            server_time = server_time.replace(tzinfo=timezone.utc)
                        
                        server_time_utc = server_time.astimezone(timezone.utc).replace(tzinfo=None)
                        client_time_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                        client_time_local = datetime.now()
                        
                        self._clock_offset = server_time_utc - client_time_utc
                        self._clock_offset_utc = server_time_utc - client_time_local
                        
                        print(f"[StableGen] Server UTC: {server_time_utc}, Client UTC: {client_time_utc}, Client Local: {client_time_local}")
                        print(f"[StableGen] Clock offsets calculated -> drift: {self._clock_offset}, UTC-assumption: {self._clock_offset_utc}")
                    except Exception as e_parse:
                        print(f"[StableGen] Error parsing HTTP Date header: {e_parse}")
            return True
        except Exception:
            return False

    def _get_vram_stats(self, server_address):
        """Return (vram_free, vram_total) in MB from ComfyUI /system_stats."""
        try:
            req = urllib.request.Request(
                f"http://{server_address}/system_stats", method='GET'
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=get_timeout('api')).read())
            dev = resp.get("devices", [{}])[0]
            vram_free = dev.get("vram_free", 0) / (1024 * 1024)
            vram_total = dev.get("vram_total", 0) / (1024 * 1024)
            return vram_free, vram_total
        except Exception:
            return None, None

    def _flush_comfyui_vram(self, server_address, retries=3, label="Post-generation"):
        """Unload all models and free VRAM on the ComfyUI server.

        The ``/free`` endpoint only **sets flags** on the prompt queue; the
        actual unloading happens asynchronously in ComfyUI's execution loop.
        This method therefore:
          1. Sends ``/free`` (unload models + free memory).
          2. Clears the execution cache history and queue.
          3. Sends ``/free`` *again* — the first round triggers model unloads
             whose CUDA tensors may still reside in PyTorch's cache; the
             second round triggers ``soft_empty_cache`` which calls
             ``torch.cuda.empty_cache()`` on the now-freed allocations.
          4. Polls ``/system_stats`` waiting for VRAM to increase.
        Retries the whole sequence up to *retries* times.
        """
        import time

        vram_before, vram_total = self._get_vram_stats(server_address)
        if vram_before is not None:
            print(f"[TRELLIS2] {label} VRAM before flush: {vram_before:.0f} MB free / {vram_total:.0f} MB total")

            # Skip the flush entirely if VRAM is already ≥90% free.
            if vram_total and vram_before > vram_total * 0.9:
                print(f"[TRELLIS2] {label} VRAM already ≥90% free — skipping flush.")
                return True

        def _send_free():
            d = json.dumps({"unload_models": True, "free_memory": True}).encode('utf-8')
            r = urllib.request.Request(
                f"http://{server_address}/free", data=d,
                headers={'Content-Type': 'application/json'}, method='POST'
            )
            urllib.request.urlopen(r, timeout=get_timeout('api'))

        def _clear_history():
            d = json.dumps({"clear": True}).encode('utf-8')
            r = urllib.request.Request(
                f"http://{server_address}/history", data=d,
                headers={'Content-Type': 'application/json'}, method='POST'
            )
            urllib.request.urlopen(r, timeout=get_timeout('api'))

        def _clear_queue():
            d = json.dumps({"clear": True}).encode('utf-8')
            r = urllib.request.Request(
                f"http://{server_address}/queue", data=d,
                headers={'Content-Type': 'application/json'}, method='POST'
            )
            urllib.request.urlopen(r, timeout=get_timeout('api'))

        for attempt in range(1, retries + 1):
            try:
                # Round 1: Unload models
                _send_free()
                time.sleep(1)

                # Clear caches
                try:
                    _clear_history()
                except Exception:
                    pass
                try:
                    _clear_queue()
                except Exception:
                    pass

                # Round 2: Free the now-released CUDA allocations
                _send_free()

                # Poll VRAM (up to 8s)
                for check in range(8):
                    time.sleep(1)
                    vram_after, _ = self._get_vram_stats(server_address)
                    if vram_after is not None:
                        print(f"[TRELLIS2] {label} VRAM after flush (check {check+1}): {vram_after:.0f} MB free")
                        if vram_before is not None and vram_after > vram_before + 500:
                            print(f"[TRELLIS2] {label} flush verified: freed {vram_after - vram_before:.0f} MB")
                            return True
                        if vram_total is not None and vram_after > vram_total * 0.7:
                            print(f"[TRELLIS2] {label} flush OK: >70% VRAM free")
                            return True

                print(f"[TRELLIS2] {label} flush attempt {attempt}/{retries}: VRAM didn't free enough — retrying")
            except Exception as e:
                print(f"[TRELLIS2] {label} flush attempt {attempt}/{retries} failed: {e}")

            if attempt < retries:
                time.sleep(2)

        print(f"[TRELLIS2] Warning: {label} VRAM flush unverified after {retries} attempts")
        return False


    def _save_prompt_to_file(
        self,
        prompt,
        output_dir,
        *,
        workflow="workflow",
        camera_id=None,
    ):
        """Save the latest graph and a non-overwriting per-view debug copy."""
        try:
            operator = getattr(self, "operator", None)
            view_index = getattr(operator, "_current_image", None)
            camera_name = None

            cameras = getattr(operator, "_cameras", ()) or ()
            camera_lookup_index = camera_id
            if camera_lookup_index is None:
                camera_lookup_index = view_index
            if isinstance(camera_lookup_index, int) and 0 <= camera_lookup_index < len(cameras):
                camera_name = getattr(cameras[camera_lookup_index], "name", None)

            _, archive_path = archive_prompt(
                prompt,
                output_dir,
                workflow=workflow,
                view_index=view_index if isinstance(view_index, int) else None,
                camera_name=camera_name,
            )
            print(f"[StableGen] Saved workflow archive: {archive_path}")
        except Exception as e:
            print(f"[StableGen] Failed to save prompt to file: {str(e)}")

    @staticmethod
    def _validate_image_bytes(img_bytes):
        """Return True if *img_bytes* can be decoded as an image by PIL."""
        if not img_bytes:
            return False
        try:
            Image.open(BytesIO(img_bytes)).verify()
            return True
        except Exception:
            return False

    def _get_images_via_http(self, server_address, prompt_id, node_id):
        """
        Fallback: fetch output image(s) for *node_id* from ComfyUI's
        /history + /view HTTP endpoints.  Returns a list of bytes objects,
        or an empty list on failure.
        """
        try:
            history_url = f"http://{server_address}/history/{prompt_id}"
            history = json.loads(urllib.request.urlopen(history_url, timeout=get_timeout('transfer')).read())
            outputs = history.get(prompt_id, {}).get("outputs", {}).get(node_id, {})
            image_list = outputs.get("images", [])
            result = []
            for img_meta in image_list:
                filename  = img_meta.get("filename", "")
                subfolder = img_meta.get("subfolder", "")
                img_type  = img_meta.get("type", "output")
                view_url  = (f"http://{server_address}/view?"
                             f"filename={urllib.parse.quote(filename)}"
                             f"&subfolder={urllib.parse.quote(subfolder)}"
                             f"&type={urllib.parse.quote(img_type)}")
                data = urllib.request.urlopen(view_url, timeout=get_timeout('transfer')).read()
                result.append(data)
            return result
        except Exception as e:
            print(f"[StableGen] HTTP image fallback failed: {e}")
            return []

    @staticmethod
    def _queue_contains_prompt(entries, prompt_id):
        """Return True when a ComfyUI queue payload contains *prompt_id*."""
        for entry in entries or []:
            if isinstance(entry, (list, tuple)):
                if len(entry) > 1 and str(entry[1]) == str(prompt_id):
                    return True
            elif isinstance(entry, dict):
                queued_id = entry.get('prompt_id') or entry.get('id')
                if queued_id is not None and str(queued_id) == str(prompt_id):
                    return True
        return False

    @staticmethod
    def _history_error_message(history_entry):
        """Extract the most useful execution error from a history entry."""
        status = history_entry.get('status', {})
        for message in reversed(status.get('messages', [])):
            if not isinstance(message, (list, tuple)) or len(message) < 2:
                continue
            if message[0] != 'execution_error' or not isinstance(message[1], dict):
                continue
            data = message[1]
            node_id = data.get('node_id', '?')
            detail = (data.get('exception_message') or data.get('error')
                      or 'Unknown ComfyUI execution error')
            return f"node {node_id}: {detail}"
        return status.get('status_str') or 'Unknown ComfyUI execution error'

    def _get_prompt_runtime_state(self, server_address, prompt_id):
        """Return ``(state, detail)`` for a queued ComfyUI prompt.

        ``state`` is one of ``completed``, ``error``, ``running``, ``queued``,
        ``unknown`` (server responded but the prompt is between states), or
        ``unavailable`` (both status requests failed).
        """
        history_error = None
        try:
            history_url = (
                f"http://{server_address}/history/"
                f"{urllib.parse.quote(str(prompt_id))}"
            )
            history = json.loads(urllib.request.urlopen(
                history_url, timeout=get_timeout('api')).read())
            entry = history.get(prompt_id) or history.get(str(prompt_id))
            if entry:
                status = entry.get('status', {})
                status_str = str(status.get('status_str', '')).lower()
                if status_str in {'error', 'failed'}:
                    return 'error', self._history_error_message(entry)
                if status.get('completed'):
                    return 'completed', status_str or 'completed'
                return 'running', status_str or 'history entry is active'
        except Exception as exc:  # noqa: BLE001
            history_error = exc

        queue_error = None
        try:
            queue_url = f"http://{server_address}/queue"
            queue = json.loads(urllib.request.urlopen(
                queue_url, timeout=get_timeout('api')).read())
            if self._queue_contains_prompt(queue.get('queue_running'), prompt_id):
                return 'running', 'present in ComfyUI running queue'
            if self._queue_contains_prompt(queue.get('queue_pending'), prompt_id):
                return 'queued', 'present in ComfyUI pending queue'
            return 'unknown', 'not yet visible in history or queue'
        except Exception as exc:  # noqa: BLE001
            queue_error = exc

        detail = f"history: {history_error}; queue: {queue_error}"
        return 'unavailable', detail

    def _connect_to_websocket(self, server_address, client_id):
        """Establishes WebSocket connection to ComfyUI server."""
        try:
            ws = websocket.WebSocket()
            # Use a short timeout for the initial TCP+WS handshake.
            ws.settimeout(get_timeout('api'))
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
            # This is a heartbeat interval, not a generation deadline. Slow
            # nodes can legitimately emit no WebSocket events for minutes;
            # timeout handling below checks /history and /queue and continues
            # waiting while the prompt is still active.
            ws.settimeout(get_timeout('generation_heartbeat'))
            return ws
        except ConnectionRefusedError:
            self._error = f"Connection to ComfyUI WebSocket was refused at {server_address}. Is ComfyUI running and accessible?"
            return None
        except (socket.gaierror, websocket.WebSocketAddressException): # Catch getaddrinfo errors specifically
            self._error = f"Could not resolve ComfyUI server address: '{server_address}'. Please check the hostname/IP and port in preferences and your network settings."
            return None
        except websocket.WebSocketTimeoutException:
            self._error = f"Connection to ComfyUI WebSocket timed out at {server_address}."
            return None
        except websocket.WebSocketBadStatusException as e: # More specific catch for handshake errors
            # e.status_code will be 404 in this case
            if e.status_code == 404:
                self._error = (f"ComfyUI endpoint not found at {server_address} (404 Not Found).")
            else:
                self._error = (f"WebSocket handshake failed with ComfyUI server at {server_address}. "
                            f"Status: {e.status_code}. The server might not be a ComfyUI instance or is misconfigured.")
            return None
        except Exception as e: # Catch-all for truly unexpected issues during connect
            self._error = f"An unexpected error occurred connecting WebSocket: {e}"
            return None

    @staticmethod
    def _inject_save_image_fallback(prompt, save_ws_node_id):
        """
        Inject a regular ``SaveImage`` node into *prompt* that mirrors the
        input of the ``SaveImageWebsocket`` node.  This ensures the image is
        also written to disk so it can be retrieved via HTTP if the WebSocket
        binary transfer is corrupted (e.g. packet loss on a remote network).

        Returns the node-ID of the injected node, or *None* if injection
        was not possible.
        """
        ws_node = prompt.get(save_ws_node_id)
        if not ws_node:
            return None

        images_input = ws_node.get("inputs", {}).get("images")
        if not images_input:
            return None

        # Pick a node-ID that is guaranteed not to collide
        fallback_id = f"_sg_fallback_{save_ws_node_id}"
        prompt[fallback_id] = {
            "inputs": {
                "filename_prefix": "StableGen_fallback",
                "images": images_input,
            },
            "class_type": "SaveImage",
            "_meta": {"title": "StableGen HTTP fallback"},
        }
        return fallback_id

    def _execute_prompt_and_get_images(self, ws, prompt, client_id, server_address, NODES):
        """Executes the prompt and collects generated images."""

        # Inject a disk-saving fallback node so we can recover via HTTP
        # if the WebSocket binary data arrives corrupted.
        fallback_node_id = self._inject_save_image_fallback(
            prompt, NODES.get('save_image', ''))

        # Send the prompt to the queue
        self.operator._server_context_stage = (
            getattr(self.operator, '_stage', '')
            if getattr(self.operator, '_pbr_active', False) else ''
        )
        self.operator._progress = 0.0
        set_server_stage(self.operator, 'Submitting Workflow')
        prompt_id = self._queue_prompt(prompt, client_id, server_address)
        set_server_stage(self.operator, 'Queued on Server')

        # Make every texturing workflow cancellable while blocked in recv().
        # cancel_generate() closes this socket after interrupting ComfyUI.
        if hasattr(self.operator, '_active_ws'):
            self.operator._active_ws = ws
            type(self.operator)._active_ws = ws
        
        # Process the WebSocket messages and collect images
        output_images = {}
        current_node = ""
        unavailable_heartbeats = 0
        unknown_heartbeats = 0
        
        while True:
            try:
                out = ws.recv()
                unavailable_heartbeats = 0
                unknown_heartbeats = 0
            except websocket.WebSocketTimeoutException:
                heartbeat = get_timeout('generation_heartbeat')
                state, detail = self._get_prompt_runtime_state(
                    server_address, prompt_id)
                if state == 'completed':
                    print(f"[StableGen] No WebSocket event for {heartbeat}s, "
                          "but ComfyUI history reports completion. "
                          "Recovering the result via HTTP.")
                    break
                if state == 'error':
                    self.operator._error = f"ComfyUI execution error: {detail}"
                    print(f"[StableGen] {self.operator._error}")
                    break
                if state in {'running', 'queued'}:
                    unavailable_heartbeats = 0
                    unknown_heartbeats = 0
                    if state == 'queued':
                        set_server_stage(self.operator, 'Queued on Server')
                    elif not current_node:
                        set_server_stage(self.operator, 'Preparing Inputs')
                    print(f"[StableGen] No WebSocket event for {heartbeat}s; "
                          f"prompt is still {state} ({detail}). Continuing to wait.")
                    continue
                if state == 'unknown':
                    unavailable_heartbeats = 0
                    unknown_heartbeats += 1
                    print(f"[StableGen] No WebSocket event for {heartbeat}s; "
                          f"prompt state is temporarily unknown "
                          f"({unknown_heartbeats}/5). Continuing to wait.")
                    if unknown_heartbeats < 5:
                        continue
                    self.operator._error = (
                        f"ComfyUI stopped reporting prompt {prompt_id} in both "
                        "history and queue after five heartbeat checks."
                    )
                    print(f"[StableGen] {self.operator._error}")
                    break

                unavailable_heartbeats += 1
                print(f"[StableGen] No WebSocket event for {heartbeat}s and "
                      f"ComfyUI status is unavailable "
                      f"({unavailable_heartbeats}/3): {detail}")
                if unavailable_heartbeats < 3:
                    continue
                self.operator._error = (
                    f"Lost contact with ComfyUI while waiting for prompt "
                    f"{prompt_id} after three heartbeat checks."
                )
                print(f"[StableGen] {self.operator._error}")
                break
            except (ConnectionError, OSError, Exception) as ws_err:
                print(f"[StableGen] WebSocket error during generation: {ws_err}")
                break

            if isinstance(out, str):
                message = json.loads(out)
                data = message.get('data', {})
                # Progress and errors can be broadcast by another queued job.
                # Older servers omit prompt_id on progress; use its node below.
                if data.get('prompt_id') not in (None, prompt_id):
                    continue

                if message['type'] == 'execution_start':
                    set_server_stage(self.operator, 'Preparing Inputs')
                
                elif message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] == prompt_id:
                        if data['node'] is None:
                            break  # Execution is complete
                        else:
                            current_node = str(data['node'])
                            node_info = prompt.get(current_node, {})
                            node_title = (node_info.get('_meta', {}).get('title')
                                          or node_info.get('class_type', current_node))
                            print(f"[StableGen] Executing node: {node_title} ({current_node})")
                            set_server_stage(self.operator, node_stage(current_node, node_info, NODES), node_title)
                            
                elif message['type'] == 'progress':
                    event_node = data.get('node')
                    if not current_node or (event_node is not None and str(event_node) != current_node):
                        continue
                    try:
                        maximum = float(data.get('max', 0))
                        progress = float(data['value']) / maximum * 100 if maximum > 0 else None
                    except (KeyError, TypeError, ValueError):
                        continue
                    if progress is None or not 0 <= progress <= 100:
                        continue
                    node_info = prompt.get(current_node, {})
                    phase = node_stage(current_node, node_info, NODES)
                    sampling = phase == 'Starting Sampler'
                    if sampling and progress > 0:
                        phase = 'Generating Image'
                    set_server_stage(self.operator, phase,
                                     getattr(self.operator, '_server_stage_detail', ''), progress)
                    if sampling:
                        # Keep per-image/overall sampling progress separate
                        # from encoder and other preprocessing progress.
                        self.operator._progress = progress
                        self.operator._server_step_text = f"Step {data['value']}/{data['max']}"
                        # Also update 3-tier detail when called from Trellis2Generate
                        if hasattr(self.operator, '_detail_progress'):
                            self.operator._detail_progress = progress
                            self.operator._detail_stage = f"Step {data['value']}/{data['max']}"
                            # Remap progress into a sub-range when gallery mode is active
                            remap = getattr(self.operator, '_progress_remap', None)
                            if remap:
                                base, span = remap
                                self.operator._phase_progress = base + (progress / 100.0) * span
                            else:
                                self.operator._phase_progress = progress
                            if hasattr(self.operator, '_update_overall'):
                                self.operator._update_overall()
                    print(f"[StableGen] {phase}: {progress:.1f}%")

                elif message['type'] == 'execution_error':
                    error_data = message.get('data', {})
                    error_msg = error_data.get('exception_message', 'Unknown error')
                    node_id = error_data.get('node_id', '?')
                    err_node_info = prompt.get(str(node_id), {})
                    err_node_name = (err_node_info.get('_meta', {}).get('title')
                                     or err_node_info.get('class_type', node_id))
                    self.operator._error = (
                        f"ComfyUI execution error ({err_node_name}): {error_msg}"
                    )
                    print(f"[StableGen] {self.operator._error}")
                    break
            else:
                # Binary data (image)
                if current_node == NODES['save_image']:  # SaveImageWebsocket node
                    print("[StableGen] Receiving generated image")
                    images_output = output_images.get(current_node, [])
                    images_output.append(out[8:])  # Skip the first 8 bytes (header)
                    output_images[current_node] = images_output

        if hasattr(self.operator, '_active_ws'):
            if getattr(self.operator, '_active_ws', None) is ws:
                self.operator._active_ws = None
            if getattr(type(self.operator), '_active_ws', None) is ws:
                type(self.operator)._active_ws = None

        # --- Validate received images; fall back to HTTP if missing/corrupt --
        save_node = NODES.get('save_image')
        ws_images = output_images.get(save_node, []) if save_node else []
        needs_http = (
            bool(save_node)
            and (not ws_images or any(
                not self._validate_image_bytes(img_bytes)
                for img_bytes in ws_images
            ))
        )

        if needs_http and fallback_node_id:
            set_server_stage(self.operator, 'Receiving Result', 'Downloading the generated image')
            reason = "missing" if not ws_images else "corrupt"
            print(f"[StableGen] WebSocket image data is {reason}. "
                  "Falling back to HTTP /history + /view download...")

            # ComfyUI can announce execution completion just before the
            # history entry or SaveImage file becomes visible. Retry briefly
            # so a harmless publication race does not fail a long generation.
            http_images = []
            retry_delays = (0.0, 0.25, 0.5, 1.0, 2.0)
            for attempt, delay in enumerate(retry_delays, start=1):
                if delay:
                    time.sleep(delay)
                candidates = self._get_images_via_http(
                    server_address, prompt_id, fallback_node_id)
                if candidates and all(
                        self._validate_image_bytes(data)
                        for data in candidates):
                    http_images = candidates
                    break
                if attempt < len(retry_delays):
                    print(f"[StableGen] HTTP image fallback not ready "
                          f"(attempt {attempt}/{len(retry_delays)}); retrying...")

            if http_images:
                output_images[save_node] = http_images
                print(f"[StableGen] Successfully retrieved "
                      f"{len(http_images)} image(s) via HTTP fallback.")
            else:
                print("[StableGen] HTTP fallback also failed. "
                      "The image data may be unrecoverable.")
        elif needs_http:
            print("[StableGen] WebSocket image data is missing or corrupt "
                  "and no HTTP fallback node was injected.")

        final_images = output_images.get(save_node, []) if save_node else []
        if (not save_node or not final_images or any(
                not self._validate_image_bytes(data)
                for data in final_images)):
            if not self.operator._error:
                self.operator._error = (
                    f"ComfyUI completed but StableGen received no valid image "
                    f"from output node {save_node!r}."
                )
            print(f"[StableGen] {self.operator._error}")
            self.operator._server_stage = None
            return {"error": "missing_output"}

        self.operator._server_stage = None
        return output_images

    def _queue_prompt(self, prompt, client_id, server_address):
        """Queues the prompt for processing by ComfyUI."""
        try:
            data = json.dumps({
                "prompt": prompt,
                "client_id": client_id
            }).encode('utf-8')
            
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
            response = json.loads(urllib.request.urlopen(req, timeout=get_timeout('api')).read())
            
            return response['prompt_id']
        except Exception as e:
            print(f"[StableGen] Failed to queue prompt: {str(e)}")
            raise



class WorkflowManager(_TexturingWorkflowMixin, _Trellis2WorkflowMixin, _WorkflowBase):
    """Full workflow manager composing texturing, mesh-gen and base infra."""
    pass


# Backward-compat: re-export top-level helpers moved to texturing.workflows
from .texturing.workflows import _texturing_prompt, _generation_prompt  # noqa: E402,F401
