import os
import json
import uuid
import websocket
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

from ..timeout_config import get_timeout
from ..utils import get_generation_dirs
from ..util.workflow_templates import prompt_text_trellis2, prompt_text_trellis2_shape_only

from io import BytesIO
import numpy as np
from PIL import Image

_ADDON_PKG = __package__.rsplit('.', 1)[0]


class _Trellis2WorkflowMixin:
    """TRELLIS.2 workflow methods for WorkflowManager."""

    @staticmethod
    def _cleanup_trellis2_temp_files():
        """Remove stale TRELLIS2 IPC temp directories and voxelgrid cache.

        The ComfyUI-TRELLIS2 custom node creates one temp directory
        (``trellis2_<random>``) per worker subprocess via a module-level
        global and reuses it for the entire ComfyUI session.  Deleting
        that directory while the worker is alive causes "Parent directory
        does not exist" errors on the next generation.

        Since only the most recently created directory can belong to the
        current (running) worker, we delete all *older* ``trellis2_*``
        directories entirely and only clean up the stale ``.pt`` tensor
        files inside the newest one, leaving the directory itself intact.
        """
        import tempfile
        import glob
        import shutil

        tmp_root = tempfile.gettempdir()

        dirs = sorted(
            (d for d in glob.glob(os.path.join(tmp_root, 'trellis2_*'))
             if os.path.isdir(d)),
            key=os.path.getctime,
        )

        if not dirs:
            return

        # Everything except the newest is from a dead worker — remove entirely.
        for d in dirs[:-1]:
            try:
                shutil.rmtree(d)
            except OSError:
                pass

        # Newest directory may be in use — only purge .pt files inside it.
        newest = dirs[-1]
        for f in os.listdir(newest):
            if f.endswith('.pt'):
                try:
                    os.remove(os.path.join(newest, f))
                except OSError:
                    pass

        # Voxelgrid cache (Windows: C:\tmp\trellis2_cache)
        for cache_dir in ('/tmp/trellis2_cache', r'C:\tmp\trellis2_cache'):
            if os.path.isdir(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except OSError:
                    pass

    @staticmethod
    def _rasterize_alpha(image_path, background_color='black'):
        """Flatten a PNG's alpha channel onto a solid background colour.

        When BG removal is set to *Skip*, the user's image may still
        contain arbitrary RGB data behind transparent pixels.  Vision
        encoders read raw RGB — not premultiplied data — so those
        hidden pixels leak into the conditioning and corrupt the
        generated shape.

        This method composites the foreground onto *background_color*,
        writes a new temp PNG (preserving the alpha channel for ComfyUI's
        LoadImage mask output), and returns the path to that temp file.

        If the image has no alpha channel, the original path is returned
        unchanged.
        """
        import tempfile

        img = Image.open(image_path)
        if img.mode != 'RGBA':
            return image_path  # nothing to rasterize

        bg_map = {'black': (0, 0, 0), 'gray': (128, 128, 128), 'white': (255, 255, 255)}
        bg_rgb = bg_map.get(background_color, (0, 0, 0))

        # Composite: result_rgb = fg_rgb * alpha + bg_rgb * (1 - alpha)
        r, g, b, a = img.split()
        bg = Image.new('RGB', img.size, bg_rgb)
        bg.paste(img, mask=a)

        # Re-attach the original alpha so ComfyUI's LoadImage still
        # provides the correct mask output via its second slot.
        result = bg.convert('RGBA')
        result.putalpha(a)

        fd, tmp_path = tempfile.mkstemp(suffix='.png', prefix='sg_rasterized_')
        os.close(fd)
        result.save(tmp_path, 'PNG')
        print(f"[TRELLIS2] Rasterized alpha onto {background_color} background: {tmp_path}")
        return tmp_path

    @staticmethod
    def _is_local_server(server_address):
        """Return True if server_address points to localhost."""
        host = server_address.split(':')[0].strip()
        return host in ('127.0.0.1', 'localhost', '0.0.0.0', '::1', '')

    def generate_trellis2(self, context, input_image_path):
        """
        Generates a 3D mesh using TRELLIS.2 via ComfyUI.

        Uploads an input image, runs the TRELLIS.2 pipeline (background removal,
        conditioning, shape generation, optionally texture generation, GLB export),
        and downloads the resulting GLB file from the ComfyUI server.
        VRAM is always flushed before AND after generation.

        Args:
            context: Blender context.
            input_image_path: Local path to the input image file.

        Returns:
            bytes: GLB file binary data on success.
            dict: {"error": "message"} on failure.
        """
        import time

        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address

        # Pre-generation flush: free any loaded diffusion/other models so
        # TRELLIS.2 has maximum VRAM available.
        print("[TRELLIS2] Pre-generation VRAM flush — freeing loaded models...")
        self._flush_comfyui_vram(server_address, label="Pre-generation")

        time.sleep(1)  # Brief pause for CUDA memory to be released

        # Verify the server is alive before proceeding
        if not self._check_server_alive(server_address):
            return {"error": "ComfyUI server is not responding. Please restart it and try again."}

        try:
            client_id = str(uuid.uuid4())
            result = self._generate_trellis2_inner(
                context, input_image_path, server_address, client_id)
            return result
        finally:
            self._flush_comfyui_vram(server_address, label="Post-generation")
            self._cleanup_trellis2_temp_files()

    def _generate_trellis2_inner(self, context, input_image_path, server_address, client_id):
        """Inner implementation of generate_trellis2, called within a try/finally VRAM flush."""
        import urllib.parse

        # Upload the input image to ComfyUI
        from .._generator_utils import upload_image_to_comfyui

        # When BG removal is skipped, rasterize the image client-side:
        # flatten the alpha channel onto the configured background colour so
        # that invisible pixels behind the alpha don't leak into the
        # conditioning encoder (which reads raw RGB, not premultiplied data).
        skip_bg = getattr(context.scene, 'trellis2_bg_removal', 'auto') == 'skip'
        rasterized_tmp = None
        if skip_bg:
            maybe_tmp = self._rasterize_alpha(
                input_image_path,
                getattr(context.scene, 'trellis2_background_color', 'black'),
            )
            if maybe_tmp != input_image_path:
                rasterized_tmp = maybe_tmp
                input_image_path = rasterized_tmp

        image_info = upload_image_to_comfyui(server_address, input_image_path)

        # Clean up the temp rasterized file after upload
        if rasterized_tmp:
            try:
                os.remove(rasterized_tmp)
            except OSError:
                pass

        if image_info is None:
            self.operator._error = f"Failed to upload input image: {input_image_path}"
            return {"error": self.operator._error}

        scene = context.scene
        skip_texture = scene.trellis2_skip_texture

        # Load the appropriate workflow template
        if skip_texture:
            prompt = json.loads(prompt_text_trellis2_shape_only)
            NODES = {
                'input_image': '1',
                'load_models': '2',
                'remove_bg': '3',
                'get_conditioning': '4',
                'image_to_shape': '5',
                'simplify': '6',
                'export_trimesh': '7',
            }
            export_node_key = 'export_trimesh'
        else:
            prompt = json.loads(prompt_text_trellis2)
            NODES = {
                'input_image': '1',
                'load_models': '2',
                'remove_bg': '3',
                'get_conditioning': '4',
                'image_to_shape': '5',
                'shape_to_textured_mesh': '6',
                'process_mesh': '8',
                'rasterize_pbr': '9',
                'export_glb': '7',
            }
            export_node_key = 'export_glb'

        # Set input image
        prompt[NODES['input_image']]["inputs"]["image"] = image_info['name']

        # Configure model settings from scene properties
        prompt[NODES['load_models']]["inputs"]["resolution"] = scene.trellis2_resolution
        prompt[NODES['load_models']]["inputs"]["precision"] = scene.trellis2_precision
        prompt[NODES['load_models']]["inputs"]["attn_backend"] = scene.trellis2_attn_backend

        # Configure background removal (or bypass it)
        skip_bg = getattr(scene, 'trellis2_bg_removal', 'auto') == 'skip'
        if skip_bg:
            # Remove the RemoveBackground node from the workflow.
            # Wire GetConditioning to use LoadImage outputs directly:
            #   LoadImage[0] → image,  LoadImage[1] → mask (alpha channel).
            # If the input has no alpha, ComfyUI's LoadImage provides an
            # all-white mask (= entire image is foreground).
            remove_bg_node_id = NODES.pop('remove_bg')
            del prompt[remove_bg_node_id]
            prompt[NODES['get_conditioning']]["inputs"]["image"] = [NODES['input_image'], 0]
            prompt[NODES['get_conditioning']]["inputs"]["mask"] = [NODES['input_image'], 1]
        else:
            prompt[NODES['remove_bg']]["inputs"]["low_vram"] = True

        # Configure conditioning
        prompt[NODES['get_conditioning']]["inputs"]["background_color"] = scene.trellis2_background_color

        # Configure shape generation
        seed = scene.trellis2_seed
        prompt[NODES['image_to_shape']]["inputs"]["seed"] = seed
        prompt[NODES['image_to_shape']]["inputs"]["ss_guidance_strength"] = scene.trellis2_ss_guidance
        prompt[NODES['image_to_shape']]["inputs"]["ss_sampling_steps"] = scene.trellis2_ss_steps
        prompt[NODES['image_to_shape']]["inputs"]["shape_guidance_strength"] = scene.trellis2_shape_guidance
        prompt[NODES['image_to_shape']]["inputs"]["shape_sampling_steps"] = scene.trellis2_shape_steps
        prompt[NODES['image_to_shape']]["inputs"]["max_tokens"] = scene.trellis2_max_tokens

        # Configure export with unique prefix for identification
        unique_prefix = f"trellis2_{uuid.uuid4().hex[:8]}"

        # When post-processing is disabled, maximize the decimation target
        # and disable remeshing so the user gets the rawest possible mesh.
        use_pp = getattr(scene, 'trellis2_post_processing_enabled', True)
        decimate_method = getattr(scene, 'trellis2_decimate_method', 'server')
        remesh_method = getattr(scene, 'trellis2_remesh_method', 'qdc')

        if skip_texture:
            bypass_server_simplify = (not use_pp) or (remesh_method != 'qdc' and decimate_method != 'server')
            if bypass_server_simplify:
                # Bypass simplify node entirely: connect export directly to shape generation
                prompt[NODES['export_trimesh']]["inputs"]["trimesh"] = [NODES['image_to_shape'], 0]
                # Remove the simplify node from the prompt
                simplify_id = NODES.get('simplify')
                if simplify_id and simplify_id in prompt:
                    del prompt[simplify_id]
            else:
                # Post-processing is enabled and uses server-side QDC or decimation
                if remesh_method == 'qdc':
                    # Server QDC handles both decimation and remeshing
                    prompt[NODES['simplify']]["inputs"]["target_face_count"] = scene.trellis2_decimation
                    prompt[NODES['simplify']]["inputs"]["remesh"] = True
                elif decimate_method == 'server':
                    # Server decimation only (no server remesh)
                    prompt[NODES['simplify']]["inputs"]["target_face_count"] = scene.trellis2_decimation
                    prompt[NODES['simplify']]["inputs"]["remesh"] = False
                prompt[NODES['simplify']]["inputs"]["fill_holes"] = False
            prompt[NODES['export_trimesh']]["inputs"]["filename_prefix"] = unique_prefix
        else:
            # Configure texture generation
            prompt[NODES['shape_to_textured_mesh']]["inputs"]["seed"] = seed
            prompt[NODES['shape_to_textured_mesh']]["inputs"]["tex_guidance_strength"] = scene.trellis2_tex_guidance
            prompt[NODES['shape_to_textured_mesh']]["inputs"]["tex_sampling_steps"] = scene.trellis2_tex_steps
 
            # Configure GLB export via process/rasterize/export pipeline
            decimate_method = getattr(scene, 'trellis2_decimate_method', 'server')
            remesh_method = getattr(scene, 'trellis2_remesh_method', 'qdc')
            
            bypass_server_process = (not use_pp) or (remesh_method != 'qdc' and decimate_method != 'server')
            if bypass_server_process:
                # Local decimation and/or local remeshing, or post-processing disabled: bypass server process_mesh entirely
                prompt[NODES['rasterize_pbr']]["inputs"]["trimesh"] = [NODES['image_to_shape'], 0]
                process_id = NODES.get('process_mesh')
                if process_id and process_id in prompt:
                    del prompt[process_id]
            else:
                if remesh_method == 'qdc':
                    prompt[NODES['process_mesh']]["inputs"]["target_face_count"] = scene.trellis2_decimation
                    prompt[NODES['process_mesh']]["inputs"]["remesh"] = "on"
                elif decimate_method == 'server':
                    prompt[NODES['process_mesh']]["inputs"]["target_face_count"] = scene.trellis2_decimation
                    prompt[NODES['process_mesh']]["inputs"]["remesh"] = "off"
            
            prompt[NODES['rasterize_pbr']]["inputs"]["texture_size"] = scene.trellis2_texture_size
            prompt[NODES['export_glb']]["inputs"]["filename_prefix"] = unique_prefix

        # Save prompt for debugging
        revision_dir = get_generation_dirs(context).get("revision", "")
        if revision_dir:
            self._save_prompt_to_file(prompt, revision_dir)

        # --- Two-phase VRAM management for textured path ---
        # NOTE: Two-phase execution (shape first, flush, then texture) was
        # removed because ComfyUI requires at least one OUTPUT_NODE per prompt
        # and the shape-only subset has none.  The TRELLIS pipeline stages
        # handle VRAM management internally (unload_shape_pipeline before
        # loading texture models), so a single full-prompt submission works.

        # Connect WebSocket
        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": "conn_failed"}

        # TRELLIS.2 simplification / post-processing can take several
        # minutes without sending any WS messages.  Use the user-
        # configurable mesh generation timeout.
        ws.settimeout(get_timeout('mesh_gen'))

        # Let the operator close this WS on cancel
        if hasattr(self.operator, '_active_ws'):
            self.operator._active_ws = ws

        prompt_id = None
        try:
            # Queue prompt
            prompt_id = self._queue_prompt(prompt, client_id, server_address)

            # Node-level progress (isolated subprocess doesn't emit within-node progress)
            # Weights approximate actual time spent per node
            if skip_texture:
                NODE_PROGRESS = {
                    NODES['input_image']:        1,
                    NODES['load_models']:        2,
                    NODES['get_conditioning']:   8,
                    NODES['image_to_shape']:     10,
                    NODES['simplify']:           85,
                    NODES['export_trimesh']:     95,
                }
                if not skip_bg:
                    NODE_PROGRESS[NODES['remove_bg']] = 5
            else:
                # Full textured pipeline (single submission)
                NODE_PROGRESS = {
                    NODES['input_image']:              1,
                    NODES['load_models']:              2,
                    NODES['get_conditioning']:         8,
                    NODES['image_to_shape']:           10,
                    NODES['shape_to_textured_mesh']:   50,
                    NODES['process_mesh']:             85,
                    NODES['rasterize_pbr']:            92,
                    NODES['export_glb']:               98,
                }
                if not skip_bg:
                    NODE_PROGRESS[NODES['remove_bg']] = 5

            # Wait for execution to complete via WebSocket
            # Also capture 'executed' events which may contain the output path
            export_node_id = NODES[export_node_key]
            glb_output_path = None

            # Friendly node labels for the progress bars
            NODE_LABELS = {
                'input_image':              'Loading Image',
                'load_models':              'Loading Models',
                'remove_bg':                'Removing Background',
                'get_conditioning':         'Conditioning',
                'image_to_shape':           'Generating Shape',
                'shape_to_textured_mesh':   'Generating Texture',
                'process_mesh':             'Processing Mesh',
                'rasterize_pbr':            'Rasterizing PBR',
                'simplify':                 'Simplifying Mesh',
                'export_trimesh':           'Exporting Mesh',
                'export_glb':               'Exporting GLB',
            }

            # Build a lookup of known step counts so we can identify
            # which sampling sub-phase a within-node progress event
            # belongs to (the TRELLIS nodes may emit separate runs of
            # progress events for SS, shape, and texture sampling).
            _ss_steps    = scene.trellis2_ss_steps
            _shape_steps = scene.trellis2_shape_steps
            _tex_steps   = getattr(scene, 'trellis2_tex_steps', 0)
            _resolution  = getattr(scene, 'trellis2_resolution', '1024_cascade')
            _is_cascade  = _resolution in ('1024_cascade', '1536_cascade')
            _current_exec_node = None  # node-key of the currently executing node
            _current_exec_node_id = None  # node-id of the currently executing node

            # Track accumulated sub-phases within a node so we can
            # compute a node-internal overall progress.
            _sub_phase_idx = 0     # resets when the executing node changes
            _sub_phase_count = 1   # how many sub-phases this node has
            _prev_p_value = 0      # track previous step value to detect restarts

            while True:
                try:
                    out = ws.recv()
                except websocket.WebSocketTimeoutException:
                    # Timeout on recv — server is still alive but a heavy
                    # step (e.g. simplification) took longer than expected.
                    # Keep waiting instead of treating it as a crash.
                    print("[TRELLIS2] WebSocket recv timed out — "
                          "server may still be processing. Retrying...")
                    continue
                except (ConnectionError, OSError, Exception) as ws_err:
                    err_name = type(ws_err).__name__
                    print(f"[TRELLIS2] WebSocket died ({err_name}): {ws_err}")
                    # Distinguish user-initiated cancel from a real crash
                    if getattr(self.operator, '_cancelled', False):
                        return {"error": "cancelled"}
                    return {"error": (
                        "ComfyUI server crashed during TRELLIS.2 generation "
                        "(likely VRAM exhaustion). Please restart ComfyUI, "
                        "reduce max_tokens or resolution, and try again."
                    )}
                if isinstance(out, str):
                    message = json.loads(out)

                    if message['type'] == 'executing':
                        data = message['data']
                        if data['prompt_id'] == prompt_id:
                            if data['node'] is None:
                                self.operator._phase_progress = 100
                                self.operator._detail_progress = 100
                                if hasattr(self.operator, '_update_overall'):
                                    self.operator._update_overall()
                                break  # Execution complete
                            else:
                                node_id = data['node']
                                node_names = {v: k for k, v in NODES.items()}
                                node_key = node_names.get(node_id, node_id)
                                node_label = NODE_LABELS.get(node_key, node_key)
                                print(f"[TRELLIS2] Executing: {node_label}")
                                self.operator._phase_stage = f"TRELLIS.2: {node_label}"
                                # Update progress based on node weight
                                if node_id in NODE_PROGRESS:
                                    self.operator._phase_progress = max(self.operator._phase_progress, NODE_PROGRESS[node_id])
                                    if hasattr(self.operator, '_update_overall'):
                                        self.operator._update_overall()
                                self.operator._detail_stage = node_label
                                self.operator._detail_progress = 0

                                # Reset sub-phase tracking for the new node if not already transitioned
                                if node_id != _current_exec_node_id:
                                    _current_exec_node = node_key
                                    _current_exec_node_id = node_id
                                    _sub_phase_idx = 0
                                    _prev_p_value = 0
                                    if node_key == 'image_to_shape':
                                        # SS sampling + SLat base (+ SLat cascade if cascade resolution)
                                        _sub_phase_count = 3 if _is_cascade else 2
                                    elif node_key == 'shape_to_textured_mesh':
                                        _sub_phase_count = 1
                                    else:
                                        _sub_phase_count = 1

                    elif message['type'] == 'executed':
                        # Capture output from export node (newer ComfyUI versions)
                        data = message.get('data', {})
                        if data.get('node') == export_node_id:
                            ws_output = data.get('output', {})
                            if ws_output:
                                for key in ['glb_path', 'file_path', 'text', 'string']:
                                    val = ws_output.get(key)
                                    if val:
                                        if isinstance(val, list) and len(val) > 0:
                                            val = val[0]
                                        if isinstance(val, str) and val:
                                            glb_output_path = val
                                            print(f"[TRELLIS2] Got path from WS executed event: {glb_output_path}")
                                            break
                                # Also check any key with a file-like value
                                if not glb_output_path:
                                    for key, val in ws_output.items():
                                        if isinstance(val, list) and len(val) > 0:
                                            val = val[0]
                                        if isinstance(val, str) and val.endswith(('.glb', '.obj', '.ply')):
                                            glb_output_path = val
                                            print(f"[TRELLIS2] Got path from WS (key={key}): {glb_output_path}")
                                            break

                    elif message['type'] == 'progress':
                        # Within-node progress (sampler steps).
                        # TRELLIS.2 nodes may emit separate runs of progress
                        # events for each internal sampling phase.  We detect
                        # the sub-phase transition by checking if the progress
                        # value decreased (restarted), which works even when the
                        # step counts (max) of different sub-phases are identical.
                        p_data = message['data']
                        # Only process progress for the currently executing node to prevent
                        # stray progress events from corrupting our stateful sub-phase tracking.
                        p_node = p_data.get('node')
                        # If the progress event is for a new node, transition to it immediately.
                        # This prevents timing races where the 'executing' event is delayed.
                        if p_node and p_node != _current_exec_node_id and p_node in NODE_PROGRESS:
                            _current_exec_node_id = p_node
                            node_names = {v: k for k, v in NODES.items()}
                            _current_exec_node = node_names.get(p_node, p_node)
                            node_label = NODE_LABELS.get(_current_exec_node, _current_exec_node)
                            print(f"[TRELLIS2] Early transition to node {_current_exec_node} via progress event")
                            self.operator._phase_stage = f"TRELLIS.2: {node_label}"
                            self.operator._detail_stage = node_label
                            self.operator._detail_progress = 0

                            # Reset sub-phase tracking
                            _sub_phase_idx = 0
                            _prev_p_value = 0
                            if _current_exec_node == 'image_to_shape':
                                _sub_phase_count = 3 if _is_cascade else 2
                            else:
                                _sub_phase_count = 1

                        if p_node and p_node != _current_exec_node_id:
                            continue

                        p_value = p_data['value']
                        p_max   = p_data['max']

                        # Filter out any progress updates that do not match the expected sampler steps.
                        # This eliminates high-level node progress (p_max=3) and stray tiled convolution tqdm progress (e.g. p_max=5 or p_max=37).
                        if _current_exec_node == 'image_to_shape':
                            if p_max not in (_ss_steps, _shape_steps):
                                continue
                        elif _current_exec_node == 'shape_to_textured_mesh':
                            if p_max != _tex_steps:
                                continue

                        step_progress = (p_value / p_max) * 100 if p_max else 0

                        # Detect sub-phase transition
                        if p_value < _prev_p_value:
                            _sub_phase_idx = min(_sub_phase_idx + 1, _sub_phase_count - 1)
                            print(f"[TRELLIS2] Sub-phase restart detected. Transitioning to sub-phase index {_sub_phase_idx}")
                        _prev_p_value = p_value

                        # ── Sub-phase identification ──
                        sub_label = ""
                        if _current_exec_node == 'image_to_shape':
                            if _sub_phase_idx == 0:
                                sub_label = "Sampling SS"
                            elif _sub_phase_idx == 1:
                                sub_label = "Sampling Shape SLat LR" if _is_cascade else "Sampling Shape SLat"
                            else:
                                sub_label = "Sampling Shape SLat HR"
                        elif _current_exec_node == 'shape_to_textured_mesh':
                            sub_label = "Sampling Texture"
                        else:
                            sub_label = "Processing"

                        # Compute node-internal overall progress accounting for weighted sub-phases.
                        # SS sampling is very fast, SLat Base is medium, SLat Cascade is slow.
                        if _current_exec_node == 'image_to_shape':
                            sub_weights = [10, 40, 50] if _is_cascade else [15, 85]
                        else:
                            sub_weights = [100]

                        idx = min(_sub_phase_idx, len(sub_weights) - 1)
                        completed_progress = sum(sub_weights[:idx])
                        current_weight = sub_weights[idx]
                        node_overall = completed_progress + (step_progress / 100.0) * current_weight

                        if step_progress != 0:
                            self.operator._detail_progress = step_progress
                            self.operator._detail_stage = (
                                f"{sub_label}: Step {p_value}/{p_max}"
                            )
                            # Interpolate the overall phase progress dynamically based on node_overall
                            if _current_exec_node_id in NODE_PROGRESS:
                                base_pct = NODE_PROGRESS[_current_exec_node_id]
                                sorted_pcts = sorted(list(NODE_PROGRESS.values()))
                                try:
                                    idx = sorted_pcts.index(base_pct)
                                    next_pct = sorted_pcts[idx + 1] if idx + 1 < len(sorted_pcts) else 100
                                except (ValueError, IndexError):
                                    next_pct = base_pct + 10  # fallback
                                span = next_pct - base_pct
                                new_phase_pct = base_pct + (node_overall / 100.0) * span
                                self.operator._phase_progress = max(self.operator._phase_progress, new_phase_pct)
                                if hasattr(self.operator, '_update_overall'):
                                    self.operator._update_overall()
                            print(f"[TRELLIS2] {sub_label}: Step {p_value}/{p_max} ({step_progress:.0f}%)")

                    elif message['type'] == 'execution_error':
                        error_data = message.get('data', {})
                        error_msg = error_data.get('exception_message', 'Unknown error')
                        self.operator._error = f"TRELLIS.2 execution error: {error_msg}"
                        print(f"[TRELLIS2] Error: {self.operator._error}")
                        return {"error": self.operator._error}
        finally:
            if hasattr(self.operator, '_active_ws'):
                self.operator._active_ws = None
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass

        # ── Retrieve the GLB file ──────────────────────────────────────
        # Strategy 1: path from WS 'executed' event (already captured above)
        # Strategy 2: path from history API
        # Strategy 3: direct file read from disk (local ComfyUI)
        # Strategy 4: HTTP download via /view endpoint
        # Strategy 5: scan ComfyUI output dir for our prefix

        glb_source_path = glb_output_path  # May already be set from WS event

        # Helper: bail early when the user hits Cancel
        def _cancelled():
            return getattr(self.operator, '_cancelled', False)

        # Strategy 2: Query history for the output path
        if not glb_source_path and not _cancelled():
            try:
                history_url = f"http://{server_address}/history/{prompt_id}"
                history_response = json.loads(urllib.request.urlopen(
                    history_url, timeout=get_timeout('api')).read())

                if prompt_id in history_response:
                    outputs = history_response[prompt_id].get("outputs", {})
                    export_output = outputs.get(export_node_id, {})
                    print(f"[TRELLIS2] History output for node {export_node_id}: {export_output}")

                    for key in ['glb_path', 'file_path', 'text', 'string']:
                        val = export_output.get(key)
                        if val:
                            if isinstance(val, list) and len(val) > 0:
                                val = val[0]
                            if isinstance(val, str) and val:
                                glb_source_path = val
                                print(f"[TRELLIS2] Got path from history (key={key}): {glb_source_path}")
                                break

                    # Check any key with a path-like value
                    if not glb_source_path:
                        for key, val in export_output.items():
                            if isinstance(val, list) and len(val) > 0:
                                val = val[0]
                            if isinstance(val, str) and val.endswith(('.glb', '.obj', '.ply')):
                                glb_source_path = val
                                print(f"[TRELLIS2] Got path from history (key={key}): {glb_source_path}")
                                break
            except Exception as e:
                print(f"[TRELLIS2] History query failed: {e}")

        # Strategy 3: Read directly from disk (works when ComfyUI is local)
        if glb_source_path and not _cancelled() and os.path.isfile(glb_source_path):
            try:
                print(f"[TRELLIS2] Reading GLB directly from disk: {glb_source_path}")
                with open(glb_source_path, 'rb') as f:
                    glb_data = f.read()
                print(f"[TRELLIS2] Read {len(glb_data)} bytes from disk")
                if glb_data and len(glb_data) > 0:
                    return glb_data
            except Exception as e:
                print(f"[TRELLIS2] Direct file read failed: {e}")

        # Strategy 4: HTTP download via /view endpoint
        if glb_source_path and not _cancelled():
            glb_filename = os.path.basename(glb_source_path)
            try:
                view_url = f"http://{server_address}/view?filename={urllib.parse.quote(glb_filename)}&type=output"
                print(f"[TRELLIS2] Downloading GLB via HTTP: {view_url}")
                glb_response = urllib.request.urlopen(view_url, timeout=get_timeout('transfer'))
                glb_data = glb_response.read()
                print(f"[TRELLIS2] Downloaded GLB: {len(glb_data)} bytes")
                if glb_data and len(glb_data) > 0:
                    return glb_data
            except Exception as e:
                print(f"[TRELLIS2] HTTP download failed: {e}")

        if _cancelled():
            return {"error": "cancelled"}

        # Strategy 5: Scan ComfyUI output directory for files matching our prefix
        # This handles the case where the export node path wasn't captured via API
        print(f"[TRELLIS2] Scanning for files with prefix '{unique_prefix}'...")

        # 5a: Try to discover ComfyUI's output directory from the server
        comfyui_output_dir = None
        if not _cancelled():
            try:
                system_url = f"http://{server_address}/system_stats"
                system_response = json.loads(urllib.request.urlopen(
                    system_url, timeout=get_timeout('api')).read())
                # Some ComfyUI versions include directory info
                comfyui_output_dir = system_response.get("output_dir")
            except Exception:
                pass

        # 5b: Try common local paths relative to server
        if not comfyui_output_dir:
            # Check if server is localhost - if so, try to find output dir
            host = server_address.split(':')[0]
            if host in ('127.0.0.1', 'localhost', '0.0.0.0', '::1'):
                # Try to discover via the /view endpoint with a known file
                # Or just check common ComfyUI locations
                common_paths = [
                    os.path.join(os.environ.get('COMFYUI_PATH', ''), 'output'),
                    'C:/ComfyUI/output',
                    os.path.expanduser('~/ComfyUI/output'),
                ]
                for candidate in common_paths:
                    if candidate and os.path.isdir(candidate):
                        comfyui_output_dir = candidate
                        break

        if comfyui_output_dir and os.path.isdir(comfyui_output_dir):
            print(f"[TRELLIS2] Scanning output dir: {comfyui_output_dir}")
            try:
                matching_files = sorted(
                    [f for f in os.listdir(comfyui_output_dir)
                     if f.startswith(unique_prefix) and f.endswith(('.glb', '.obj', '.ply'))],
                    key=lambda f: os.path.getmtime(os.path.join(comfyui_output_dir, f)),
                    reverse=True
                )
                if matching_files:
                    found_path = os.path.join(comfyui_output_dir, matching_files[0])
                    print(f"[TRELLIS2] Found matching file: {found_path}")
                    with open(found_path, 'rb') as f:
                        glb_data = f.read()
                    print(f"[TRELLIS2] Read {len(glb_data)} bytes from disk")
                    if glb_data and len(glb_data) > 0:
                        return glb_data
            except Exception as e:
                print(f"[TRELLIS2] Output dir scan failed: {e}")

        # 5c: Try HTTP download with timestamp-based filename guesses
        # The export nodes use format: {prefix}_{YYYYMMDD_HHMMSS}.glb
        # Use a short timeout per request so slow remote connections don't
        # block for minutes, and check for cancellation each iteration.
        is_remote = not self._is_local_server(server_address)
        scan_range = 30 if is_remote else 120  # fewer guesses for remote
        scan_timeout = get_timeout('scan')
        if is_remote:
            scan_timeout = max(1.0, scan_timeout / 2.0)

        clock_offset = getattr(self, '_clock_offset', timedelta(0))
        now_local = datetime.now() + clock_offset
        candidates = [now_local]
        if is_remote:
            clock_offset_utc = getattr(self, '_clock_offset_utc', None)
            if clock_offset_utc is not None:
                now_utc = datetime.now() + clock_offset_utc
                if abs((now_utc - now_local).total_seconds()) > 5:
                    candidates.append(now_utc)

        for delta_seconds in range(0, scan_range):
            if _cancelled():
                return {"error": "cancelled"}
            for delta in [timedelta(seconds=-delta_seconds), timedelta(seconds=delta_seconds)]:
                for base_now in candidates:
                    candidate_time = base_now + delta
                    candidate_name = f"{unique_prefix}_{candidate_time.strftime('%Y%m%d_%H%M%S')}.glb"
                    try:
                        view_url = f"http://{server_address}/view?filename={urllib.parse.quote(candidate_name)}&type=output"
                        glb_response = urllib.request.urlopen(view_url, timeout=scan_timeout)
                        glb_data = glb_response.read()
                        if glb_data and len(glb_data) > 0:
                            print(f"[TRELLIS2] Found GLB via timestamp scan: {candidate_name} ({len(glb_data)} bytes)")
                            return glb_data
                    except Exception:
                        continue

        if _cancelled():
            return {"error": "cancelled"}

        self.operator._error = (
            f"Failed to retrieve GLB from ComfyUI. "
            f"The workflow completed but the output file could not be located. "
            f"Prefix: {unique_prefix}"
        )
        return {"error": self.operator._error}

