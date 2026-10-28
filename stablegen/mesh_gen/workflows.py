import os
import json
import email.utils
import ntpath
import posixpath
import re
import uuid
import websocket
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

from ..timeout_config import get_timeout
from ..utils import get_generation_dirs
from ..util.workflow_templates import prompt_text_trellis2, prompt_text_trellis2_shape_only

from io import BytesIO
import numpy as np
from PIL import Image

_ADDON_PKG = __package__.rsplit('.', 1)[0]
_SKINTOKENS_PROGRESS_EVENT = "stablegen_skintokens_progress"


def _progress_percent(data):
    """Return a clamped percentage from a ComfyUI progress payload."""
    try:
        value = float(data.get("value", 0.0))
        maximum = float(data.get("max", 100.0))
    except (TypeError, ValueError):
        return None
    if maximum <= 0:
        return None
    return max(0.0, min(100.0, value * 100.0 / maximum))


def _message_node_matches(data, expected_node):
    node = data.get("node")
    if isinstance(node, (list, tuple)) and node:
        node = node[0]
    return str(node or "") == str(expected_node)


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

    @staticmethod
    def _join_remote_path(base_path, *parts):
        path_module = (
            ntpath
            if re.match(r"^[A-Za-z]:[\\/]", base_path or "")
            or "\\" in (base_path or "")
            else posixpath
        )
        result = base_path
        for part in parts:
            for component in re.split(r"[\\/]", part or ""):
                if component:
                    result = path_module.join(result, component)
        return path_module.normpath(result)

    @staticmethod
    def _comfyui_remote_upload_limit(server_address):
        """Return the host's configured upload limit in bytes when discoverable."""
        try:
            stats_url = f"http://{server_address}/system_stats"
            stats = json.loads(urllib.request.urlopen(
                stats_url, timeout=get_timeout('api')).read())
        except Exception:
            return None
        system = stats.get("system", {}) if isinstance(stats, dict) else {}
        argv = system.get("argv", []) if isinstance(system, dict) else []
        if not isinstance(argv, list):
            return None
        argv = [str(value) for value in argv]
        limit_mb = None
        for index, value in enumerate(argv):
            if value == "--max-upload-size" and index + 1 < len(argv):
                limit_mb = argv[index + 1]
                break
            prefix = "--max-upload-size="
            if value.startswith(prefix):
                limit_mb = value[len(prefix):]
                break
        if limit_mb is None:
            # This is ComfyUI's command-line default when the option is absent.
            limit_mb = 100.0
        try:
            return max(0, int(float(limit_mb) * 1024 * 1024))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _comfyui_base_from_internal_folder_paths(cls, server_address):
        """Infer the ComfyUI base directory from its registered model paths.

        Recent ComfyUI builds expose ``/internal/folder_paths`` even when
        ``/system_stats`` contains only a relative ``main.py`` argv entry and
        omits the input/output roots.  The default model directories live below
        ``<base>/models``.  Extra model paths may point elsewhere, so candidates
        are scored across categories with each category's first (default) path
        weighted most heavily.
        """
        try:
            url = f"http://{server_address}/internal/folder_paths"
            registered = json.loads(urllib.request.urlopen(
                url, timeout=get_timeout('api')).read())
        except Exception as exc:
            print(
                "[StableGen] Could not query ComfyUI internal folder paths: "
                f"{exc}"
            )
            return None
        if not isinstance(registered, dict):
            return None

        scores = {}
        order = []
        for values in registered.values():
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, (list, tuple)):
                continue
            category_candidates = set()
            for index, value in enumerate(values):
                if not isinstance(value, str):
                    continue
                match = re.match(
                    r"^(.*)[\\/]models(?:[\\/].*)?$",
                    value.strip(),
                    flags=re.IGNORECASE,
                )
                if not match or not match.group(1):
                    continue
                candidate = ntpath.normpath(match.group(1)) if (
                    re.match(r"^[A-Za-z]:[\\/]", value) or "\\" in value
                ) else posixpath.normpath(match.group(1))
                key = candidate.lower() if "\\" in candidate else candidate
                if key in category_candidates:
                    continue
                category_candidates.add(key)
                if key not in scores:
                    scores[key] = [0, candidate]
                    order.append(key)
                scores[key][0] += 3 if index == 0 else 1

        if not scores:
            return None
        best_key = max(order, key=lambda key: scores[key][0])
        base_dir = scores[best_key][1]
        print(
            "[StableGen] Inferred ComfyUI base directory from registered "
            f"model paths: {base_dir}"
        )
        return base_dir

    @classmethod
    def _comfyui_uploaded_file_path(
            cls, server_address, upload_info):
        """Resolve an uploaded input reference to an absolute host filesystem path."""
        for key in ("path", "filepath", "full_path"):
            value = upload_info.get(key)
            if value and (ntpath.isabs(value) or posixpath.isabs(value)):
                return value

        try:
            stats_url = f"http://{server_address}/system_stats"
            stats = json.loads(urllib.request.urlopen(
                stats_url, timeout=get_timeout('api')).read())
        except Exception as exc:
            print(f"[StableGen] Could not query ComfyUI input directory: {exc}")
            return None

        system = stats.get("system", {}) if isinstance(stats, dict) else {}
        input_dir = None
        for container in (stats, system):
            if not isinstance(container, dict):
                continue
            for key in ("input_dir", "input_directory"):
                if container.get(key):
                    input_dir = container[key]
                    break
            if input_dir:
                break

        argv = system.get("argv", []) if isinstance(system, dict) else []
        argv = [str(value) for value in argv] if isinstance(argv, list) else []

        def argument_value(option):
            for index, value in enumerate(argv):
                if value == option and index + 1 < len(argv):
                    return argv[index + 1]
                prefix = option + "="
                if value.startswith(prefix):
                    return value[len(prefix):]
            return None

        base_dir = argument_value("--base-directory")
        if not input_dir:
            input_dir = argument_value("--input-directory")
        if input_dir and not (ntpath.isabs(input_dir) or posixpath.isabs(input_dir)):
            if base_dir:
                input_dir = cls._join_remote_path(base_dir, input_dir)
            elif argv and (ntpath.isabs(argv[0]) or posixpath.isabs(argv[0])):
                script_dir = ntpath.dirname(argv[0]) if "\\" in argv[0] else posixpath.dirname(argv[0])
                input_dir = cls._join_remote_path(script_dir, input_dir)
        if not input_dir and base_dir:
            input_dir = cls._join_remote_path(base_dir, "input")
        if not input_dir:
            output_dir = stats.get("output_dir") or system.get("output_dir")
            if output_dir:
                path_module = ntpath if "\\" in output_dir else posixpath
                input_dir = cls._join_remote_path(
                    path_module.dirname(output_dir), "input")
        if not input_dir:
            inferred_base = cls._comfyui_base_from_internal_folder_paths(
                server_address)
            if inferred_base:
                input_dir = cls._join_remote_path(inferred_base, "input")
        if not input_dir and argv and (ntpath.isabs(argv[0]) or posixpath.isabs(argv[0])):
            path_module = ntpath if "\\" in argv[0] else posixpath
            input_dir = cls._join_remote_path(
                path_module.dirname(argv[0]), "input")
        if not input_dir:
            # ComfyUI normally starts with its base directory as the process
            # working directory.  A relative reference is preferable to
            # inventing an absolute host path and remains correct through
            # localhost tunnels such as zrok/SSH.  Custom input-directory users
            # are handled by the explicit metadata and argv branches above.
            input_dir = "input"
            print(
                "[StableGen] ComfyUI did not expose an absolute input directory; "
                "using a process-relative input path"
            )

        return cls._join_remote_path(
            input_dir,
            upload_info.get("subfolder", ""),
            upload_info.get("name", ""),
        )

    @staticmethod
    def _is_trellis2_mesh_output(value):
        """Return True when *value* looks like a TRELLIS.2 mesh output file."""
        return isinstance(value, str) and value.lower().endswith(('.glb', '.obj', '.ply'))

    @classmethod
    def _extract_trellis2_file_refs(cls, value):
        """Collect mesh output references from nested ComfyUI output metadata."""
        refs = []
        seen = set()

        def add_ref(ref):
            path = ref.get('path', '')
            filename = ref.get('filename', '')
            if path and not cls._is_trellis2_mesh_output(path):
                return
            if filename and not cls._is_trellis2_mesh_output(filename):
                return
            if not path and not filename:
                return

            key = (
                path,
                filename,
                ref.get('subfolder', ''),
                ref.get('type', 'output'),
            )
            if key in seen:
                return
            seen.add(key)
            refs.append(ref)

        def visit(item):
            if isinstance(item, str):
                text = item.strip()
                if cls._is_trellis2_mesh_output(text):
                    add_ref({'path': text})
                return

            if isinstance(item, dict):
                filename = item.get('filename') or item.get('name')
                if isinstance(filename, str):
                    filename = filename.strip()
                    if cls._is_trellis2_mesh_output(filename):
                        add_ref({
                            'filename': filename,
                            'subfolder': str(item.get('subfolder') or ''),
                            'type': str(item.get('type') or 'output'),
                        })

                for key, child in item.items():
                    if key in ('filename', 'name', 'subfolder', 'type'):
                        continue
                    visit(child)
                return

            if isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        return refs

    @staticmethod
    def _path_looks_absolute(path):
        """Return True for POSIX or Windows absolute-looking paths."""
        normalized = path.replace('\\', '/')
        return (
            os.path.isabs(path)
            or normalized.startswith('/')
            or (len(normalized) > 1 and normalized[1] == ':')
        )

    @classmethod
    def _comfyui_view_url_for_file_ref(cls, server_address, ref):
        """Build a ComfyUI /view URL from a file reference."""
        filename = ref.get('filename') or ''
        subfolder = ref.get('subfolder') or ''
        file_type = ref.get('type') or 'output'

        if not filename:
            path = ref.get('path') or ''
            if not path:
                return None
            normalized = path.replace('\\', '/')
            filename = posixpath.basename(normalized)
            if not cls._path_looks_absolute(path):
                subfolder = posixpath.dirname(normalized)

        if not filename:
            return None

        params = {
            'filename': filename,
            'subfolder': subfolder,
            'type': file_type,
        }
        return f"http://{server_address}/view?{urllib.parse.urlencode(params)}"

    @staticmethod
    def _describe_trellis2_file_ref(ref):
        if ref.get('path'):
            return ref['path']
        subfolder = ref.get('subfolder') or ''
        filename = ref.get('filename') or ''
        return f"{subfolder}/{filename}" if subfolder else filename

    @staticmethod
    def _normalize_trellis2_timezone_offset(offset):
        """Round and validate a timezone offset."""
        if offset is None:
            return None

        total_seconds = offset.total_seconds()
        if abs(total_seconds) > 14 * 60 * 60:
            return None

        # Real-world timezone offsets are 15-minute aligned; inferred offsets
        # can be off by a few seconds because filenames and mtimes are not
        # atomic, so snap to timezone granularity instead of preserving jitter.
        return timedelta(minutes=round(total_seconds / (15 * 60.0)) * 15)

    @classmethod
    def _parse_trellis2_timezone_offset(cls, value, reference_utc=None):
        """Parse a server-provided timezone name or UTC offset."""
        if value is None or isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            number = float(value)
            if abs(number) <= 14:
                seconds = number * 60 * 60
            elif abs(number) <= 14 * 60:
                seconds = number * 60
            elif abs(number) <= 14 * 60 * 60:
                seconds = number
            else:
                return None
            return cls._normalize_trellis2_timezone_offset(timedelta(seconds=seconds))

        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        upper = text.upper()
        if upper in ('UTC', 'GMT', 'Z'):
            return timedelta(0)

        match = re.search(
            r'(?:UTC|GMT)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?',
            text,
            re.IGNORECASE,
        )
        if match:
            sign = -1 if match.group(1) == '-' else 1
            hours = int(match.group(2))
            minutes = int(match.group(3) or '0')
            if hours <= 14 and minutes < 60:
                return cls._normalize_trellis2_timezone_offset(
                    timedelta(minutes=sign * (hours * 60 + minutes))
                )

        try:
            from zoneinfo import ZoneInfo

            base = reference_utc or datetime.now(timezone.utc)
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            offset = base.astimezone(ZoneInfo(text)).utcoffset()
            return cls._normalize_trellis2_timezone_offset(offset)
        except Exception:
            return None

    @classmethod
    def _extract_trellis2_timezone_offsets(cls, value, reference_utc=None):
        """Collect explicit timezone offsets from nested server metadata."""
        offsets = []
        seen = set()

        def add(offset):
            offset = cls._normalize_trellis2_timezone_offset(offset)
            if offset is None:
                return
            key = int(offset.total_seconds())
            if key in seen:
                return
            seen.add(key)
            offsets.append(offset)

        def key_looks_timezone_related(key):
            key = key.lower().replace('-', '_')
            return (
                key in (
                    'timezone',
                    'time_zone',
                    'tz',
                    'tzname',
                    'utc_offset',
                    'gmt_offset',
                    'timezone_offset',
                    'server_timezone',
                    'server_tz',
                    'server_utc_offset',
                )
                or key.endswith('_timezone')
                or key.endswith('_utc_offset')
                or key.endswith('_gmt_offset')
            )

        def visit(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    if isinstance(key, str) and key_looks_timezone_related(key):
                        add(cls._parse_trellis2_timezone_offset(child, reference_utc))
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        return offsets

    @staticmethod
    def _timestamp_from_trellis2_file_ref(ref):
        filename = ref.get('filename') or ''
        if not filename and ref.get('path'):
            filename = posixpath.basename(ref['path'].replace('\\', '/'))

        match = re.search(
            r'_(\d{8}_\d{6})\.(?:glb|obj|ply)$',
            filename,
            re.IGNORECASE,
        )
        if not match:
            return None

        try:
            return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            return None

    @staticmethod
    def _get_trellis2_view_last_modified(view_url):
        """Return the Last-Modified header from a ComfyUI /view URL."""
        request_attempts = [
            urllib.request.Request(view_url, method='HEAD'),
            urllib.request.Request(view_url, headers={'Range': 'bytes=0-0'}),
        ]

        for req in request_attempts:
            try:
                with urllib.request.urlopen(req, timeout=get_timeout('api')) as response:
                    modified_header = response.headers.get('Last-Modified')
                    if modified_header:
                        return modified_header
            except Exception:
                continue

        return None

    @classmethod
    def _infer_trellis2_timezone_offsets_from_history(cls, server_address, history):
        """Infer server-local UTC offsets from output filenames and mtimes."""
        offsets = []
        seen_offsets = set()
        seen_urls = set()
        attempted = 0
        checked = 0

        for file_ref in cls._extract_trellis2_file_refs(history):
            if attempted >= 16 or checked >= 8:
                break

            local_timestamp = cls._timestamp_from_trellis2_file_ref(file_ref)
            if local_timestamp is None:
                continue

            view_url = cls._comfyui_view_url_for_file_ref(server_address, file_ref)
            if not view_url or view_url in seen_urls:
                continue
            seen_urls.add(view_url)
            attempted += 1

            try:
                modified_header = cls._get_trellis2_view_last_modified(view_url)
                if not modified_header:
                    continue

                modified = email.utils.parsedate_to_datetime(modified_header)
                if modified.tzinfo is None:
                    modified = modified.replace(tzinfo=timezone.utc)
                modified_utc = modified.astimezone(timezone.utc).replace(tzinfo=None)

                offset = cls._normalize_trellis2_timezone_offset(local_timestamp - modified_utc)
                if offset is None:
                    continue
                key = int(offset.total_seconds())
                if key in seen_offsets:
                    continue
                seen_offsets.add(key)
                offsets.append(offset)
            except Exception:
                continue

            checked += 1

        return offsets

    @staticmethod
    def _get_trellis2_prompt_completion_utc(history_entry):
        """Return the prompt completion timestamp as naive UTC."""
        try:
            messages = history_entry.get('status', {}).get('messages', [])
            timestamp_ms = None
            for message in messages:
                if (
                    isinstance(message, (list, tuple))
                    and len(message) >= 2
                    and message[0] == 'execution_success'
                    and isinstance(message[1], dict)
                ):
                    timestamp_ms = message[1].get('timestamp')
                    break

            if timestamp_ms is None and messages:
                last_message = messages[-1]
                if isinstance(last_message, (list, tuple)) and len(last_message) >= 2:
                    timestamp_ms = last_message[1].get('timestamp')

            if timestamp_ms is None:
                return None

            return datetime.fromtimestamp(
                float(timestamp_ms) / 1000.0,
                timezone.utc,
            ).replace(tzinfo=None)
        except Exception:
            return None

    @staticmethod
    def _extract_trellis2_output_text(history_entry, node_id):
        """Extract a string output from a ComfyUI history entry."""
        try:
            output = history_entry.get('outputs', {}).get(node_id, {})
            for key in ('text', 'STRING', 'string', 'value'):
                value = output.get(key)
                if isinstance(value, list) and value:
                    value = value[0]
                if isinstance(value, str) and value:
                    return value
        except Exception:
            return None
        return None

    @classmethod
    def _fetch_trellis2_server_time_string_offset(cls, server_address):
        """Ask a lightweight ComfyUI time node for server-local time."""
        try:
            object_info = json.loads(urllib.request.urlopen(
                f"http://{server_address}/object_info",
                timeout=get_timeout('api'),
            ).read())
            if (
                'Time String (WLSH)' not in object_info
                or 'ShowText|pysssss' not in object_info
            ):
                return None

            body = json.dumps({
                'client_id': str(uuid.uuid4()),
                'prompt': {
                    '1': {
                        'class_type': 'Time String (WLSH)',
                        'inputs': {'style': '%Y-%m-%d-%H%M%S'},
                    },
                    '2': {
                        'class_type': 'ShowText|pysssss',
                        'inputs': {'text': ['1', 0]},
                    },
                },
            }).encode('utf-8')
            req = urllib.request.Request(
                f"http://{server_address}/prompt",
                data=body,
                method='POST',
                headers={'Content-Type': 'application/json'},
            )
            queued = json.loads(urllib.request.urlopen(
                req,
                timeout=get_timeout('api'),
            ).read())
            prompt_id = queued.get('prompt_id')
            if not prompt_id:
                return None

            import time

            history_entry = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                history = json.loads(urllib.request.urlopen(
                    f"http://{server_address}/history/{urllib.parse.quote(prompt_id)}",
                    timeout=get_timeout('api'),
                ).read())
                history_entry = history.get(prompt_id)
                if history_entry:
                    completed = history_entry.get('status', {}).get('completed')
                    if completed:
                        break
                time.sleep(0.2)

            if not history_entry:
                return None

            server_local_text = cls._extract_trellis2_output_text(history_entry, '2')
            if not server_local_text:
                return None

            server_local = datetime.strptime(server_local_text, '%Y-%m-%d-%H%M%S')
            completed_utc = cls._get_trellis2_prompt_completion_utc(history_entry)
            if completed_utc is None:
                return None

            return cls._normalize_trellis2_timezone_offset(server_local - completed_utc)
        except Exception as e:
            print(f"[TRELLIS2] Server timezone probe via Time String failed: {e}")
            return None

    @classmethod
    def _fetch_trellis2_server_timezone_offsets(cls, server_address, reference_utc=None):
        """Fetch or infer actual ComfyUI server-local UTC offsets."""
        offsets = []
        seen = set()

        def add(source, offset):
            offset = cls._normalize_trellis2_timezone_offset(offset)
            if offset is None:
                return
            key = int(offset.total_seconds())
            if key in seen:
                return
            seen.add(key)
            offsets.append((source, offset))

        try:
            req = urllib.request.Request(
                f"http://{server_address}/system_stats",
                method='GET',
            )
            with urllib.request.urlopen(req, timeout=get_timeout('api')) as response:
                headers = response.headers
                body = response.read()

            for key in (
                'X-Server-Timezone',
                'X-Timezone',
                'Timezone',
                'X-Server-UTC-Offset',
                'X-UTC-Offset',
                'X-Timezone-Offset',
                'X-GMT-Offset',
            ):
                add(
                    f"header:{key}",
                    cls._parse_trellis2_timezone_offset(headers.get(key), reference_utc),
                )

            try:
                system_stats = json.loads(body.decode('utf-8'))
                for offset in cls._extract_trellis2_timezone_offsets(system_stats, reference_utc):
                    add('system_stats', offset)
            except Exception:
                pass
        except Exception as e:
            print(f"[TRELLIS2] Server timezone fetch from /system_stats failed: {e}")

        if not offsets:
            add('time-string-prompt', cls._fetch_trellis2_server_time_string_offset(server_address))

        if not offsets:
            try:
                history_url = f"http://{server_address}/history"
                history = json.loads(urllib.request.urlopen(
                    history_url, timeout=get_timeout('api')).read())
                for offset in cls._infer_trellis2_timezone_offsets_from_history(server_address, history):
                    add('history+last-modified', offset)
            except Exception as e:
                print(f"[TRELLIS2] Server timezone inference from history failed: {e}")

        return offsets

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

    def generate_skintokens_autorig_file(self, context, input_glb_path):
        """Run the StableGen SkinTokens node on a GLB path accessible to ComfyUI."""
        server_address = context.preferences.addons[_ADDON_PKG].preferences.server_address
        scene = context.scene

        def _set_phase(phase, label, percent, detail=None):
            if hasattr(self.operator, '_set_pipeline_phase'):
                self.operator._set_pipeline_phase(phase, label, percent, detail)
            else:
                self.operator._progress = percent
                self.operator._stage = detail or label

        def _transfer_detail(action, completed, total, attempt):
            completed_mb = completed / (1024 * 1024)
            if total:
                total_mb = total / (1024 * 1024)
                percent = completed * 100.0 / total
                detail = f"{action}: {completed_mb:.1f}/{total_mb:.1f} MiB"
            else:
                percent = 0.0
                detail = f"{action}: {completed_mb:.1f} MiB"
            if attempt > 1:
                detail += f" (attempt {attempt})"
            return percent, detail

        if not getattr(scene, 'skintokens_autorig_available', False):
            return {
                "error": (
                    "StableGenSkinTokensAutoRig node was not detected in ComfyUI. "
                    "Use the server refresh button after installing the SkinTokens node."
                )
            }
        if not self._check_server_alive(server_address):
            return {"error": "ComfyUI server is not responding. Please restart it and try again."}

        # Always upload standalone auto-rig inputs.  An address that looks local
        # can still terminate on another computer through zrok, SSH forwarding,
        # a reverse proxy, a container, or a VM.  In those cases passing the
        # Blender-side absolute path makes SkinTokens look for a file that only
        # exists on the client.  Uploading is also valid for a genuinely local
        # ComfyUI instance and gives both cases the same deterministic behavior.
        from .._generator_utils import (
            _TransferCancelled,
            download_file_with_resume,
            upload_file_to_comfyui,
        )

        upload_limit = self._comfyui_remote_upload_limit(server_address)
        source_size = os.path.getsize(input_glb_path)
        if upload_limit and source_size > upload_limit:
            source_mb = source_size / (1024 * 1024)
            limit_mb = upload_limit / (1024 * 1024)
            required_mb = ((int(source_mb) // 64) + 1) * 64
            return {
                "error": (
                    f"The exported GLB is {source_mb:.2f} MB, exceeding the "
                    f"ComfyUI upload limit of {limit_mb:.0f} MB. "
                    "On the ComfyUI host, set --max-upload-size "
                    f"{required_mb} or higher and restart ComfyUI."
                )
            }
        _set_phase(1, "Source Upload", 0, "Starting upload")

        def _upload_progress(completed, total, attempt):
            percent, detail = _transfer_detail(
                "Uploading source", completed, total, attempt)
            _set_phase(1, "Source Upload", percent, detail)

        upload_info = upload_file_to_comfyui(
            server_address,
            input_glb_path,
            file_type="input",
            subfolder="stablegen_autorig",
            progress_cb=_upload_progress,
            cancel_cb=lambda: getattr(self.operator, '_cancelled', False),
            max_retries=4,
        )
        if not upload_info:
            if getattr(self.operator, '_cancelled', False):
                return {"error": "cancelled"}
            return {
                "error": (
                    "Could not upload the exported GLB to ComfyUI. "
                    "Check the ComfyUI upload-size limit and network connection."
                )
            }
        comfyui_input_path = self._comfyui_uploaded_file_path(
            server_address, upload_info)
        if not comfyui_input_path:
            return {
                "error": (
                    "The GLB was uploaded to ComfyUI, but StableGen could not "
                    "discover the host input directory from /system_stats."
                )
            }
        print(f"[StableGen] Auto-rig source uploaded to: {comfyui_input_path}")
        _set_phase(1, "Source Upload", 100, "Upload complete")

        self._flush_comfyui_vram(server_address, label="Pre-auto-rigging")

        unique_prefix = f"stablegen_autorig_{uuid.uuid4().hex[:8]}"
        prompt = {
            "1": {
                "class_type": "StableGenSkinTokensAutoRig",
                "inputs": {
                    "input_path": comfyui_input_path,
                    "output_prefix": unique_prefix,
                    "install_policy": "auto",
                    "use_transfer": bool(getattr(scene, 'trellis2_autorig_use_transfer', True)),
                    "use_skeleton": bool(getattr(scene, 'trellis2_autorig_use_skeleton', False)),
                    "use_postprocess": bool(getattr(scene, 'trellis2_autorig_use_postprocess', False)),
                    "top_k": int(getattr(scene, 'trellis2_autorig_top_k', 5)),
                    "top_p": float(getattr(scene, 'trellis2_autorig_top_p', 0.95)),
                    "temperature": float(getattr(scene, 'trellis2_autorig_temperature', 1.0)),
                    "repetition_penalty": float(getattr(scene, 'trellis2_autorig_repetition_penalty', 2.0)),
                    "num_beams": int(getattr(scene, 'trellis2_autorig_num_beams', 10)),
                    "model_ckpt": getattr(scene, 'trellis2_autorig_model_ckpt', ''),
                    "hf_path": getattr(scene, 'trellis2_autorig_hf_path', ''),
                },
            }
        }

        client_id = str(uuid.uuid4())
        ws = self._connect_to_websocket(server_address, client_id)
        if ws is None:
            return {"error": getattr(self, '_error', 'Could not connect to ComfyUI WebSocket.')}
        ws.settimeout(get_timeout('mesh_gen'))
        if hasattr(self.operator, '_active_ws'):
            self.operator._active_ws = ws
            type(self.operator)._active_ws = ws

        prompt_id = None
        output_refs = []

        def _cancelled():
            return getattr(self.operator, '_cancelled', False)

        try:
            prompt_id = self._queue_prompt(prompt, client_id, server_address)
            _set_phase(2, "SkinTokens Rigging", 0, "Queued on ComfyUI")

            while True:
                if _cancelled():
                    return {"error": "cancelled"}
                try:
                    out = ws.recv()
                except websocket.WebSocketTimeoutException:
                    print("[StableGen] SkinTokens WebSocket recv timed out; still waiting...")
                    continue
                except (ConnectionError, OSError, Exception) as ws_err:
                    if _cancelled():
                        return {"error": "cancelled"}
                    return {"error": f"ComfyUI connection failed during SkinTokens auto-rigging: {ws_err}"}

                if not isinstance(out, str):
                    continue

                message = json.loads(out)
                if message.get('type') == 'executing':
                    data = message.get('data', {})
                    if data.get('prompt_id') != prompt_id:
                        continue
                    if data.get('node') is None:
                        _set_phase(
                            2, "SkinTokens Rigging", 100,
                            "Rig generation complete")
                        break
                    if data.get('node') == "1":
                        print("[StableGen] Executing node: Auto-Rigging")
                        _set_phase(
                            2, "SkinTokens Rigging",
                            max(self.operator._phase_progress, 1),
                            "Starting SkinTokens")

                elif message.get('type') == 'executed':
                    data = message.get('data', {})
                    if data.get('node') == "1":
                        refs = self._extract_trellis2_file_refs(data.get('output', {}))
                        if refs:
                            output_refs.extend(refs)

                elif message.get('type') == _SKINTOKENS_PROGRESS_EVENT:
                    data = message.get('data', {})
                    if not _message_node_matches(data, "1"):
                        continue
                    if str(data.get('run_id') or '') != unique_prefix:
                        continue
                    percent = _progress_percent(data)
                    stage = str(data.get('stage') or '').strip()
                    detail = str(data.get('detail') or '').strip()
                    if percent is not None:
                        percent = max(self.operator._phase_progress, percent)
                        _set_phase(
                            2, "SkinTokens Rigging", percent,
                            detail or stage or "Auto-Rigging")
                    elif stage:
                        _set_phase(
                            2, "SkinTokens Rigging",
                            self.operator._phase_progress, detail or stage)

                elif message.get('type') == 'progress':
                    data = message.get('data', {})
                    if not _message_node_matches(data, "1"):
                        continue
                    percent = _progress_percent(data)
                    if percent is not None:
                        _set_phase(
                            2, "SkinTokens Rigging",
                            max(self.operator._phase_progress, percent),
                            "Auto-Rigging")

                elif message.get('type') == 'execution_error':
                    error_data = message.get('data', {})
                    error_msg = error_data.get('exception_message', 'Unknown error')
                    return {"error": f"SkinTokens execution error: {error_msg}"}
        finally:
            if hasattr(self.operator, '_active_ws'):
                self.operator._active_ws = None
                type(self.operator)._active_ws = None
            try:
                ws.close()
            except Exception:
                pass

        if not output_refs and prompt_id and not _cancelled():
            try:
                history_url = f"http://{server_address}/history/{prompt_id}"
                history_response = json.loads(urllib.request.urlopen(
                    history_url, timeout=get_timeout('api')).read())
                history_entry = history_response.get(prompt_id, {})
                output_refs = self._extract_trellis2_file_refs(
                    history_entry.get("outputs", {}).get("1", {})
                )
            except Exception as hist_err:
                print(f"[StableGen] SkinTokens history lookup failed: {hist_err}")

        for file_ref in output_refs:
            if _cancelled():
                return {"error": "cancelled"}
            path = file_ref.get('path') or ''
            if path and os.path.isfile(path):
                try:
                    _set_phase(3, "Result Download", 0, "Reading local result")
                    with open(path, 'rb') as handle:
                        data = handle.read()
                    if data:
                        _set_phase(3, "Result Download", 100, "Result ready")
                        return data
                except OSError as read_err:
                    print(f"[StableGen] SkinTokens direct file read failed: {read_err}")

            view_url = self._comfyui_view_url_for_file_ref(server_address, file_ref)
            if not view_url:
                continue
            try:
                _set_phase(3, "Result Download", 0, "Starting download")

                def _download_progress(completed, total, attempt):
                    percent, detail = _transfer_detail(
                        "Downloading result", completed, total, attempt)
                    _set_phase(3, "Result Download", percent, detail)

                data = download_file_with_resume(
                    view_url,
                    progress_cb=_download_progress,
                    cancel_cb=_cancelled,
                    max_retries=6,
                )
                if data:
                    _set_phase(3, "Result Download", 100, "Download complete")
                    return data
            except _TransferCancelled:
                return {"error": "cancelled"}
            except Exception as view_err:
                print(f"[StableGen] SkinTokens HTTP download failed: {view_err}")

        comfyui_output_dir = None
        try:
            system_url = f"http://{server_address}/system_stats"
            system_response = json.loads(urllib.request.urlopen(
                system_url, timeout=get_timeout('api')).read())
            comfyui_output_dir = system_response.get("output_dir")
        except Exception:
            pass

        if comfyui_output_dir and os.path.isdir(comfyui_output_dir):
            try:
                matching_files = sorted(
                    [
                        name for name in os.listdir(comfyui_output_dir)
                        if name.startswith(unique_prefix) and name.lower().endswith(".glb")
                    ],
                    key=lambda name: os.path.getmtime(os.path.join(comfyui_output_dir, name)),
                    reverse=True,
                )
                if matching_files:
                    found_path = os.path.join(comfyui_output_dir, matching_files[0])
                    with open(found_path, 'rb') as handle:
                        data = handle.read()
                    if data:
                        return data
            except Exception as scan_err:
                print(f"[StableGen] SkinTokens output scan failed: {scan_err}")

        if _cancelled():
            return {"error": "cancelled"}
        return {
            "error": (
                "SkinTokens completed, but StableGen could not retrieve the rigged GLB. "
                f"Output prefix: {unique_prefix}"
            )
        }

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

        image_info = upload_image_to_comfyui(server_address, input_image_path, operator=self.operator)

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
        autorig_enabled = bool(getattr(scene, 'trellis2_autorig_enabled', False))
        if autorig_enabled and not getattr(scene, 'skintokens_autorig_available', False):
            return {
                "error": (
                    "SkinTokens auto-rigging is enabled, but the "
                    "StableGenSkinTokensAutoRig node was not detected in ComfyUI."
                )
            }

        # When post-processing is disabled, maximize the decimation target
        # and disable remeshing so the user gets the rawest possible mesh.
        use_pp = getattr(scene, 'trellis2_post_processing_enabled', True)
        decimate_method = getattr(scene, 'trellis2_decimate_method', 'server')
        remesh_method = getattr(scene, 'trellis2_remesh_method', 'qdc')
        if autorig_enabled and decimate_method == 'collapse':
            print(
                "[TRELLIS2] SkinTokens auto-rigging runs on the ComfyUI host; "
                "using server-side decimation instead of local collapse decimation"
            )
            decimate_method = 'server'

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

        if autorig_enabled:
            rig_node_id = str(
                max(int(node_id) for node_id in prompt.keys() if str(node_id).isdigit()) + 1
            )
            prompt[rig_node_id] = {
                "class_type": "StableGenSkinTokensAutoRig",
                "inputs": {
                    "input_glb": [NODES[export_node_key], 0],
                    "input_path": "",
                    "output_prefix": f"{unique_prefix}_skintokens",
                    "install_policy": "auto",
                    "use_transfer": bool(getattr(scene, 'trellis2_autorig_use_transfer', True)),
                    "use_skeleton": bool(getattr(scene, 'trellis2_autorig_use_skeleton', False)),
                    "use_postprocess": bool(getattr(scene, 'trellis2_autorig_use_postprocess', False)),
                    "top_k": int(getattr(scene, 'trellis2_autorig_top_k', 5)),
                    "top_p": float(getattr(scene, 'trellis2_autorig_top_p', 0.95)),
                    "temperature": float(getattr(scene, 'trellis2_autorig_temperature', 1.0)),
                    "repetition_penalty": float(getattr(scene, 'trellis2_autorig_repetition_penalty', 2.0)),
                    "num_beams": int(getattr(scene, 'trellis2_autorig_num_beams', 10)),
                    "model_ckpt": getattr(scene, 'trellis2_autorig_model_ckpt', ''),
                    "hf_path": getattr(scene, 'trellis2_autorig_hf_path', ''),
                },
            }
            NODES['autorig'] = rig_node_id
            export_node_key = 'autorig'
            print(f"[TRELLIS2] SkinTokens auto-rigging enabled via node {rig_node_id}")

        # Save prompt for debugging
        revision_dir = get_generation_dirs(context).get("revision", "")
        if revision_dir:
            self._save_prompt_to_file(prompt, revision_dir, workflow="trellis2")

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
            type(self.operator)._active_ws = ws

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
            if autorig_enabled:
                NODE_PROGRESS[NODES['autorig']] = 5

            # Wait for execution to complete via WebSocket
            # Also capture 'executed' events which may contain the output path
            export_node_id = NODES[export_node_key]
            glb_output_refs = []

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
                'autorig':                  'Auto-Rigging',
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
                        if data.get('prompt_id') == prompt_id:
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
                                if autorig_enabled and node_key == 'autorig':
                                    autorig_phase = getattr(self.operator, '_autorig_phase', 0)
                                    if autorig_phase:
                                        if getattr(self.operator, '_current_phase', 0) != autorig_phase:
                                            self.operator._phase_progress = 100
                                            if hasattr(self.operator, '_update_overall'):
                                                self.operator._update_overall()
                                            self.operator._current_phase = autorig_phase
                                            self.operator._phase_progress = 0
                                        total_phases = getattr(self.operator, '_total_phases', autorig_phase)
                                        self.operator._overall_stage = (
                                            f"Phase {autorig_phase}/{total_phases}: Auto-Rigging"
                                        )
                                    self.operator._phase_stage = "Auto-Rigging"
                                else:
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
                                refs = self._extract_trellis2_file_refs(ws_output)
                                if refs:
                                    glb_output_refs.extend(refs)
                                    print(
                                        "[TRELLIS2] Got file reference from WS "
                                        f"executed event: {self._describe_trellis2_file_ref(refs[0])}"
                                    )

                    elif message['type'] == _SKINTOKENS_PROGRESS_EVENT:
                        if not autorig_enabled:
                            continue
                        p_data = message.get('data', {})
                        autorig_node_id = NODES.get('autorig')
                        if not _message_node_matches(p_data, autorig_node_id):
                            continue
                        expected_run_id = f"{unique_prefix}_skintokens"
                        if str(p_data.get('run_id') or '') != expected_run_id:
                            continue

                        autorig_phase = getattr(self.operator, '_autorig_phase', 0)
                        if autorig_phase:
                            if getattr(self.operator, '_current_phase', 0) != autorig_phase:
                                self.operator._current_phase = autorig_phase
                                self.operator._phase_progress = 0
                            total_phases = getattr(
                                self.operator, '_total_phases', autorig_phase)
                            self.operator._overall_stage = (
                                f"Phase {autorig_phase}/{total_phases}: Auto-Rigging"
                            )

                        percent = _progress_percent(p_data)
                        if percent is not None:
                            self.operator._phase_progress = max(
                                self.operator._phase_progress, percent)
                            self.operator._detail_progress = percent
                        stage = str(p_data.get('stage') or 'Auto-Rigging').strip()
                        detail = str(p_data.get('detail') or '').strip()
                        self.operator._phase_stage = stage
                        self.operator._detail_stage = detail or stage
                        if hasattr(self.operator, '_update_overall'):
                            self.operator._update_overall()

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
                            if autorig_enabled and _current_exec_node == 'autorig':
                                autorig_phase = getattr(self.operator, '_autorig_phase', 0)
                                if autorig_phase:
                                    if getattr(self.operator, '_current_phase', 0) != autorig_phase:
                                        self.operator._phase_progress = 0
                                    self.operator._current_phase = autorig_phase
                                    total_phases = getattr(self.operator, '_total_phases', autorig_phase)
                                    self.operator._overall_stage = (
                                        f"Phase {autorig_phase}/{total_phases}: Auto-Rigging"
                                    )
                                self.operator._phase_stage = "Auto-Rigging"
                            else:
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

                        if _current_exec_node == 'autorig':
                            self.operator._phase_progress = max(
                                self.operator._phase_progress, step_progress)
                            self.operator._detail_progress = step_progress
                            if not self.operator._detail_stage:
                                self.operator._detail_stage = "Auto-Rigging"
                            if hasattr(self.operator, '_update_overall'):
                                self.operator._update_overall()
                            continue

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
                type(self.operator)._active_ws = None
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

        glb_file_refs = list(glb_output_refs)  # May already be set from WS event
        prompt_history_entry = None

        # Helper: bail early when the user hits Cancel
        def _cancelled():
            return getattr(self.operator, '_cancelled', False)

        # Strategy 2: Query history for the output path
        if not glb_file_refs and not _cancelled():
            try:
                history_url = f"http://{server_address}/history/{prompt_id}"
                history_response = json.loads(urllib.request.urlopen(
                    history_url, timeout=get_timeout('api')).read())

                if prompt_id in history_response:
                    prompt_history_entry = history_response[prompt_id]
                    outputs = history_response[prompt_id].get("outputs", {})
                    export_output = outputs.get(export_node_id, {})
                    print(f"[TRELLIS2] History output for node {export_node_id}: {export_output}")

                    glb_file_refs = self._extract_trellis2_file_refs(export_output)
                    if glb_file_refs:
                        print(
                            "[TRELLIS2] Got file reference from history: "
                            f"{self._describe_trellis2_file_ref(glb_file_refs[0])}"
                        )
            except Exception as e:
                print(f"[TRELLIS2] History query failed: {e}")

        # Strategy 3: Read directly from disk (works when ComfyUI is local)
        for file_ref in glb_file_refs:
            if _cancelled():
                return {"error": "cancelled"}
            glb_source_path = file_ref.get('path')
            if glb_source_path and os.path.isfile(glb_source_path):
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
        for file_ref in glb_file_refs:
            if _cancelled():
                return {"error": "cancelled"}
            view_url = self._comfyui_view_url_for_file_ref(server_address, file_ref)
            if not view_url:
                continue
            try:
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
        server_utc_now = datetime.now(timezone.utc).replace(tzinfo=None) + clock_offset
        reference_utc = server_utc_now.replace(tzinfo=timezone.utc)
        prompt_completion_utc = self._get_trellis2_prompt_completion_utc(prompt_history_entry)
        timestamp_utc_base = prompt_completion_utc or server_utc_now
        client_utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        client_local_offset = datetime.now() - client_utc_now
        candidate_bases = []

        def add_candidate_base(label, value):
            for _existing_label, existing_value in candidate_bases:
                if abs((value - existing_value).total_seconds()) <= 1:
                    return
            candidate_bases.append((label, value))

        if prompt_completion_utc:
            print(
                "[TRELLIS2] Timestamp scan using prompt completion UTC: "
                f"{prompt_completion_utc.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        add_candidate_base('client-local', timestamp_utc_base + client_local_offset)
        add_candidate_base('utc', timestamp_utc_base)
        if prompt_completion_utc:
            add_candidate_base('current-client-local', datetime.now() + clock_offset)
            add_candidate_base('current-utc', server_utc_now)

        server_timezone_offsets = self._fetch_trellis2_server_timezone_offsets(
            server_address,
            reference_utc,
        )
        if server_timezone_offsets:
            print(
                "[TRELLIS2] Server timezone candidates: "
                + ", ".join(
                    f"{source}=UTC{offset.total_seconds() / 3600.0:+.2f}"
                    for source, offset in server_timezone_offsets
                )
            )
            for source, offset in server_timezone_offsets:
                hours = offset.total_seconds() / 3600.0
                add_candidate_base(
                    f'server-local({source}, UTC{hours:+.2f})',
                    timestamp_utc_base + offset,
                )
        else:
            print(
                "[TRELLIS2] Server timezone was not exposed by ComfyUI "
                "and could not be inferred from history metadata"
            )

        print(
            "[TRELLIS2] Timestamp scan candidate clocks: "
            + ", ".join(label for label, _value in candidate_bases)
        )

        for delta_seconds in range(0, scan_range):
            if _cancelled():
                return {"error": "cancelled"}
            deltas = [timedelta(0)] if delta_seconds == 0 else [
                timedelta(seconds=-delta_seconds),
                timedelta(seconds=delta_seconds),
            ]
            for delta in deltas:
                for _label, base_now in candidate_bases:
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
