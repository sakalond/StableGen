"""Shared utilities for generation operators (ComfyUIGenerate, Trellis2Generate)."""

import os
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import json
import math
import mimetypes
import re
import tempfile
import time
import traceback
import uuid
import requests

from .timeout_config import get_timeout
from .generation_status import reporting_upload

def redraw_ui(context):
    """Redraws the UI to reflect changes in the operator's progress and status."""
    for area in context.screen.areas:
        area.tag_redraw()


def setup_studio_lighting(context, scale=1.0):
    """Create a three-point studio lighting rig (key, fill, rim).

    Re-usable from any generation mode (TRELLIS.2, PBR decomposition, etc.).

    Args:
        context: Blender context.
        scale: Scene scale factor — lights are placed at ``scale * 2.5`` distance.
    """
    S = max(scale, 0.5)
    dist = S * 2.5

    light_defs = [
        ("SG_Key",  200, 1.5 * S, (1.0, 0.96, 0.90),   45, 40),
        ("SG_Fill",  80, 2.5 * S, (0.90, 0.94, 1.0),   -60, 15),
        ("SG_Rim",  120, 0.8 * S, (1.0, 1.0, 1.0),     170, 55),
    ]

    collection = context.collection
    created = []

    for name, power, size, color, az_deg, el_deg in light_defs:
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        az = math.radians(az_deg)
        el = math.radians(el_deg)
        x = dist * math.cos(el) * math.sin(az)
        y = -dist * math.cos(el) * math.cos(az)
        z = dist * math.sin(el)

        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_data.energy = power
        light_data.size = size
        light_data.color = color

        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        collection.objects.link(light_obj)

        light_obj.location = (x, y, z)
        direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector((x, y, z))
        rot = direction.to_track_quat('-Z', 'Y')
        light_obj.rotation_euler = rot.to_euler()

        created.append(light_obj)

    print(f"[StableGen] Studio lighting created: {[o.name for o in created]}")
    return created


def _pbr_setup_studio_lights(context, to_texture):
    """Calculate scene scale from target objects and set up studio lights."""
    max_dim = 1.0
    for obj in to_texture:
        if hasattr(obj, 'dimensions'):
            max_dim = max(max_dim, *obj.dimensions)
    setup_studio_lighting(context, scale=max_dim)


def upload_image_to_comfyui(server_address, image_path, image_type="input", *, operator=None):
    """
    Uploads an image file to the ComfyUI server's /upload/image endpoint.

    Args:
        server_address (str): The address:port of the ComfyUI server (e.g., "127.0.0.1:8188").
        image_path (str): The local path to the image file to upload.
        image_type (str): The type parameter for the upload (usually "input").
        operator: Optional generation operator whose UI reports this transfer.

    Returns:
        dict: A dictionary containing the server's response (e.g., {'name': 'filename.png', 'subfolder': '', 'type': 'input'})
              Returns None if the upload fails or file doesn't exist.
    """
    if not os.path.exists(image_path):
        # This is expected for optional files, so don't log as an error
        # print(f"Debug: Image file not found at {image_path}, cannot upload.")
        return None
    if not os.path.isfile(image_path):
        print(f"[StableGen] Error: Path exists but is not a file: {image_path}")
        return None

    upload_url = f"http://{server_address}/upload/image"
    print(f"[StableGen] Uploading {os.path.basename(image_path)} to {upload_url}...")

    try:
        with reporting_upload(operator, image_path), open(image_path, 'rb') as f:
            # Determine mime type based on extension
            mime_type = 'application/octet-stream' # Default fallback
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_path.lower().endswith('.webp'):
                mime_type = 'image/webp'
            # Add other types if needed (e.g., .bmp, .gif)

            files = {'image': (os.path.basename(image_path), f, mime_type)}
            # 'overwrite': 'true' prevents errors if the same filename is uploaded again
            # useful for re-running generations with the same intermediate files.
            data = {'overwrite': 'true', 'type': image_type}

            # Increased timeout for potentially large images or slow networks
            response = requests.post(upload_url, files=files, data=data, timeout=get_timeout('transfer'))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        print(f"[StableGen]   Upload successful for '{os.path.basename(image_path)}'. Server response: {response_data}")

        # Crucial Validation
        if 'name' not in response_data:
             print(f"[StableGen]   Error: ComfyUI upload response for {os.path.basename(image_path)} missing 'name'. Response: {response_data}")
             return None
        # End Validation

        return response_data # Should contain 'name', often 'subfolder', 'type'

    except requests.exceptions.Timeout:
        print(f"[StableGen]   Error: Timeout uploading image {os.path.basename(image_path)} to {upload_url}.")
    except requests.exceptions.ConnectionError:
        print(f"[StableGen]   Error: Connection failed when uploading image {os.path.basename(image_path)} to {upload_url}. Is ComfyUI running and accessible?")
    except requests.exceptions.HTTPError as e:
         print(f"[StableGen]   Error: HTTP Error {e.response.status_code} uploading image {os.path.basename(image_path)} to {upload_url}.")
         print(f"[StableGen]   Server response content: {e.response.text}") # Show response body on error
    except requests.exceptions.RequestException as e:
        print(f"[StableGen]   Error uploading image {os.path.basename(image_path)} to {upload_url}: {e}")
    except json.JSONDecodeError:
        print(f"[StableGen]   Error decoding ComfyUI response after uploading {os.path.basename(image_path)}. Response text: {response.text}")
    except Exception as e:
        print(f"[StableGen]   An unexpected error occurred during image upload of {os.path.basename(image_path)}: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors

    return None


class _TransferCancelled(Exception):
    """Internal signal used to stop a transfer without retrying it."""


class _MultipartFileStream:
    """Replayable streaming multipart body for one file.

    ``requests`` normally reads every file in ``files=`` into memory while it
    constructs the multipart body.  This iterable keeps large GLBs on disk and
    reports progress as their bytes are handed to the HTTP connection.
    """

    def __init__(
            self, file_path, mime_type, fields, progress_cb=None,
            cancel_cb=None, attempt=1, chunk_size=1024 * 1024):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.progress_cb = progress_cb
        self.cancel_cb = cancel_cb
        self.attempt = attempt
        self.chunk_size = chunk_size
        self.boundary = f"----StableGen{uuid.uuid4().hex}"

        field_parts = []
        for name, value in fields.items():
            field_parts.append(
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n".encode("utf-8")
            )
        safe_filename = os.path.basename(file_path).replace('"', "_")
        self.prefix = b"".join(field_parts) + (
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{safe_filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
        self.suffix = f"\r\n--{self.boundary}--\r\n".encode("ascii")
        self.content_length = len(self.prefix) + self.file_size + len(self.suffix)

    def __len__(self):
        return self.content_length

    def __iter__(self):
        if self.cancel_cb and self.cancel_cb():
            raise _TransferCancelled()
        yield self.prefix
        sent = 0
        if self.progress_cb:
            self.progress_cb(sent, self.file_size, self.attempt)
        while sent < self.file_size:
            if self.cancel_cb and self.cancel_cb():
                raise _TransferCancelled()
            # Do not retain an open Windows file handle across ``yield``.  If
            # the socket aborts, requests may keep the suspended iterator alive
            # in an exception traceback until the next retry has completed.
            with open(self.file_path, "rb") as handle:
                handle.seek(sent)
                chunk = handle.read(self.chunk_size)
            if not chunk:
                break
            yield chunk
            sent += len(chunk)
            if self.progress_cb:
                self.progress_cb(sent, self.file_size, self.attempt)
        yield self.suffix


def upload_file_to_comfyui(
        server_address, file_path, file_type="input", subfolder="",
        progress_cb=None, cancel_cb=None, max_retries=3):
    """Stream an arbitrary file through ComfyUI's upload route.

    ComfyUI's standard upload API has no append/resume operation.  Interrupted
    uploads are therefore retried from byte zero with ``overwrite=true``.  The
    operation is safe and idempotent because every attempt sends the same file
    to the same destination and ComfyUI only exposes the completed upload.
    """
    if not os.path.isfile(file_path):
        print(f"[StableGen] File upload source does not exist: {file_path}")
        return None

    upload_url = f"http://{server_address}/upload/image"
    filename = os.path.basename(file_path)
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    if filename.lower().endswith(".glb"):
        mime_type = "model/gltf-binary"
    print(
        f"[StableGen] Uploading {filename} to ComfyUI "
        f"input/{subfolder or ''} ..."
    )

    last_error = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            body = _MultipartFileStream(
                file_path,
                mime_type,
                {
                    "overwrite": "true",
                    "type": file_type,
                    "subfolder": subfolder,
                },
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                attempt=attempt,
            )
            response = requests.post(
                upload_url,
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={body.boundary}",
                    "Content-Length": str(len(body)),
                },
                timeout=get_timeout("transfer"),
            )
            response.raise_for_status()
            result = response.json()
            if not result.get("name"):
                print(
                    "[StableGen] ComfyUI file upload response did not include a "
                    f"filename: {result}"
                )
                return None
            print(f"[StableGen] File upload successful: {result}")
            return result
        except _TransferCancelled:
            print(f"[StableGen] Upload cancelled: {filename}")
            return None
        except requests.exceptions.HTTPError as exc:
            # Authentication, upload-size, and other 4xx errors will not become
            # healthy by replaying a very large request.
            last_error = exc
            status = exc.response.status_code if exc.response is not None else 0
            print(
                f"[StableGen] HTTP {status} while uploading {filename}: "
                f"{exc.response.text if exc.response is not None else exc}"
            )
            if 400 <= status < 500:
                break
        except (requests.exceptions.RequestException, OSError, ValueError,
                json.JSONDecodeError) as exc:
            if cancel_cb and cancel_cb():
                print(f"[StableGen] Upload cancelled: {filename}")
                return None
            last_error = exc
            print(
                f"[StableGen] Upload attempt {attempt}/{max_retries} failed "
                f"for {filename}: {exc}"
            )
        if attempt < max_retries:
            time.sleep(min(2 ** (attempt - 1), 4))
    if last_error:
        print(f"[StableGen] Could not upload {filename}: {last_error}")
    return None


def download_file_with_resume(
        url, progress_cb=None, cancel_cb=None, max_retries=5,
        chunk_size=1024 * 1024):
    """Download *url* with validated HTTP Range resumption.

    Partial data is only appended after a matching ``206 Content-Range``.  If a
    server ignores Range, changes the entity, or returns an incompatible range,
    the temporary file is truncated and that response becomes a clean restart.
    ``If-Range`` prevents bytes from two different file versions being joined.
    """
    downloaded = 0
    total_size = None
    validator = None
    last_error = None

    with tempfile.TemporaryFile() as temp_file:
        for attempt in range(1, max(1, int(max_retries)) + 1):
            if cancel_cb and cancel_cb():
                raise _TransferCancelled()

            headers = {}
            requested_offset = downloaded
            if requested_offset:
                headers["Range"] = f"bytes={requested_offset}-"
                if validator:
                    headers["If-Range"] = validator

            try:
                with requests.get(
                        url, headers=headers, stream=True,
                        timeout=get_timeout("transfer")) as response:
                    if response.status_code == 416 and total_size == downloaded:
                        break
                    response.raise_for_status()

                    append_response = requested_offset > 0 and response.status_code == 206
                    if append_response:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.match(
                            r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range)
                        if not match or int(match.group(1)) != requested_offset:
                            temp_file.seek(0)
                            temp_file.truncate()
                            downloaded = 0
                            total_size = None
                            validator = None
                            raise IOError(
                                "Server returned an incompatible Content-Range: "
                                f"{content_range or '<missing>'}"
                            )
                        if match.group(3) != "*":
                            total_size = int(match.group(3))
                    else:
                        # A 200 response to a Range request means Range is not
                        # supported (or If-Range detected a changed entity).
                        temp_file.seek(0)
                        temp_file.truncate()
                        downloaded = 0
                        requested_offset = 0
                        length = response.headers.get("Content-Length")
                        total_size = int(length) if length and length.isdigit() else None

                    validator = (
                        response.headers.get("ETag")
                        or response.headers.get("Last-Modified")
                        or validator
                    )
                    temp_file.seek(downloaded)
                    if progress_cb:
                        progress_cb(downloaded, total_size, attempt)
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if cancel_cb and cancel_cb():
                            raise _TransferCancelled()
                        if not chunk:
                            continue
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            progress_cb(downloaded, total_size, attempt)

                    if total_size is not None and downloaded < total_size:
                        raise IOError(
                            f"HTTP body ended at {downloaded} of {total_size} bytes"
                        )
                    break
            except _TransferCancelled:
                raise
            except (requests.exceptions.RequestException, OSError, ValueError) as exc:
                last_error = exc
                print(
                    f"[StableGen] Download attempt {attempt}/{max_retries} "
                    f"interrupted at {downloaded} bytes: {exc}"
                )
                if attempt < max_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise
        else:
            if last_error:
                raise last_error

        if not downloaded:
            return b""
        temp_file.seek(0)
        data = temp_file.read()
        if total_size is not None and len(data) != total_size:
            raise IOError(
                f"Downloaded size mismatch: received {len(data)} of {total_size} bytes"
            )
        return data
