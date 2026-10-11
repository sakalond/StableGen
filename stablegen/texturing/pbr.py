"""PBR decomposition pipeline — extracted as a mixin for ComfyUIGenerate.

The `_PBRMixin` class provides all PBR-related instance methods.  
`ComfyUIGenerate` inherits from this mixin so existing `self.xxx()`  
call sites remain unchanged.
"""

import os
import json
import io
import math
import hashlib
import traceback
from statistics import median
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import bpy  # pylint: disable=import-error
from ..utils import get_file_path, get_dir_path, get_generation_dirs
from ..timeout_config import get_timeout

_ADDON_PKG = __package__.rsplit('.', 1)[0]


class _PBRMixin:
    """Mixin providing PBR decomposition methods to ComfyUIGenerate."""

    @staticmethod
    def _pbr_settings_hash(settings_dict):
        """Return a short hex hash for a dict of PBR settings."""
        import hashlib
        raw = json.dumps(settings_dict, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _pbr_per_map_settings(self, context):
        """Compute per-map settings dicts that affect each PBR output.

        Returns ``{map_name: {setting_key: value, ...}}`` for every
        map type.  Two dicts compare equal iff the same generation would
        produce identical output.

        Tiling modes are resolved to effective booleans so that e.g.
        ``selective`` (albedo-only) and ``custom`` with only albedo
        enabled produce identical hashes for every map.
        """
        scene = context.scene
        # ── Common model settings (affect all server-side outputs) ────
        common = {
            'resolution': getattr(scene, 'pbr_processing_resolution', 768),
            'native_res': getattr(scene, 'pbr_use_native_resolution', True),
            'denoise_steps': getattr(scene, 'pbr_denoise_steps', 4),
            'ensemble_size': getattr(scene, 'pbr_ensemble_size', 1),
        }

        albedo_source = getattr(scene, 'pbr_albedo_source', 'delight')

        # ── Resolve effective per-map tiling booleans ─────────────────
        tile_mode = getattr(scene, 'pbr_tiling', 'off')
        tile_grid = getattr(scene, 'pbr_tile_grid', 2)
        tile_superres = getattr(scene, 'pbr_tile_superres', False)

        tile_albedo = getattr(scene, 'pbr_tile_albedo', True) if tile_mode == 'custom' else (tile_mode in ('selective', 'all'))
        tile_material = getattr(scene, 'pbr_tile_material', False) if tile_mode == 'custom' else (tile_mode == 'all')
        tile_normal = getattr(scene, 'pbr_tile_normal', False) if tile_mode == 'custom' else (tile_mode == 'all')
        tile_height = getattr(scene, 'pbr_tile_height', False) if tile_mode == 'custom' else (tile_mode == 'all')
        tile_emission = getattr(scene, 'pbr_tile_emission', False) if tile_mode == 'custom' else False

        def tiling_dict(is_tiled):
            """Return tiling sub-dict; grid/superres only matter when tiled."""
            if is_tiled:
                result = {'tiled': True, 'tile_grid': tile_grid,
                          'tile_superres': tile_superres}
                inference = getattr(scene, 'pbr_tile_inference', 'SOURCE')
                if inference != 'SOURCE':
                    # Only recorded when it differs from the old behaviour, so maps made
                    # before this option existed still count as up to date.
                    result['tile_inference'] = inference
                return result
            return {'tiled': False}

        settings = {}

        settings['albedo'] = {
            **common,
            **tiling_dict(tile_albedo),
            'source': albedo_source,
            'delight_strength': getattr(scene, 'pbr_delight_strength', 1.0) if albedo_source == 'delight' else None,
            'auto_saturation': getattr(scene, 'pbr_albedo_auto_saturation', False),
            'saturation_mode': getattr(scene, 'pbr_albedo_saturation_mode', 'MEDIAN') if getattr(scene, 'pbr_albedo_auto_saturation', False) else None,
        }
        settings['roughness'] = {
            **common,
            **tiling_dict(tile_material),
        }
        settings['metallic'] = {
            **common,
            **tiling_dict(tile_material),
        }
        settings['normal'] = {
            **common,
            **tiling_dict(tile_normal),
        }
        settings['height'] = {
            **common,
            **tiling_dict(tile_height),
        }

        emission_method = getattr(scene, 'pbr_emission_method', 'residual')
        emission_base = {
            'method': emission_method,
            'threshold': getattr(scene, 'pbr_emission_threshold', 0.2),
        }
        if emission_method == 'hsv':
            emission_base.update({
                'sat_min': getattr(scene, 'pbr_emission_saturation_min', 0.5),
                'val_min': getattr(scene, 'pbr_emission_value_min', 0.85),
                'bloom': getattr(scene, 'pbr_emission_bloom', 5.0),
            })
        else:
            # Residual method uses a model pass
            emission_base.update({
                **common,
                **tiling_dict(tile_emission),
            })
        settings['emission'] = emission_base

        return settings

    def _pbr_sidecar_path(self, context):
        """Return the path for the single PBR settings sidecar JSON file."""
        dirs = get_generation_dirs(context)
        base_dir = dirs["pbr"]
        material_suffix = f"-{self._material_id}" if self._material_id is not None else ""
        return os.path.join(base_dir, f"pbr_settings{material_suffix}.json")

    def _save_pbr_settings(self, context, camera_id):
        """Write current per-map settings hashes for *camera_id* into
        the shared sidecar JSON file (merges with existing data)."""
        settings = self._pbr_per_map_settings(context)
        hashes = {k: self._pbr_settings_hash(v) for k, v in settings.items()}
        path = self._pbr_sidecar_path(context)
        # Load existing data to merge
        data = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                pass
        data[str(camera_id)] = hashes
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=1)
        except Exception as e:
            print(f"[StableGen] Warning: could not write PBR settings "
                  f"sidecar {path}: {e}")

    def _load_pbr_settings(self, context, camera_id):
        """Load stored per-map settings hashes for *camera_id* from the
        shared sidecar JSON file.

        Returns a dict ``{map_name: hash_string}`` or ``None`` if no
        sidecar or camera entry exists.
        """
        path = self._pbr_sidecar_path(context)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data.get(str(camera_id))
        except Exception:
            return None

    def _find_existing_pbr_maps(self, context, camera_id):
        """Check which enabled PBR maps already exist on disk for a camera.

        Returns ``(existing, missing)`` where *existing* is a dict
        ``{map_name: file_path}`` of maps already on disk **whose settings
        hash matches the current configuration**, and *missing* is a set of
        map names that are enabled but either not found or stale.
        """
        scene = context.scene
        existing = {}
        missing = set()

        map_checks = []
        if getattr(scene, 'pbr_map_albedo', True):
            map_checks.append('albedo')
        if getattr(scene, 'pbr_map_roughness', True):
            map_checks.append('roughness')
        if getattr(scene, 'pbr_map_metallic', True):
            map_checks.append('metallic')
        if getattr(scene, 'pbr_map_normal', True):
            map_checks.append('normal')
        if getattr(scene, 'pbr_map_height', False):
            map_checks.append('height')
        if getattr(scene, 'pbr_map_emission', False):
            map_checks.append('emission')

        if not map_checks:
            return existing, missing

        # Load stored hashes and compute current ones for comparison
        stored_hashes = self._load_pbr_settings(context, camera_id)
        current_settings = self._pbr_per_map_settings(context)
        current_hashes = {k: self._pbr_settings_hash(v)
                         for k, v in current_settings.items()}

        for map_name in map_checks:
            path = get_file_path(
                context, "pbr", subtype=map_name,
                camera_id=camera_id, material_id=self._material_id
            )
            if os.path.exists(path):
                # Check if settings have changed since last generation
                if (stored_hashes is not None
                        and stored_hashes.get(map_name) == current_hashes.get(map_name)):
                    existing[map_name] = path
                else:
                    # File exists but settings changed → stale, regenerate
                    missing.add(map_name)
            else:
                missing.add(map_name)

        return existing, missing

    # ── Tiled PBR processing ──────────────────────────────────────────

    @staticmethod
    def _create_tile_blend_mask(h, w, overlap_x, overlap_y,
                                fade_left, fade_right,
                                fade_top, fade_bottom):
        """Return a float32 [H, W, 1] weight mask with cosine fade on inner edges.

        Only the edges that border a neighbouring tile (indicated by the
        ``fade_*`` booleans) get a cosine-weighted ramp.
        ``overlap_x`` controls left/right ramp length,
        ``overlap_y`` controls top/bottom ramp length.
        Image-boundary edges stay at full weight (1.0).
        """
        mask = np.ones((h, w, 1), dtype=np.float32)

        if overlap_x > 0:
            ramp_x = np.linspace(0.0, np.pi, overlap_x, dtype=np.float32)
            ramp_x = (1.0 - np.cos(ramp_x)) * 0.5      # 0 → 1 cosine ease
            if fade_left and overlap_x <= w:
                mask[:, :overlap_x, 0] *= ramp_x[np.newaxis, :]
            if fade_right and overlap_x <= w:
                mask[:, -overlap_x:, 0] *= ramp_x[np.newaxis, ::-1]

        if overlap_y > 0:
            ramp_y = np.linspace(0.0, np.pi, overlap_y, dtype=np.float32)
            ramp_y = (1.0 - np.cos(ramp_y)) * 0.5
            if fade_top and overlap_y <= h:
                mask[:overlap_y, :, 0] *= ramp_y[:, np.newaxis]
            if fade_bottom and overlap_y <= h:
                mask[-overlap_y:, :, 0] *= ramp_y[::-1, np.newaxis]

        return mask

    def _process_model_tiled(self, context, image_path, model_name=None,
                              process_fn=None):
        """Run a model on an N×N grid of overlapping tiles.

        Each tile is **upscaled to the full image's longest edge** before
        processing, so the model spends its full resolution budget on
        only 1/N² of the spatial area — effectively N²× the detail.
        The stitched output is at the **upscaled** resolution (~N× the
        original), producing super-resolution PBR maps.

        Args:
            model_name: Passed to ``generate_pbr_maps()``.
            process_fn: Optional callable ``(context, tile_path) → result``.
                        If given, called instead of ``generate_pbr_maps``.
                        May return ``bytes`` (single image) or ``list[bytes]``.

        Returns the same ``list[bytes]`` as ``generate_pbr_maps()``.
        """
        import tempfile

        OVERLAP = 64  # pixels each tile extends beyond its boundary
        MIN_DIM = 256 # skip tiling if either dimension is too small

        scene = context.scene
        N = getattr(scene, 'pbr_tile_grid', 2)

        src = Image.open(image_path)
        W, H = src.size
        longest = max(W, H)

        if W < MIN_DIM or H < MIN_DIM:
            src.close()
            print(f"[StableGen]     Image {W}×{H} too small for tiling, "
                  f"processing normally")
            if process_fn is not None:
                fb = process_fn(context, image_path)
                return [fb] if isinstance(fb, bytes) else fb
            return self.workflow_manager.generate_pbr_maps(
                context, image_path, model_name=model_name)

        # Compute tile boundaries in original-image coordinates
        x_bounds = [W * c // N for c in range(N + 1)]   # [0, W/N, 2W/N, …, W]
        y_bounds = [H * r // N for r in range(N + 1)]

        tile_info = []  # list of (l, t, r, b, fade_left, fade_right, fade_top, fade_bottom)
        for row in range(N):
            for col in range(N):
                l = x_bounds[col]   - (OVERLAP if col > 0     else 0)
                r = x_bounds[col+1] + (OVERLAP if col < N - 1 else 0)
                t = y_bounds[row]   - (OVERLAP if row > 0     else 0)
                b = y_bounds[row+1] + (OVERLAP if row < N - 1 else 0)
                # Clamp to image bounds
                l = max(l, 0);  r = min(r, W)
                t = max(t, 0);  b = min(b, H)
                tile_info.append((
                    l, t, r, b,
                    col > 0,        # fade_left
                    col < N - 1,    # fade_right
                    row > 0,        # fade_top
                    row < N - 1,    # fade_bottom
                ))

        total_tiles = len(tile_info)

        # Uniform scale factor — upscale tiles so longest edge ≈ full image
        sample_tw = tile_info[0][2] - tile_info[0][0]
        sample_th = tile_info[0][3] - tile_info[0][1]
        sample_longest = max(sample_tw, sample_th)
        scale = longest / sample_longest if sample_longest < longest else 1.0

        # ── Process each tile ─────────────────────────────────
        all_tile_results = []   # list of list[bytes]
        tile_up_sizes = []      # (up_w, up_h) per tile
        tmp_dir = tempfile.gettempdir()

        for i, (l, t, r, b, fl, fr, ft, fb) in enumerate(tile_info):
            tw, th = r - l, b - t
            tile_img = src.crop((l, t, r, b))

            tile_longest = max(tw, th)
            if tile_longest < longest:
                up_w = int(round(tw * scale))
                up_h = int(round(th * scale))
                tile_img = tile_img.resize(
                    (up_w, up_h), Image.LANCZOS)
                print(f"[StableGen]     Tile {i+1}/{total_tiles}:  "
                      f"{tw}×{th} → {up_w}×{up_h}  "
                      f"region ({l},{t})→({r},{b})")
            else:
                up_w, up_h = tw, th
                print(f"[StableGen]     Tile {i+1}/{total_tiles}:  "
                      f"{tw}×{th}  region ({l},{t})→({r},{b})")
            tile_up_sizes.append((up_w, up_h))

            tile_path = os.path.join(tmp_dir, f"sg_tile_{i}.png")
            tile_img.save(tile_path)

            if process_fn is not None:
                result = process_fn(context, tile_path)
                # Normalise to list[bytes] (StableDelight returns bytes)
                if isinstance(result, bytes):
                    result = [result]
            elif getattr(context.scene, 'pbr_tile_inference', 'SOURCE') == 'MODEL':
                # Each tile at the model's inference resolution: the grid then adds
                # detail at the scale Marigold was trained at, rather than feeding it
                # tiles far larger than that.
                result = self.workflow_manager.generate_pbr_maps(
                    context, tile_path, model_name=model_name,
                    processing_resolution=getattr(
                        context.scene, 'pbr_processing_resolution', 768))
            else:
                result = self.workflow_manager.generate_pbr_maps(
                    context, tile_path, model_name=model_name,
                    force_native_resolution=True)

            try:
                os.remove(tile_path)
            except OSError:
                pass

            if isinstance(result, dict):
                print(f"[StableGen]     Tile {i+1} failed, falling back "
                      f"to full-image processing")
                src.close()
                if process_fn is not None:
                    fb = process_fn(context, image_path)
                    return [fb] if isinstance(fb, bytes) else fb
                return self.workflow_manager.generate_pbr_maps(
                    context, image_path, model_name=model_name)

            # Resize model output to expected upscaled tile dims
            resized = []
            for map_bytes in result:
                map_img = Image.open(io.BytesIO(map_bytes))
                if map_img.size != (up_w, up_h):
                    map_img = map_img.resize((up_w, up_h), Image.LANCZOS)
                buf = io.BytesIO()
                map_img.save(buf, format='PNG')
                resized.append(buf.getvalue())
            all_tile_results.append(resized)

        src.close()

        # ── Stitch tiles ──────────────────────────────────────
        superres = getattr(scene, 'pbr_tile_superres', False)

        if superres:
            # Super-resolution: stitch at the upscaled tile size (~N× original)
            out_W = int(round(W * scale))
            out_H = int(round(H * scale))
            scaled_overlap = int(round(OVERLAP * scale))
        else:
            # Original resolution: downscale tiles back before stitching
            out_W, out_H = W, H
            scaled_overlap = OVERLAP  # overlap in original coords

        num_maps = len(all_tile_results[0])
        stitched = []

        for map_idx in range(num_maps):
            canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)
            weight = np.zeros((out_H, out_W, 1), dtype=np.float32)

            for i, (l, t, r, b, fl, fr, ft, fb) in enumerate(tile_info):
                tile_bytes = all_tile_results[i][map_idx]
                tile_img = Image.open(io.BytesIO(tile_bytes)).convert('RGB')

                if superres:
                    up_w, up_h = tile_up_sizes[i]
                    sl = int(round(l * scale))
                    st = int(round(t * scale))
                    sr = min(sl + up_w, out_W)
                    sb = min(st + up_h, out_H)
                else:
                    # Downscale tile back to original crop dimensions
                    orig_w, orig_h = r - l, b - t
                    if tile_img.size != (orig_w, orig_h):
                        tile_img = tile_img.resize(
                            (orig_w, orig_h), Image.LANCZOS)
                    sl, st, sr, sb = l, t, r, b

                tile_arr = np.asarray(tile_img, dtype=np.float32)
                tile_arr = tile_arr[:sb - st, :sr - sl]

                overlap_px = 2 * scaled_overlap
                mask = self._create_tile_blend_mask(
                    sb - st, sr - sl,
                    overlap_x=overlap_px,
                    overlap_y=overlap_px,
                    fade_left=fl, fade_right=fr,
                    fade_top=ft, fade_bottom=fb)

                canvas[st:sb, sl:sr] += tile_arr * mask
                weight[st:sb, sl:sr] += mask

            canvas /= np.maximum(weight, 1e-6)
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            out_img = Image.fromarray(canvas, mode='RGB')
            buf = io.BytesIO()
            out_img.save(buf, format='PNG')
            stitched.append(buf.getvalue())

        sr_label = " (super-res)" if superres else ""
        print(f"[StableGen]     Stitched {N}×{N} output: {out_W}×{out_H}"
              f"{sr_label}")

        return stitched

    def _convert_normals_cam_to_world(self, map_bytes, camera_id):
        """Convert a Marigold camera-space normal map to world space.

        Marigold normals are in camera space (X = right, Y = up,
        Z = toward the camera).  The PNG encodes them as
        ``(N + 1) / 2``, mapping [-1, 1] to [0, 1].

        This method:
          1. Decodes the PNG into an RGB float array.
          2. Converts [0, 1] → [-1, 1].
          3. Rotates each normal by the camera's world rotation
             (Marigold OpenGL convention and Blender camera-local
             convention agree: X-right, Y-up, Z-toward-viewer).
          4. Re-normalises and re-encodes to [0, 1] PNG bytes.

        Returns PNG bytes, or ``None`` on failure.
        """
        try:
            img = Image.open(io.BytesIO(map_bytes)).convert('RGB')
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0,1]

            # Decode to [-1, 1]
            normals = arr * 2.0 - 1.0  # [H, W, 3]  (X, Y, Z) camera-space

            # Marigold (OpenGL) convention:
            #   X = right, Y = up, Z = toward camera (out of screen)
            # Blender camera-local convention:
            #   X = right, Y = up, Z = behind camera (camera looks -Z)
            #
            # These AGREE: a surface facing the camera has Z = +1 in
            # both systems (+Z_local points toward the viewer).
            # No axis flip is needed.

            # Get camera rotation (camera-local → world)
            cam = self._cameras[camera_id]
            cam_rot = np.array(cam.matrix_world.to_3x3(), dtype=np.float32)  # 3×3

            # Rotate all normals: N_world = cam_rot @ N_local
            # Reshape to [H*W, 3] for batch matrix multiply
            H, W, _ = normals.shape
            flat = normals.reshape(-1, 3)           # [H*W, 3]
            flat_world = (cam_rot @ flat.T).T       # [H*W, 3]

            # Re-normalise (avoid div-by-zero)
            norms = np.linalg.norm(flat_world, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            flat_world /= norms

            # Encode back to [0, 1] PNG
            world_normals = flat_world.reshape(H, W, 3)
            encoded = np.clip((world_normals + 1.0) / 2.0, 0.0, 1.0)
            encoded_u8 = (encoded * 255.0).astype(np.uint8)

            out_img = Image.fromarray(encoded_u8, mode='RGB')
            buf = io.BytesIO()
            out_img.save(buf, format='PNG')
            print(f"[StableGen]   Converted normal map to world space for camera {camera_id}")
            return buf.getvalue()
        except Exception as err:
            print(f"[StableGen]   Warning: camera→world normal conversion failed: {err}")
            return None

    # ── PBR post-processing helpers ───────────────────────────────────

    @staticmethod
    def _compute_image_mean_saturation(image_path_or_bytes, sample_limit=100_000, mask=None):
        """Compute the mean HSV saturation for a single image.

        *image_path_or_bytes* can be a file path (str) or raw PNG bytes.
        Skips very dark pixels whose saturation is unreliable. *mask* is an
        optional boolean array (any size; resized to the image) restricting
        the measurement to the object.
        Returns a float in [0, 1], or 0.0 on error.
        """
        try:
            if isinstance(image_path_or_bytes, str):
                img = Image.open(image_path_or_bytes).convert('RGB')
            else:
                img = Image.open(io.BytesIO(image_path_or_bytes)).convert('RGB')

            rgb = np.asarray(img, dtype=np.float32) / 255.0
            value = rgb.max(axis=-1)
            saturation = (value - rgb.min(axis=-1)) / np.maximum(value, 1e-6)
            keep = value > 0.05
            if mask is not None:
                mask_img = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
                if mask_img.size != img.size:
                    mask_img = mask_img.resize(img.size, Image.NEAREST)
                keep &= np.asarray(mask_img) > 127
            values = saturation[keep]
            step = max(1, values.size // sample_limit)
            values = values[::step]
            return float(values.mean()) if values.size else 0.0
        except Exception as err:
            print(f"[StableGen]     Warning: saturation computation failed: {err}")
            return 0.0

    @staticmethod
    def _object_mask(image_path):
        """Return the opaque region of a generated image as a boolean array.

        Generated images keep their colour under transparent pixels (often a
        saturated backdrop), which must not count towards the object's
        saturation. Returns ``None`` when the image has no usable alpha.
        """
        try:
            with Image.open(image_path) as img:
                if 'A' not in img.getbands():
                    return None
                alpha = np.asarray(img.getchannel('A'))
        except Exception:
            return None
        mask = alpha > 127
        if mask.all() or not mask.any():
            return None
        return mask

    def _albedo_saturation_ratio(self, original_path, raw_path):
        """Saturation of the generated image over the raw albedo, measured on the object.

        Returns the ratio clamped to [0.5, 3.0], or ``None`` when the albedo is
        nearly grey and the ratio is meaningless.
        """
        mask = self._object_mask(original_path)
        s_orig = self._compute_image_mean_saturation(original_path, mask=mask)
        s_albedo = self._compute_image_mean_saturation(raw_path, mask=mask)
        if s_albedo < 0.01:
            return None
        return max(0.5, min(s_orig / s_albedo, 3.0))

    def _compute_uniform_saturation(self, context, use_median=True):
        """Compute a uniform auto-saturation ratio across ALL cameras.

        Compares each camera's original rendered image against its raw
        PBR albedo, computes per-camera ratios, then aggregates them
        using the **median** (robust to outlier cameras) or arithmetic
        **mean**, depending on *use_median*.

        Returns a float multiplier (1.0 = no correction needed).
        """
        import statistics

        ratios = []
        num_cameras = len(self._cameras) if hasattr(self, '_cameras') else 0

        for cam_idx in range(num_cameras):
            original_path = get_file_path(
                context, "generated", camera_id=cam_idx,
                material_id=self._material_id)
            raw_path = get_file_path(
                context, "pbr", subtype="raw_albedo",
                camera_id=cam_idx,
                material_id=self._material_id)

            if not os.path.exists(original_path) or not os.path.exists(raw_path):
                continue

            ratio = self._albedo_saturation_ratio(original_path, raw_path)
            if ratio is None:
                continue  # skip grayscale albedos
            ratios.append(ratio)

        if not ratios:
            return 1.0

        if use_median:
            agg = statistics.median(ratios)
            label = "median"
        else:
            agg = sum(ratios) / len(ratios)
            label = "mean"

        print(f"[StableGen] Auto saturation: per-camera ratios = "
              f"{[f'{r:.2f}' for r in ratios]}, {label} = {agg:.2f}")
        return agg

    def _apply_albedo_postprocessing_batch(self, context):
        """Apply auto-saturation correction to all camera albedos.

        Must be called AFTER decomposition saves raw files for all cameras.
        By default computes the averaged saturation ratio across all cameras
        for uniform correction.  With per-camera mode, each camera gets its
        own individual ratio.

        When correction is disabled, restores raw (unprocessed) albedos
        so that a previously corrected active file doesn't persist.
        """
        auto_on = getattr(context.scene, 'pbr_albedo_auto_saturation', True)

        if not auto_on:
            self._restore_raw_albedos(context)
            return

        sat_mode = getattr(context.scene, 'pbr_albedo_saturation_mode', 'MEDIAN')

        if sat_mode == 'PER_CAMERA':
            # Per-camera mode: individual ratio per camera
            self._apply_per_camera_saturation(context)
        else:
            # Uniform mode (MEDIAN or MEAN)
            effective_sat = self._compute_uniform_saturation(
                context, use_median=(sat_mode == 'MEDIAN'))

            if effective_sat == 1.0:
                print("[StableGen] Albedo saturation correction: no-op (ratio ≈ 1.0)")
                self._restore_raw_albedos(context)
                return

            for cam_idx in sorted(self._pbr_maps.keys()):
                cam_maps = self._pbr_maps[cam_idx]
                if 'albedo' not in cam_maps:
                    continue

                raw_path = get_file_path(
                    context, "pbr", subtype="raw_albedo",
                    camera_id=cam_idx,
                    material_id=self._material_id)
                if not os.path.exists(raw_path):
                    continue

                with open(raw_path, 'rb') as f:
                    raw_bytes = f.read()
                processed = self._postprocess_albedo(raw_bytes, effective_sat)
                with open(cam_maps['albedo'], 'wb') as f:
                    f.write(processed)
                print(f"[StableGen]     Corrected albedo saturation cam {cam_idx} "
                      f"(uniform ×{effective_sat:.2f})")

    def _apply_per_camera_saturation(self, context):
        """Apply individual saturation correction for each camera."""
        for cam_idx in sorted(self._pbr_maps.keys()):
            cam_maps = self._pbr_maps[cam_idx]
            if 'albedo' not in cam_maps:
                continue

            raw_path = get_file_path(
                context, "pbr", subtype="raw_albedo",
                camera_id=cam_idx,
                material_id=self._material_id)
            original_path = get_file_path(
                context, "generated", camera_id=cam_idx,
                material_id=self._material_id)

            if not os.path.exists(raw_path) or not os.path.exists(original_path):
                continue

            ratio = self._albedo_saturation_ratio(original_path, raw_path)
            if ratio is None:
                continue

            with open(raw_path, 'rb') as f:
                raw_bytes = f.read()
            processed = self._postprocess_albedo(raw_bytes, ratio)
            with open(cam_maps['albedo'], 'wb') as f:
                f.write(processed)
            print(f"[StableGen]     Corrected albedo saturation cam {cam_idx} "
                  f"(per-camera ×{ratio:.2f})")

    def _restore_raw_albedos(self, context):
        """Copy raw (unprocessed) albedos back to active paths.

        Ensures that disabling saturation correction and reprojecting
        uses the original decomposition output, not a previously corrected file.
        """
        import shutil
        for cam_idx in sorted(self._pbr_maps.keys()):
            cam_maps = self._pbr_maps[cam_idx]
            if 'albedo' not in cam_maps:
                continue
            raw_path = get_file_path(
                context, "pbr", subtype="raw_albedo",
                camera_id=cam_idx,
                material_id=self._material_id)
            if os.path.exists(raw_path) and os.path.exists(cam_maps['albedo']):
                shutil.copy2(raw_path, cam_maps['albedo'])
                print(f"[StableGen]     Restored raw albedo for camera {cam_idx}")

    @staticmethod
    def _postprocess_albedo(image_bytes, saturation=1.0):
        """Apply saturation adjustment to albedo PNG bytes.

        Returns post-processed PNG bytes, or the original bytes unchanged
        if saturation is 1.0 (no-op).
        """
        if saturation == 1.0:
            return image_bytes
        from PIL import Image, ImageEnhance
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = ImageEnhance.Color(img).enhance(saturation)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def _save_raw_copy(self, context, subtype, camera_id, data_bytes):
        """Save an unprocessed (raw) copy of a PBR map for later re-processing."""
        raw_path = get_file_path(
            context, "pbr", subtype=f"raw_{subtype}",
            camera_id=camera_id,
            material_id=self._material_id)
        try:
            with open(raw_path, 'wb') as f:
                f.write(data_bytes)
        except Exception as err:
            print(f"[StableGen]     Warning: failed to save raw {subtype}: {err}")

    def _ensure_raw_copies(self, context, camera_id, existing_maps):
        """Ensure a raw albedo copy exists for backward compatibility.

        For files generated before the saturation correction feature, the
        active albedo IS the unprocessed original.  Copy it to the raw slot
        so that the batch saturation correction can read from it.
        """
        if 'albedo' not in existing_maps:
            return
        raw_path = get_file_path(
            context, "pbr", subtype="raw_albedo",
            camera_id=camera_id,
            material_id=self._material_id)
        if not os.path.exists(raw_path) and os.path.exists(existing_maps['albedo']):
            try:
                import shutil
                shutil.copy2(existing_maps['albedo'], raw_path)
                print(f"[StableGen]     Created missing raw albedo from active file "
                      f"(camera {camera_id})")
            except Exception as err:
                print(f"[StableGen]     Warning: could not create raw albedo copy: {err}")

    def _run_pbr_decomposition_batched(self, context, camera_images,
                                       per_camera_missing=None):
        """Run PBR decomposition **model-first** across all cameras.

        Instead of iterating cameras → models (which forces repeated model
        loading/unloading), this iterates models → cameras so each model's
        weights are loaded once and reused for every camera image.

        Args:
            context: Blender context.
            camera_images: ``{cam_idx: image_path}`` for cameras that need
                processing.
            per_camera_missing: Optional ``{cam_idx: set_of_missing_map_names}``.
                When provided (reproject mode), only the models needed to
                produce the missing maps are executed, and cameras that
                already have a model's outputs are skipped.
        """
        scene = context.scene
        self._stage = "PBR Decomposition"
        self._progress = 0
        num_cams = len(camera_images)
        cam_ids = sorted(camera_images.keys())

        # When per_camera_missing is provided, compute the union of all
        # missing maps to determine which models we actually need.
        if per_camera_missing:
            all_missing = set()
            for missing_set in per_camera_missing.values():
                all_missing |= missing_set
            print(f"[StableGen] Running selective PBR decomposition on "
                  f"{num_cams} camera(s), missing maps: {all_missing}")
        else:
            all_missing = None
            print(f"[StableGen] Running batched PBR decomposition on "
                  f"{num_cams} camera(s)…")

        # Free VRAM before loading PBR models — the previous generation
        # model may still be cached, and Marigold/StableDelight need
        # GPU memory.
        try:
            server_address = context.preferences.addons[
                _ADDON_PKG].preferences.server_address
            self.workflow_manager._flush_comfyui_vram(
                server_address, retries=1, label="PBR pre-load")
        except Exception as e:
            print(f"[StableGen] VRAM flush before PBR failed (non-fatal): {e}")

        want_albedo = getattr(scene, 'pbr_map_albedo', True)
        want_roughness = getattr(scene, 'pbr_map_roughness', True)
        want_metallic = getattr(scene, 'pbr_map_metallic', True)
        want_normal = getattr(scene, 'pbr_map_normal', True)
        want_height = getattr(scene, 'pbr_map_height', False)
        want_emission = getattr(scene, 'pbr_map_emission', False)
        emission_method = getattr(scene, 'pbr_emission_method', 'residual')
        albedo_source = getattr(scene, 'pbr_albedo_source', 'delight')

        use_delight = (want_albedo and albedo_source == 'delight')
        use_lighting_albedo = (want_albedo and albedo_source == 'lighting')

        # ── Determine which Marigold models to run ────────────────────
        # When all_missing is set (selective rerun), only models whose
        # outputs intersect with the missing set are scheduled.
        need_appearance = (
            (want_albedo and albedo_source == 'marigold')
            or want_roughness
            or want_metallic
        )
        need_normals = want_normal
        need_height = want_height
        # IID-Lighting is needed for 'residual' emission OR 'lighting' albedo
        need_iid_lighting = (
            (want_emission and emission_method == 'residual')
            or use_lighting_albedo
        )

        if all_missing is not None:
            # Selective mode: narrow down to only models producing missing maps
            need_appearance = need_appearance and bool(
                all_missing & {'albedo', 'roughness', 'metallic'})
            need_normals = need_normals and 'normal' in all_missing
            need_height = need_height and 'height' in all_missing
            need_iid_lighting = need_iid_lighting and bool(
                all_missing & {'emission', 'albedo'})
            use_delight = use_delight and 'albedo' in all_missing

        models_to_run = []
        if need_appearance:
            models_to_run.append((
                'prs-eth/marigold-iid-appearance-v1-1',
                ['albedo', 'material'],
            ))
        if need_normals:
            models_to_run.append((
                'prs-eth/marigold-normals-lcm-v0-1',
                ['normal'],
            ))
        if need_height:
            models_to_run.append((
                'prs-eth/marigold-depth-lcm-v1-0',
                ['height'],
            ))
        if need_iid_lighting:
            models_to_run.append((
                'prs-eth/marigold-iid-lighting-v1-1',
                ['_lighting_albedo', '_lighting_shading', '_lighting_residual'],
            ))

        # Count extra post-processing steps (HSV emission)
        extra_emission_steps = 0
        if want_emission and emission_method == 'hsv':
            extra_emission_steps = 1

        total_model_steps = (len(models_to_run)
                             + (1 if use_delight else 0)
                             + extra_emission_steps)
        if total_model_steps == 0:
            print("[StableGen] No PBR maps enabled, skipping decomposition")
            return

        # Activate PBR progress tracking for the UI
        self._pbr_active = True
        self._pbr_step = 0
        self._pbr_total_steps = total_model_steps
        self._pbr_cam_total = num_cams

        # Ensure every camera has an entry in _pbr_maps
        for cam_idx in cam_ids:
            if cam_idx not in self._pbr_maps:
                self._pbr_maps[cam_idx] = {}

        # ── Tiling settings ───────────────────────────────────────────
        tiling_mode = getattr(scene, 'pbr_tiling', 'off')
        tile_grid = getattr(scene, 'pbr_tile_grid', 2)

        tile_model_keys = set()

        # Custom mode per-map toggles (read once, used later)
        custom_tile_albedo = False
        custom_tile_material = False
        custom_tile_normal = False
        custom_tile_height = False
        custom_tile_emission = False
        # Whether IID-Appearance is the albedo provider
        appearance_provides_albedo = (not use_delight
                                      and not use_lighting_albedo)

        if tiling_mode == 'selective':
            # When using lighting albedo, don't tile appearance
            # (its albedo will be discarded — only material channels needed).
            if not use_lighting_albedo:
                tile_model_keys.add('appearance')
            if use_lighting_albedo:
                tile_model_keys.add('lighting')
        elif tiling_mode == 'all':
            tile_model_keys = {'appearance', 'normals', 'depth', 'lighting'}
        elif tiling_mode == 'custom':
            custom_tile_albedo = getattr(scene, 'pbr_tile_albedo', True)
            custom_tile_material = getattr(scene, 'pbr_tile_material', False)
            custom_tile_normal = getattr(scene, 'pbr_tile_normal', False)
            custom_tile_height = getattr(scene, 'pbr_tile_height', False)
            custom_tile_emission = (
                getattr(scene, 'pbr_tile_emission', False)
                and want_emission
                and emission_method == 'residual'
            )

            # IID-Appearance: tile if ANY of its outputs are tiled
            need_tile_appearance = (
                custom_tile_material
                or (custom_tile_albedo and appearance_provides_albedo)
            )
            if need_tile_appearance:
                tile_model_keys.add('appearance')

            # IID-Lighting: tile if albedo or residual/emission output
            # needs tiling
            need_tile_lighting = (
                (use_lighting_albedo and custom_tile_albedo)
                or custom_tile_emission
            )
            if need_tile_lighting:
                tile_model_keys.add('lighting')

            # Single-output models
            if custom_tile_normal:
                tile_model_keys.add('normals')
            if custom_tile_height:
                tile_model_keys.add('depth')

        model_step = 0

        # Helper: map model short names to the final PBR map names they
        # produce, used for per-camera skipping in selective mode.
        _MODEL_OUTPUT_MAPS = {
            'appearance': {'albedo', 'roughness', 'metallic'},
            'normals':    {'normal'},
            'depth':      {'height'},
            'lighting':   {'emission', 'albedo'},  # residual → emission, lighting → albedo
        }

        def _camera_needs_model(cam_idx, model_key):
            """Return True if this camera still needs outputs from *model_key*."""
            if per_camera_missing is None or cam_idx not in per_camera_missing:
                return True  # full run, always needed
            return bool(per_camera_missing[cam_idx] & _MODEL_OUTPUT_MAPS.get(model_key, set()))

        # ── StableDelight pass (all cameras) ──────────────────────────
        if use_delight:
            model_step += 1
            self._pbr_step = model_step
            tile_delight = (tiling_mode in ('selective', 'all')
                            or (tiling_mode == 'custom' and custom_tile_albedo))
            tile_label = f" (tiled {tile_grid}×{tile_grid})" if tile_delight else ""
            print(f"[StableGen]   Model {model_step}/{total_model_steps}: "
                  f"StableDelight{tile_label}")

            for ci, cam_idx in enumerate(cam_ids):
                # Skip camera if albedo already exists in selective mode
                if per_camera_missing and cam_idx in per_camera_missing:
                    if 'albedo' not in per_camera_missing[cam_idx]:
                        print(f"[StableGen]     Camera {cam_idx}: albedo exists, skipping StableDelight")
                        continue
                image_path = camera_images[cam_idx]
                self._pbr_cam = ci
                self._stage = (f"PBR: StableDelight "
                               f"(cam {ci+1}/{num_cams})")
                self._progress = 0
                print(f"[StableGen]     Camera {cam_idx} ({ci+1}/{num_cams})…")

                if tile_delight:
                    result = self._process_model_tiled(
                        context, image_path,
                        process_fn=self.workflow_manager.generate_delight_map)
                    if isinstance(result, list) and result:
                        result = result[0]
                else:
                    result = self.workflow_manager.generate_delight_map(
                        context, image_path
                    )

                if isinstance(result, dict) and "error" in result:
                    print(f"[StableGen]     StableDelight error: {result['error']}")
                    self._error = f"StableDelight failed: {result['error']}"
                    return
                elif isinstance(result, bytes):
                    pbr_path = get_file_path(
                        context, "pbr", subtype="albedo",
                        camera_id=cam_idx,
                        material_id=self._material_id)
                    try:
                        self._save_raw_copy(context, "albedo", cam_idx, result)
                        with open(pbr_path, 'wb') as f:
                            f.write(result)
                        self._pbr_maps[cam_idx]['albedo'] = pbr_path
                        print(f"[StableGen]     Saved albedo (delight) → "
                              f"{os.path.basename(pbr_path)}")
                    except Exception as err:
                        print(f"[StableGen]     Failed to save delight albedo: {err}")

        # ── Marigold model passes (all cameras per model) ─────────────
        for model_name, map_names in models_to_run:
            model_step += 1
            self._pbr_step = model_step
            model_short = model_name.split('/')[-1]
            should_tile = any(k in model_short for k in tile_model_keys)
            is_iid = 'appearance' in model_short
            is_iid_lighting = 'lighting' in model_short

            tile_label = f" (tiled {tile_grid}×{tile_grid})" if should_tile else ""
            print(f"[StableGen]   Model {model_step}/{total_model_steps}: "
                  f"{model_name}{tile_label}")

            for ci, cam_idx in enumerate(cam_ids):
                # Selective mode: skip camera if it doesn't need this model
                model_key = ('appearance' if is_iid
                             else 'lighting' if is_iid_lighting
                             else 'normals' if 'normals' in model_short
                             else 'depth')
                if not _camera_needs_model(cam_idx, model_key):
                    print(f"[StableGen]     Camera {cam_idx}: already has outputs for "
                          f"{model_key}, skipping")
                    continue
                image_path = camera_images[cam_idx]
                self._pbr_cam = ci
                self._stage = (f"PBR: {model_short} "
                               f"(cam {ci+1}/{num_cams})")
                self._progress = 0

                # ── Dual-run logic ─────────────────────────────────
                # Dual-run = run tiled + untiled, then stitch outputs.
                # Needed when some outputs of a multi-output model
                # should be tiled and others should not.
                dual_run_iid = False
                dual_run_iid_reverse = False   # tile material, untile albedo
                dual_run_lighting = False

                if tiling_mode == 'selective':
                    dual_run_iid = (
                        should_tile and is_iid and not use_delight
                    )
                    dual_run_lighting = (
                        should_tile and is_iid_lighting
                        and use_lighting_albedo
                    )
                elif tiling_mode == 'custom' and should_tile:
                    if is_iid:
                        # IID-Appearance: [albedo, material]
                        want_tiled_albedo = (custom_tile_albedo
                                             and appearance_provides_albedo)
                        want_tiled_material = custom_tile_material
                        # Dual-run only when there's a mismatch
                        if want_tiled_albedo and not want_tiled_material:
                            dual_run_iid = True
                        elif want_tiled_material and not want_tiled_albedo:
                            dual_run_iid = True
                            dual_run_iid_reverse = True
                        # If both tiled → full tile (no dual-run)
                        # If neither → shouldn't reach here (should_tile=False)
                    elif is_iid_lighting:
                        # IID-Lighting: [albedo, shading, residual]
                        # Determine per-output tiling needs
                        want_tiled_l_albedo = (use_lighting_albedo
                                               and custom_tile_albedo)
                        want_tiled_l_residual = custom_tile_emission
                        # Shading is always untiled (no benefit).
                        # Dual-run when at least one output is tiled
                        # but not all (shading is never tiled, so dual-run
                        # is needed whenever ANY lighting output is tiled).
                        any_tiled = want_tiled_l_albedo or want_tiled_l_residual
                        if any_tiled:
                            dual_run_lighting = True

                extra = ""
                if dual_run_iid and not dual_run_iid_reverse:
                    extra = " + untiled material"
                elif dual_run_iid and dual_run_iid_reverse:
                    extra = " + untiled albedo"
                elif dual_run_lighting:
                    extra = " + untiled shading"
                print(f"[StableGen]     Camera {cam_idx} ({ci+1}/{num_cams}){extra}…")

                if dual_run_iid:
                    result_tiled = self._process_model_tiled(
                        context, image_path, model_name)
                    result_untiled = self.workflow_manager.generate_pbr_maps(
                        context, image_path, model_name=model_name)
                    if (isinstance(result_tiled, list)
                            and len(result_tiled) >= 2
                            and isinstance(result_untiled, list)
                            and len(result_untiled) >= 2):
                        if dual_run_iid_reverse:
                            # Tile material, keep albedo untiled
                            result = [result_untiled[0], result_tiled[1]]
                        else:
                            # Tile albedo, keep material untiled
                            result = [result_tiled[0], result_untiled[1]]
                    elif isinstance(result_tiled, dict):
                        result = result_tiled
                    elif isinstance(result_untiled, dict):
                        result = result_untiled
                    else:
                        result = (result_tiled
                                  if isinstance(result_tiled, list)
                                  else result_untiled)
                elif dual_run_lighting:
                    # IID-Lighting: stitch per-output from tiled/untiled
                    result_tiled = self._process_model_tiled(
                        context, image_path, model_name)
                    result_untiled = self.workflow_manager.generate_pbr_maps(
                        context, image_path, model_name=model_name)
                    if (isinstance(result_tiled, list)
                            and len(result_tiled) >= 3
                            and isinstance(result_untiled, list)
                            and len(result_untiled) >= 3):
                        # Per-output: pick tiled or untiled based on toggles
                        if tiling_mode == 'custom':
                            tile_l_albedo = (use_lighting_albedo
                                             and custom_tile_albedo)
                            tile_l_residual = custom_tile_emission
                        else:
                            # selective: tile albedo only
                            tile_l_albedo = True
                            tile_l_residual = False
                        r_albedo = (result_tiled[0] if tile_l_albedo
                                    else result_untiled[0])
                        r_shading = result_untiled[1]   # always untiled
                        r_residual = (result_tiled[2] if tile_l_residual
                                      else result_untiled[2])
                        result = [r_albedo, r_shading, r_residual]
                    elif isinstance(result_tiled, dict):
                        result = result_tiled
                    elif isinstance(result_untiled, dict):
                        result = result_untiled
                    else:
                        result = (result_tiled
                                  if isinstance(result_tiled, list)
                                  else result_untiled)
                elif should_tile:
                    # Tile the model even when StableDelight handles albedo —
                    # the IID-Appearance albedo output is discarded by
                    # _save_pbr_map_outputs, but roughness/metallic still
                    # benefit from tiling.
                    result = self._process_model_tiled(
                        context, image_path, model_name)
                else:
                    result = self.workflow_manager.generate_pbr_maps(
                        context, image_path, model_name=model_name)

                if isinstance(result, dict) and "error" in result:
                    print(f"[StableGen]     PBR model error: {result['error']}")
                    self._error = (f"PBR decomposition failed: "
                                     f"{result['error']}")
                    return

                # ── Save each output component ────────────────────
                self._save_pbr_map_outputs(
                    context, result, map_names, cam_idx,
                    want_roughness=want_roughness,
                    want_metallic=want_metallic,
                    use_delight=use_delight,
                    use_lighting_albedo=use_lighting_albedo)

        # ── Post-Marigold emission passes ─────────────────────────────
        if want_emission:
            if emission_method == 'residual':
                # IID-Lighting residual was already saved by the model
                # loop above – now gate it with roughness/metallic.
                model_step += 1
                self._pbr_step = model_step
                self._progress = 0
                print(f"[StableGen]   Post-processing emission (residual "
                      f"gating) – step {model_step}/{total_model_steps}")
                for ci, cam_idx in enumerate(cam_ids):
                    if per_camera_missing and cam_idx in per_camera_missing:
                        if 'emission' not in per_camera_missing[cam_idx]:
                            continue
                    self._pbr_cam = ci
                    self._stage = (f"PBR: Emission gating "
                                   f"(cam {ci+1}/{num_cams})")
                    self._progress = (ci / num_cams) * 100
                    self._gate_emission_residual(context, cam_idx)
                self._progress = 100

            elif emission_method == 'hsv':
                model_step += 1
                self._pbr_step = model_step
                self._progress = 0
                print(f"[StableGen]   Emission via HSV threshold "
                      f"– step {model_step}/{total_model_steps}")
                for ci, cam_idx in enumerate(cam_ids):
                    if per_camera_missing and cam_idx in per_camera_missing:
                        if 'emission' not in per_camera_missing[cam_idx]:
                            continue
                    self._pbr_cam = ci
                    self._stage = (f"PBR: HSV emission "
                                   f"(cam {ci+1}/{num_cams})")
                    self._progress = (ci / num_cams) * 100
                    image_path = camera_images[cam_idx]
                    self._generate_emission_hsv(context, cam_idx, image_path)
                self._progress = 100

        # Deactivate PBR progress tracking
        self._pbr_active = False
        self._pbr_step = 0
        self._pbr_total_steps = 0

        print(f"[StableGen] PBR decomposition complete for "
              f"{num_cams} camera(s)")

    def _save_pbr_map_outputs(self, context, result, map_names, camera_id,
                               want_roughness=True, want_metallic=True,
                               use_delight=False, use_lighting_albedo=False):
        """Save the output images from a single model run to disk.

        Handles the IID material channel split (R=roughness, G=metallic)
        and the camera→world normal conversion.

        Args:
            result: list of bytes (one per output map).
            map_names: list of names corresponding to each output.
            camera_id: Camera index.
            want_roughness / want_metallic: Whether to save those channels.
            use_delight: Whether StableDelight handles albedo (skip IID albedo).
        """
        scene = context.scene
        for i, map_bytes in enumerate(result):
            map_name = map_names[i] if i < len(map_names) else f"component_{i}"

            # ── IID-Appearance "material" channel split ───────────
            if map_name == 'material':
                try:
                    mat_img = Image.open(io.BytesIO(map_bytes)).convert('RGB')
                except Exception as err:
                    print(f"[StableGen]     Failed to decode material image: {err}")
                    continue

                if want_roughness:
                    rough_img = mat_img.getchannel('R').convert('L')
                    rough_path = get_file_path(
                        context, "pbr", subtype="roughness",
                        camera_id=camera_id,
                        material_id=self._material_id)
                    try:
                        rough_img.save(rough_path)
                        self._pbr_maps[camera_id]['roughness'] = rough_path
                        print(f"[StableGen]     Saved roughness (material R) → "
                              f"{os.path.basename(rough_path)}")
                    except Exception as err:
                        print(f"[StableGen]     Failed to save roughness: {err}")

                if want_metallic:
                    metal_img = mat_img.getchannel('G').convert('L')
                    metal_path = get_file_path(
                        context, "pbr", subtype="metallic",
                        camera_id=camera_id,
                        material_id=self._material_id)
                    try:
                        metal_img.save(metal_path)
                        self._pbr_maps[camera_id]['metallic'] = metal_path
                        print(f"[StableGen]     Saved metallic (material G) → "
                              f"{os.path.basename(metal_path)}")
                    except Exception as err:
                        print(f"[StableGen]     Failed to save metallic: {err}")
                continue

            # ── IID-Lighting intermediate outputs ─────────────────
            # These are prefixed with '_lighting_' to avoid toggle
            # checks.  The residual is saved for later gating; albedo
            # and shading are discarded unless using lighting albedo.
            if map_name.startswith('_lighting_'):
                lighting_key = map_name  # e.g. '_lighting_residual'
                lighting_path = get_file_path(
                    context, "pbr", subtype=lighting_key.lstrip('_'),
                    camera_id=camera_id,
                    material_id=self._material_id)
                try:
                    with open(lighting_path, 'wb') as f:
                        f.write(map_bytes)
                    self._pbr_maps[camera_id][lighting_key] = lighting_path
                    print(f"[StableGen]     Saved {lighting_key} → "
                          f"{os.path.basename(lighting_path)}")

                    # When using IID-Lighting as albedo source, copy the
                    # lighting albedo to the main albedo slot as well.
                    if map_name == '_lighting_albedo' and use_lighting_albedo:
                        albedo_path = get_file_path(
                            context, "pbr", subtype="albedo",
                            camera_id=camera_id,
                            material_id=self._material_id)
                        self._save_raw_copy(context, "albedo", camera_id, map_bytes)
                        with open(albedo_path, 'wb') as f:
                            f.write(map_bytes)
                        self._pbr_maps[camera_id]['albedo'] = albedo_path
                        print(f"[StableGen]     Saved albedo (IID-Lighting) → "
                              f"{os.path.basename(albedo_path)}")
                except Exception as err:
                    print(f"[StableGen]     Failed to save {lighting_key}: {err}")
                continue

            # Skip maps the user didn't enable
            toggle_attr = f"pbr_map_{map_name}"
            if hasattr(scene, toggle_attr) and not getattr(scene, toggle_attr):
                continue

            # Skip Marigold IID-Appearance albedo when another source handles it
            if map_name == 'albedo' and (use_delight or use_lighting_albedo):
                continue

            pbr_path = get_file_path(
                context, "pbr", subtype=map_name,
                camera_id=camera_id,
                material_id=self._material_id)
            try:
                # ── Camera→world-space conversion for normals ─────
                if map_name == 'normal':
                    converted = self._convert_normals_cam_to_world(
                        map_bytes, camera_id)
                    if converted is not None:
                        map_bytes = converted

                # ── Raw save for albedo (post-processing applied in batch) ─
                if map_name == 'albedo':
                    self._save_raw_copy(context, "albedo", camera_id, map_bytes)

                with open(pbr_path, 'wb') as f:
                    f.write(map_bytes)
                self._pbr_maps[camera_id][map_name] = pbr_path
                print(f"[StableGen]     Saved {map_name} → "
                      f"{os.path.basename(pbr_path)}")
            except Exception as err:
                print(f"[StableGen]     Failed to save PBR map {map_name}: {err}")

    # ── Emission extraction methods ───────────────────────────────────

    def _gate_emission_residual(self, context, camera_id):
        """Gate the IID-Lighting residual with roughness/metallic to
        produce a cleaner emission map.

        emission = residual × (1 − metallic) × roughness_mask, then
        threshold.  High-metallic + low-roughness regions are likely
        specular reflections, not true emission.
        """
        scene = context.scene
        cam_maps = self._pbr_maps.get(camera_id, {})
        residual_path = cam_maps.get('_lighting_residual')
        if not residual_path or not os.path.exists(residual_path):
            print(f"[StableGen]     Emission residual not found for camera {camera_id}")
            return

        threshold = getattr(scene, 'pbr_emission_threshold', 0.2)

        try:
            residual = np.array(Image.open(residual_path).convert('RGB')).astype(np.float32) / 255.0

            # Optionally gate with metallic if available — high metallic
            # surfaces produce strong specular that isn't true emission.
            metal_path = cam_maps.get('metallic')
            if metal_path and os.path.exists(metal_path):
                metal_img = Image.open(metal_path).convert('L')
                # Resize metallic to match residual dims (they may differ
                # when tiling/super-res settings aren't identical).
                res_h, res_w = residual.shape[:2]
                if metal_img.size != (res_w, res_h):
                    metal_img = metal_img.resize(
                        (res_w, res_h), Image.Resampling.BILINEAR)
                metallic = np.array(metal_img).astype(np.float32) / 255.0
                gate = (1.0 - metallic)[:, :, np.newaxis]
                residual = residual * gate

            # Apply soft threshold — pixels below the threshold are
            # smoothly faded out rather than hard-clipped.
            luminance = np.mean(residual, axis=2)
            # Smooth fade: rescale [0, threshold] → 0, [threshold, 1] → preserved
            fade = np.clip((luminance - threshold * 0.5) / max(threshold * 0.5, 1e-6),
                           0.0, 1.0)[:, :, np.newaxis]
            emission = residual * fade

            emission_img = Image.fromarray(
                np.clip(emission * 255, 0, 255).astype(np.uint8), mode='RGB')
            emission_path = get_file_path(
                context, "pbr", subtype="emission",
                camera_id=camera_id,
                material_id=self._material_id)
            emission_img.save(emission_path)
            self._pbr_maps[camera_id]['emission'] = emission_path
            print(f"[StableGen]     Saved emission (residual gated) → "
                  f"{os.path.basename(emission_path)}")
        except Exception as err:
            print(f"[StableGen]     Failed to generate emission (residual): {err}")
            import traceback; traceback.print_exc()

    def _generate_emission_hsv(self, context, camera_id, image_path):
        """Extract emission via high-saturation + high-value thresholding
        in HSV space.  Glowing objects retain colour intensity while
        normally-lit surfaces desaturate under bright light.
        """
        scene = context.scene
        sat_min = getattr(scene, 'pbr_emission_saturation_min', 0.5)
        val_min = getattr(scene, 'pbr_emission_value_min', 0.85)
        bloom_radius = getattr(scene, 'pbr_emission_bloom', 5.0)
        threshold = getattr(scene, 'pbr_emission_threshold', 0.3)

        try:
            original = np.array(Image.open(image_path).convert('RGB'))
            rgb_f = original.astype(np.float32) / 255.0

            # RGB → HSV
            maxc = rgb_f.max(axis=2)
            minc = rgb_f.min(axis=2)
            delta = maxc - minc
            saturation = np.where(maxc > 1e-6, delta / maxc, 0.0)
            value = maxc

            # Build binary mask: high saturation AND high value
            mask = ((saturation >= sat_min) & (value >= val_min)).astype(np.float32)

            # Apply bloom (Gaussian blur) to soften edges
            if bloom_radius > 0:
                try:
                    import cv2
                    ksize = int(bloom_radius * 4) | 1  # must be odd
                    mask = cv2.GaussianBlur(mask, (ksize, ksize), bloom_radius)
                except ImportError:
                    # Fallback: simple box blur via PIL
                    from PIL import ImageFilter
                    mask_pil = Image.fromarray(
                        (mask * 255).astype(np.uint8), mode='L')
                    mask_pil = mask_pil.filter(
                        ImageFilter.GaussianBlur(radius=bloom_radius))
                    mask = np.array(mask_pil).astype(np.float32) / 255.0

            # Apply threshold on the blurred mask
            mask = np.where(mask > threshold, mask, 0.0)

            # Multiply mask by original RGB to get coloured emission
            emission = rgb_f * mask[:, :, np.newaxis]

            emission_img = Image.fromarray(
                np.clip(emission * 255, 0, 255).astype(np.uint8), mode='RGB')
            emission_path = get_file_path(
                context, "pbr", subtype="emission",
                camera_id=camera_id,
                material_id=self._material_id)
            emission_img.save(emission_path)
            self._pbr_maps[camera_id]['emission'] = emission_path
            print(f"[StableGen]     Saved emission (HSV threshold) → "
                  f"{os.path.basename(emission_path)}")
        except Exception as err:
            print(f"[StableGen]     Failed to generate emission (HSV): {err}")
            import traceback; traceback.print_exc()
