"""Export operators for 3D printing (STL, 3MF with multi-color support).

The 3MF exporter uses the baked texture to produce per-triangle material
assignments for multi-color / multi-filament printing (OrcaSlicer-FullSpectrum
and similar slicers).  Colors are quantised via k-means clustering so that
each face maps to one of *N* filament colours.
"""

import colorsys
import math
import os
import random
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

import bpy  # pylint: disable=import-error
from ..utils import get_dir_path, sg_modal_active
from ..core import ADDON_PKG, get_addon_prefs

NS_3MF = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_MAT = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"


# ── Palette Definition & Operators ───────────────────────────────────────

class StableGenPaletteColor(bpy.types.PropertyGroup):
    def _update_color(self, context):
        from ..core.callbacks import update_palette_color
        update_palette_color(self, context)

    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(0.5, 0.5, 0.5),
        description="Filament Color",
        update=_update_color
    )
    name: bpy.props.StringProperty(
        name="Name",
        default="Filament"
    )


def prepopulate_palette(scene):
    if not hasattr(scene, "stablegen_print_palette") or len(scene.stablegen_print_palette) > 0:
        return
    
    cyan = (0.0, 1.0, 1.0)
    magenta = (1.0, 0.0, 1.0)
    yellow = (1.0, 1.0, 0.0)
    white = (1.0, 1.0, 1.0)
    
    colors_info = [
        ("Cyan", cyan),
        ("Magenta", magenta),
        ("Yellow", yellow),
        ("White", white),
    ]
    
    for name, col in colors_info:
        item = scene.stablegen_print_palette.add()
        item.name = name
        item.color = col


def _find_closest_palette_color(color, palette_rgbs):
    """Find the index of the closest color in the palette (Euclidean distance)."""
    best_dist = float('inf')
    best_idx = 0
    for idx, p_color in enumerate(palette_rgbs):
        dist = sum((color[j] - p_color[j]) ** 2 for j in range(3))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


class StableGenAddPaletteColor(bpy.types.Operator):
    """Add a new filament color to the print palette."""
    bl_idname = "stablegen.add_palette_color"
    bl_label = "Add Color"
    bl_options = {'UNDO'}

    def execute(self, context):
        scene = context.scene
        item = scene.stablegen_print_palette.add()
        item.color = (random.random(), random.random(), random.random())
        item.name = f"Filament {len(scene.stablegen_print_palette)}"
        scene.stablegen_print_palette_index = len(scene.stablegen_print_palette) - 1
        scene.stablegen_print_preset = 'CUSTOM'
        return {'FINISHED'}


class StableGenRemovePaletteColor(bpy.types.Operator):
    """Remove the selected filament color from the print palette."""
    bl_idname = "stablegen.remove_palette_color"
    bl_label = "Remove Color"
    bl_options = {'UNDO'}

    def execute(self, context):
        scene = context.scene
        idx = scene.stablegen_print_palette_index
        if 0 <= idx < len(scene.stablegen_print_palette):
            scene.stablegen_print_palette.remove(idx)
            scene.stablegen_print_palette_index = max(0, idx - 1)
            scene.stablegen_print_preset = 'CUSTOM'
        return {'FINISHED'}


class StableGenPreviewQuantization(bpy.types.Operator):
    """Preview the quantized palette colors on the active mesh object."""
    bl_idname = "object.stablegen_preview_quantization"
    bl_label = "Preview Palette Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        
        if not scene.stablegen_print_palette:
            prepopulate_palette(scene)
            
        import numpy as np
        palette_rgbs_linear = np.array([item.color for item in scene.stablegen_print_palette], dtype=np.float32)
        palette_rgbs = [tuple(c) for c in linear_to_srgb_numpy(palette_rgbs_linear)]
        if not palette_rgbs:
            self.report({'WARNING'}, "No colors defined in the print palette.")
            return {'CANCELLED'}
            
        image = _find_texture_image(obj)
        
        # Get triangulated mesh temporarily to read UVs and map colors
        temp, eval_obj = _get_triangulated_mesh(obj, apply_transforms=False)
        try:
            import numpy as np
            
            # Collect face colors
            if image:
                face_colors = _collect_face_colors(temp, image)[:, :3]
            else:
                # If no image, check for a color layer other than SG_PrintPreview first
                color_layer = None
                for attr in temp.color_attributes:
                    if attr.name != "SG_PrintPreview":
                        color_layer = attr
                        break
                if not color_layer:
                    color_layer = temp.color_attributes.get("SG_PrintPreview")
                
                if color_layer:
                    n_loops = len(temp.loops)
                    loop_colors = np.empty(n_loops * 4, dtype=np.float32)
                    color_layer.data.foreach_get("color", loop_colors)
                    loop_colors = loop_colors.reshape((-1, 4))
                    
                    n_tris = len(temp.loop_triangles)
                    tri_loops = np.empty(n_tris * 3, dtype=np.int32)
                    temp.loop_triangles.foreach_get("loops", tri_loops)
                    tri_loops = tri_loops.reshape((-1, 3))
                    
                    face_colors = loop_colors[tri_loops].mean(axis=1)[:, :3]
                    if color_layer.data_type == 'BYTE_COLOR':
                        face_colors = srgb_to_linear_numpy(face_colors)
                else:
                    face_colors = np.ones((len(temp.loop_triangles), 3), dtype=np.float32)
            
            face_colors_srgb = face_colors # texture already in srgb
            
            if getattr(scene, "stablegen_print_dithered", True):
                # Dithered Mode: compute physical mixed colors
                mode, channel_indices = classify_filaments(palette_rgbs)
                if mode == 'solid':
                    # Fallback: Frank-Wolfe general solver on all palette colors
                    fil_rgbs = np.array(palette_rgbs, dtype=np.float32)
                    demands = solve_color_mixing_numpy(fil_rgbs, face_colors_srgb)
                    
                    # Reconstruct mixed color using the chosen mixing model
                    model_formula = getattr(scene, "stablegen_print_solver_formula", 'KUBELKA_MUNK')
                    if model_formula == 'KUBELKA_MUNK':
                        scat_weight = getattr(scene, "stablegen_print_scattering_weight", 10.0)
                        fil_val = fil_rgbs.max(axis=1)
                        fil_min = fil_rgbs.min(axis=1)
                        fil_sat = np.zeros_like(fil_val)
                        mask = fil_val > 1e-5
                        fil_sat[mask] = (fil_val[mask] - fil_min[mask]) / fil_val[mask]
                        S_fil = 1.0 + scat_weight * (1.0 - fil_sat) * fil_val
                        S_fil = S_fil[:, None]
                        R_clip = np.clip(fil_rgbs, 0.001, 0.999)
                        K_fil = S_fil * ((1.0 - R_clip)**2) / (2.0 * R_clip)
                        
                        K_mix = np.dot(demands, K_fil)
                        S_mix = np.dot(demands, S_fil)
                        ratio = K_mix / S_mix
                        mixed_srgb = 1.0 + ratio - np.sqrt(ratio**2 + 2.0 * ratio)
                    elif model_formula == 'ADDITIVE':
                        mixed_srgb = np.dot(demands, fil_rgbs)
                    else:  # SUBTRACTIVE
                        mixed_srgb = 1.0 - np.dot(demands, 1.0 - fil_rgbs)
                else:
                    # standard dithered subtractive formula
                    mixed_srgb = np.empty_like(face_colors_srgb)
                    fil_rgbs = np.array([palette_rgbs[idx] for idx in channel_indices], dtype=np.float32)
                    for i in range(len(face_colors_srgb)):
                        r, g, b = face_colors_srgb[i] * 255.0
                        fractions = decompose_subtractive(r, g, b, mode, palette_rgbs, channel_indices)
                        mixed_srgb[i] = np.clip(1.0 - sum(fractions[j] * (1.0 - fil_rgbs[j]) for j in range(len(fractions))), 0.0, 1.0)
                
                # Convert the sRGB mixed colors back to linear for Blender color attribute
                mapped_colors = srgb_to_linear_numpy(mixed_srgb)
            else:
                # Solid Mode: match to the closest physical filament in the palette
                palette_arr = np.array(palette_rgbs, dtype=np.float32)  # (K, 3)
                
                # Vectorized closest color matching
                best_idx = _match_colors_hsv(
                    face_colors_srgb, palette_rgbs,
                    chroma_threshold=getattr(scene, "stablegen_print_chroma_threshold", 0.35)
                )
                
                # Apply majority-vote smoothing (island cleanup)
                smoothing_passes = getattr(scene, "stablegen_print_smoothing", 2)
                if smoothing_passes > 0:
                    n_tris = len(temp.loop_triangles)
                    tri_verts = np.empty(n_tris * 3, dtype=np.int32)
                    temp.loop_triangles.foreach_get("vertices", tri_verts)
                    tri_verts = tri_verts.reshape((-1, 3))
                    
                    best_idx_smoothed = _smooth_colors_vectorized(
                        tri_verts, best_idx + 1, len(temp.vertices), len(palette_rgbs), iterations=smoothing_passes
                    )
                    best_idx = best_idx_smoothed - 1
                    
                mapped_srgb = palette_arr[best_idx]  # (N, 3)
                mapped_colors = srgb_to_linear_numpy(mapped_srgb)
                
        finally:
            eval_obj.to_mesh_clear()
            
        # Debug logging to inspect mapped color ranges
        if len(mapped_colors) > 0:
            print(f"[StableGen Preview Debug] mode={mode if getattr(scene, 'stablegen_print_dithered', True) else 'solid'}")
            print(f"  palette_rgbs={palette_rgbs}")
            print(f"  mapped_colors min={mapped_colors.min(axis=0)}, max={mapped_colors.max(axis=0)}, mean={mapped_colors.mean(axis=0)}")

        # Apply the mapped colors to a Color Attribute layer on the actual mesh data (obj.data)
        mesh = obj.data
        color_layer = mesh.color_attributes.get("SG_PrintPreview")
        if color_layer and color_layer.data_type != 'FLOAT_COLOR':
            mesh.color_attributes.remove(color_layer)
            color_layer = None
            
        if not color_layer:
            color_layer = mesh.color_attributes.new(
                name="SG_PrintPreview",
                type='FLOAT_COLOR',
                domain='CORNER',
            )
            
        mesh.calc_loop_triangles()
        if len(mesh.loop_triangles) != len(mapped_colors):
            self.report({'ERROR'}, "Mesh triangle count mismatch. Try applying modifiers first.")
            return {'CANCELLED'}
            
        # Assign to loop corners vectorially
        n_loops = len(mesh.loops)
        loop_colors = np.ones((n_loops, 4), dtype=np.float32)  # RGBA
        
        tri_loops = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops)
        tri_loops = tri_loops.reshape((-1, 3))
        
        mapped_colors_rgba = np.ones((len(mapped_colors), 4), dtype=np.float32)
        mapped_colors_rgba[:, :3] = mapped_colors
        
        loop_colors[tri_loops] = mapped_colors_rgba[:, None, :]
        
        color_layer.data.foreach_set("color", loop_colors.ravel())
                
        # Switch viewport shading color mode to show the Color Attribute
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        try:
                            # Switch shading type to SOLID to display color attributes
                            space.shading.type = 'SOLID'
                            space.shading.color_type = 'VERTEX'
                            try:
                                space.shading.light = 'FLAT'
                            except Exception:
                                pass
                            if hasattr(mesh, "color_attributes"):
                                mesh.color_attributes.active = color_layer
                            if hasattr(mesh, "attributes"):
                                mesh.attributes.active_color = color_layer
                        except TypeError:
                            # Fallback to Blender 4.x: set to ATTRIBUTE and set attribute_name
                            try:
                                space.shading.type = 'SOLID'
                                space.shading.color_type = 'ATTRIBUTE'
                                space.shading.attribute_name = "SG_PrintPreview"
                                try:
                                    space.shading.light = 'FLAT'
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        except Exception:
                            pass
                        
        self.report({'INFO'}, f"Color preview applied.")
        return {'FINISHED'}
                
                
class StableGenClearQuantizationPreview(bpy.types.Operator):
    """Clear the quantized color preview and restore material display."""
    bl_idname = "object.stablegen_clear_quantization_preview"
    bl_label = "Clear Preview"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        preview_obj = bpy.data.objects.get("SG_SlicedPreview")
        if preview_obj:
            orig_name = preview_obj.get("original_name")
            orig_obj = None
            if orig_name:
                orig_obj = bpy.data.objects.get(orig_name)
            
            preview_mesh = preview_obj.data
            bpy.data.objects.remove(preview_obj, do_unlink=True)
            if preview_mesh:
                bpy.data.meshes.remove(preview_mesh, do_unlink=True)
                
            if orig_obj:
                orig_obj.hide_viewport = False
                orig_obj.hide_set(False)
                for select_obj in context.selected_objects:
                    select_obj.select_set(False)
                orig_obj.select_set(True)
                context.view_layer.objects.active = orig_obj
                
                color_layer = orig_obj.data.color_attributes.get("SG_PrintPreview")
                if color_layer:
                    orig_obj.data.color_attributes.remove(color_layer)
        else:
            obj = context.active_object
            if obj and obj.type == 'MESH':
                color_layer = obj.data.color_attributes.get("SG_PrintPreview")
                if color_layer:
                    obj.data.color_attributes.remove(color_layer)
            
        # Restore viewport shading to material
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        shading = space.shading
                        is_our_shading = False
                        
                        # Blender 4.x check
                        if getattr(shading, "color_type", "") == 'ATTRIBUTE':
                            if getattr(shading, "attribute_name", "") == "SG_PrintPreview":
                                is_our_shading = True
                        # Blender 5.x check
                        elif getattr(shading, "color_type", "") == 'VERTEX':
                            is_our_shading = True
                            
                        if is_our_shading:
                            try:
                                shading.color_type = 'MATERIAL'
                                shading.type = 'MATERIAL'  # Switch back to Material Preview mode
                                try:
                                    shading.light = 'STUDIO'
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            
def _generate_blue_noise_tile(size=64, seed=42):
    """Generate a toroidal blue noise threshold tile via FFT spectral shaping.

    Produces a seamlessly-tiling ``size``×``size`` float32 matrix with values
    uniformly distributed in [0, 1) and blue-noise spatial structure (energy
    concentrated in high frequencies, no clumping).  Used for ordered spatial
    dithering to break up visible banding on flat surfaces.
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    noise = rng.rand(size, size).astype(np.float64)

    F = np.fft.fft2(noise)
    F_shifted = np.fft.fftshift(F)

    cy, cx = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist_norm = dist / (size * 0.5)
    hp_filter = dist_norm ** 1.5

    F_filtered = F_shifted * hp_filter
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

    flat = result.ravel()
    order = flat.argsort()
    ranks = np.empty(size * size, dtype=np.float64)
    ranks[order] = np.arange(size * size, dtype=np.float64) / (size * size)

    return ranks.reshape(size, size).astype(np.float32)


def _subdivide_triangle_barycentric(tri, n_div):
    """Subdivide a triangle into ``n_div``² sub-triangles via a barycentric grid.

    ``tri`` is a list of 3 vertices, each a list of 3 float coords.  Returns a
    list of sub-triangles in the same format.  When ``n_div <= 1`` the original
    triangle is returned unchanged.
    """
    if n_div <= 1:
        return [tri]
    v0, v1, v2 = tri[0], tri[1], tri[2]
    sub = []
    inv = 1.0 / n_div
    for i in range(n_div):
        for j in range(n_div - i):
            k = n_div - i - j
            p00 = [(i * v0[c] + j * v1[c] + k * v2[c]) * inv for c in range(3)]
            p10 = [((i + 1) * v0[c] + j * v1[c] + (k - 1) * v2[c]) * inv for c in range(3)]
            p01 = [(i * v0[c] + (j + 1) * v1[c] + (k - 1) * v2[c]) * inv for c in range(3)]
            sub.append([p00, p10, p01])
            if k > 1:
                p11 = [((i + 1) * v0[c] + (j + 1) * v1[c] + (k - 2) * v2[c]) * inv for c in range(3)]
                sub.append([p10, p11, p01])
    return sub


def _bn_pick_filament(t_val, cum, n_channels, li_fallback):
    """Select filament index from a blue noise threshold via the demand CDF."""
    if cum is None:
        return li_fallback % n_channels
    fi = 0
    for ch in range(n_channels - 1):
        if t_val >= cum[ch]:
            fi = ch + 1
        else:
            break
    return fi


def _bn_assign_tri(tri, ox, oy, bn_tile, bn_size, bn_scale, cum, n_channels, li_fallback,
                   max_div=16, target_area=1.0):
    """Assign filament(s) to a triangle via blue noise dithering.

    Large triangles (XY projection area > ``target_area`` mm²) are subdivided
    barycentrically so that each sub-triangle receives an independent blue-noise
    sample.  Returns a list of ``(sub_tri, filament_idx)`` tuples.
    """
    v0, v1, v2 = tri[0], tri[1], tri[2]
    area_2d = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])) * 0.5

    if area_2d > target_area:
        n_div = min(int(math.sqrt(area_2d / target_area)) + 1, max_div)
        if n_div > 1:
            fine_tris = _subdivide_triangle_barycentric(tri, n_div)
            results = []
            for ft in fine_tris:
                cx_s = (ft[0][0] + ft[1][0] + ft[2][0]) / 3.0
                cy_s = (ft[0][1] + ft[1][1] + ft[2][1]) / 3.0
                t_val = float(bn_tile[int(cx_s * bn_scale + ox) % bn_size,
                                      int(cy_s * bn_scale + oy) % bn_size])
                fi = _bn_pick_filament(t_val, cum, n_channels, li_fallback)
                results.append((ft, fi))
            return results

    cx_s = (v0[0] + v1[0] + v2[0]) / 3.0
    cy_s = (v0[1] + v1[1] + v2[1]) / 3.0
    t_val = float(bn_tile[int(cx_s * bn_scale + ox) % bn_size,
                          int(cy_s * bn_scale + oy) % bn_size])
    fi = _bn_pick_filament(t_val, cum, n_channels, li_fallback)
    return [(tri, fi)]


def _preview_sliced_worker(all_raw_verts, all_raw_tris, face_colors_srgb, unique_keys, unique_colors_arr,
                       palette_rgbs, mode, channel_indices, h_target, LH, n_layers, cyan_idx, n_channels,
                       z_min, scale_factor, is_dithered, chroma_threshold, smoothing_passes, status,
                       solver_settings=None, dither_method='Z_SEQUENCE',
                       single_filament_internal=False, internal_visibility_threshold=0.01,
                       face_visibility=None):
    try:
        import numpy as np
        import math

        if not is_dithered:
            status["stage"] = "Matching solid colors"
            status["progress"] = 20.0
            
            best_idx = _match_colors_hsv(
                face_colors_srgb, palette_rgbs,
                chroma_threshold=chroma_threshold
            )
            
            if smoothing_passes > 0:
                status["stage"] = "Smoothing color islands"
                status["progress"] = 50.0
                best_idx_smoothed = _smooth_colors_vectorized(
                    all_raw_tris, best_idx + 1, len(all_raw_verts), len(palette_rgbs), iterations=smoothing_passes
                )
                best_idx = best_idx_smoothed - 1
            
            status["stage"] = "Finalizing mesh structure"
            status["progress"] = 90.0
            
            final_verts = all_raw_verts.tolist()
            final_tris = all_raw_tris.tolist()
            final_paint = best_idx.tolist()
        else:
            scaled_verts = all_raw_verts.copy()
            scaled_verts[:, 0] *= scale_factor
            scaled_verts[:, 1] *= scale_factor
            scaled_verts[:, 2] = (scaled_verts[:, 2] - z_min) * scale_factor
            
            status["stage"] = "Calculating mixing demands"
            status["progress"] = 10.0
            
            if mode == 'solid':
                fil_rgbs = np.array(palette_rgbs, dtype=np.float32)
                if solver_settings is None:
                    solver_settings = {}
                iters = solver_settings.get("iterations", 80)
                init_m = solver_settings.get("init_method", 'CLOSEST')
                formula = solver_settings.get("model_formula", 'KUBELKA_MUNK')
                s_weight = solver_settings.get("scat_weight", 10.0)
                p_weight = solver_settings.get("penalty_weight", 1.0)
                
                demands = solve_color_mixing_numpy(
                    1.0 - fil_rgbs, 1.0 - unique_colors_arr,
                    iterations=iters,
                    init_method=init_m,
                    model_formula=formula,
                    scat_weight=s_weight,
                    penalty_weight=p_weight
                )
            else:
                demands = np.empty((len(unique_colors_arr), n_channels), dtype=np.float32)
                for idx, color in enumerate(unique_colors_arr):
                    if status.get("cancel_requested", False):
                        return
                    r, g, b = round(color[0] * 255.0), round(color[1] * 255.0), round(color[2] * 255.0)
                    demands[idx] = decompose_subtractive(r, g, b, mode, palette_rgbs, channel_indices)
            unique_colors_set2 = [tuple(round(c * 255.0) for c in color) for color in unique_colors_arr]

            if dither_method == 'BLUE_NOISE':
                status["stage"] = "Building blue noise dither data"
                status["progress"] = 30.0

                blue_noise_tile = _generate_blue_noise_tile(128)
                bn_size = blue_noise_tile.shape[0]
                bn_scale = 4.0
                _off_rng = np.random.RandomState(98765)
                layer_ox = _off_rng.randint(0, bn_size, size=n_layers).tolist()
                layer_oy = _off_rng.randint(0, bn_size, size=n_layers).tolist()

                color_cumdemands = {}
                for idx, color_key in enumerate(unique_colors_set2):
                    if status.get("cancel_requested", False):
                        return
                    row = demands[idx]
                    total = sum(max(0.0, d) for d in row)
                    if total < 0.001:
                        color_cumdemands[color_key] = None
                    else:
                        cum = []
                        s = 0.0
                        for d in row:
                            s += max(0.0, d) / total
                            cum.append(s)
                        cum[-1] = 1.0
                        color_cumdemands[color_key] = cum
            else:
                status["stage"] = "Generating color sequences"
                status["progress"] = 30.0

                color_seqs = {}
                for idx, color_key in enumerate(unique_colors_set2):
                    if status.get("cancel_requested", False):
                        return
                    row = demands[idx]
                    total = sum(max(0.0, d) for d in row)
                    seq = [0] * n_layers

                    if total < 0.001:
                        if cyan_idx >= 0:
                            nc = min(3, n_channels - cyan_idx)
                            for li in range(n_layers):
                                seq[li] = cyan_idx + (li % nc)
                        else:
                            for li in range(n_layers):
                                seq[li] = li % n_channels
                    else:
                        frac = [max(0.0, d) / total for d in row]
                        placed = [0] * n_channels
                        for li in range(n_layers):
                            best_ch = 0
                            best_d = -float('inf')
                            for ch in range(n_channels):
                                if frac[ch] < 0.001:
                                    continue
                                d = frac[ch] * (li + 1) - placed[ch]
                                tie_break = d + ch * 1e-6 + (1e-7 if li % n_channels == ch else 0.0)
                                if tie_break > best_d:
                                    best_d = tie_break
                                    best_ch = ch
                            seq[li] = best_ch
                            placed[best_ch] += 1
                    color_seqs[color_key] = seq

            status["stage"] = "Clipping triangles crossing layers"
            status["progress"] = 50.0

            clipped_tris = []
            ZOFF = LH / 2

            total_tris_count = len(all_raw_tris)
            for fi in range(total_tris_count):
                if status.get("cancel_requested", False):
                    return
                if fi % max(1, total_tris_count // 10) == 0:
                    status["progress"] = 50.0 + (fi / total_tris_count) * 30.0

                t_idx = all_raw_tris[fi]
                v1, v2, v3 = scaled_verts[t_idx[0]], scaled_verts[t_idx[1]], scaled_verts[t_idx[2]]
                fz_min = min(v1[2], v2[2], v3[2])
                fz_max = max(v1[2], v2[2], v3[2])
                tri = [list(v1), list(v2), list(v3)]

                li_start = max(0, int(math.floor((fz_min - ZOFF) / LH)))
                li_end = min(n_layers - 1, int(math.ceil((fz_max - ZOFF) / LH)))

                color_key = unique_keys[fi]

                if single_filament_internal and face_visibility is not None and fi < len(face_visibility) and face_visibility[fi] < internal_visibility_threshold:
                    internal_fi = n_channels - 1
                    if li_start == li_end or li_start >= n_layers:
                        cz_c = (fz_min + fz_max) / 2
                        li2 = max(0, min(n_layers - 1, int(math.floor((cz_c - ZOFF) / LH))))
                        clipped_tris.append((tri, internal_fi))
                    else:
                        for li in range(li_start, li_end + 1):
                            if li >= n_layers:
                                break
                            zB = ZOFF + li * LH
                            zT = ZOFF + (li + 1) * LH
                            for st in cTZ(tri, zB, zT):
                                clipped_tris.append((st, internal_fi))
                    continue

                if dither_method == 'BLUE_NOISE':
                    cum = color_cumdemands.get(color_key)
                    if li_start == li_end or li_start >= n_layers:
                        cz = (fz_min + fz_max) / 2
                        li2 = max(0, min(n_layers - 1, int(math.floor((cz - ZOFF) / LH))))
                        for ft, fi in _bn_assign_tri(tri, layer_ox[li2], layer_oy[li2],
                                                      blue_noise_tile, bn_size, bn_scale,
                                                      cum, n_channels, li2):
                            clipped_tris.append((ft, fi))
                    else:
                        for li in range(li_start, li_end + 1):
                            if li >= n_layers:
                                break
                            zB = ZOFF + li * LH
                            zT = ZOFF + (li + 1) * LH
                            sub_tris = cTZ(tri, zB, zT)
                            ox = layer_ox[li]
                            oy = layer_oy[li]
                            for st in sub_tris:
                                for ft, fi in _bn_assign_tri(st, ox, oy,
                                                              blue_noise_tile, bn_size, bn_scale,
                                                              cum, n_channels, li):
                                    clipped_tris.append((ft, fi))
                else:
                    seq = color_seqs[color_key]
                    if li_start == li_end or li_start >= n_layers:
                        cz = (fz_min + fz_max) / 2
                        li2 = max(0, min(n_layers - 1, int(math.floor((cz - ZOFF) / LH))))
                        clipped_tris.append((tri, seq[li2]))
                    else:
                        for li in range(li_start, li_end + 1):
                            if li >= n_layers:
                                break
                            zB = ZOFF + li * LH
                            zT = ZOFF + (li + 1) * LH
                            sub_tris = cTZ(tri, zB, zT)
                            for st in sub_tris:
                                clipped_tris.append((st, seq[li]))
            
            status["stage"] = "Welding coplanar slice vertices"
            status["progress"] = 80.0
            
            weld_map = {}
            welded_verts = []
            welded_tris = []
            paint_arr = []
            
            total_clipped_tris = len(clipped_tris)
            for idx, (tri, filament_idx) in enumerate(clipped_tris):
                if status.get("cancel_requested", False):
                    return
                if idx % max(1, total_clipped_tris // 10) == 0:
                    status["progress"] = 80.0 + (idx / total_clipped_tris) * 15.0
                
                tri_idx = []
                for pt in tri:
                    key = (round(pt[0], 6), round(pt[1], 6), round(pt[2], 6))
                    if key not in weld_map:
                        weld_map[key] = len(welded_verts)
                        welded_verts.append(pt)
                    tri_idx.append(weld_map[key])
                welded_tris.append(tri_idx)
                global_filament_idx = channel_indices[filament_idx]
                paint_arr.append(global_filament_idx)
                
            status["stage"] = "Scaling geometry coordinates"
            status["progress"] = 95.0
            
            final_verts = []
            for pt in welded_verts:
                bx = pt[0] / scale_factor
                by = pt[1] / scale_factor
                bz = (pt[2] / scale_factor) + z_min
                final_verts.append((bx, by, bz))
                
            final_tris = welded_tris
            final_paint = paint_arr
            
        status["result"] = (final_verts, final_tris, final_paint)
        status["done"] = True
    except Exception as e:
        status["error"] = e
        status["done"] = True


class StableGenPreviewSliced(bpy.types.Operator):
    """Slice and dither the active mesh, and preview the actual 3D printed layers in the viewport."""
    bl_idname = "object.stablegen_preview_sliced"
    bl_label = "Preview Slices"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _thread_status = None
    _progress = 0.0
    _stage = ""
    _obj = None

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and not sg_modal_active(context)

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        self._obj = obj
        
        if not scene.stablegen_print_palette:
            prepopulate_palette(scene)
            
        import numpy as np
        palette_rgbs_linear = np.array([item.color for item in scene.stablegen_print_palette], dtype=np.float32)
        palette_rgbs = [tuple(c) for c in linear_to_srgb_numpy(palette_rgbs_linear)]
        if not palette_rgbs:
            self.report({'WARNING'}, "No colors defined in the print palette.")
            return {'CANCELLED'}
            
        import numpy as np
        import math
        import threading

        # 1. Clear any existing preview object
        existing_preview = bpy.data.objects.get("SG_SlicedPreview")
        if existing_preview:
            orig_name = existing_preview.get("original_name")
            if orig_name:
                orig_obj = bpy.data.objects.get(orig_name)
                if orig_obj:
                    orig_obj.hide_viewport = False
                    orig_obj.hide_set(False)
            preview_mesh = existing_preview.data
            bpy.data.objects.remove(existing_preview, do_unlink=True)
            if preview_mesh:
                bpy.data.meshes.remove(preview_mesh, do_unlink=True)

        # 2. Extract geometry and colors (Fast, must run on main thread)
        image = _find_texture_image(obj)
        temp, eval_obj = _get_triangulated_mesh(obj, apply_transforms=False)
        try:
            n_v = len(temp.vertices)
            v_co = np.empty(n_v * 3, dtype=np.float32)
            temp.vertices.foreach_get("co", v_co)
            all_raw_verts = v_co.reshape((-1, 3))
            
            if image:
                face_colors = _collect_face_colors(temp, image)[:, :3]
            else:
                color_layer = None
                for attr in temp.color_attributes:
                    if attr.name != "SG_PrintPreview":
                        color_layer = attr
                        break
                if not color_layer:
                    color_layer = temp.color_attributes.get("SG_PrintPreview")
                
                if color_layer:
                    n_loops = len(temp.loops)
                    loop_colors = np.empty(n_loops * 4, dtype=np.float32)
                    color_layer.data.foreach_get("color", loop_colors)
                    loop_colors = loop_colors.reshape((-1, 4))
                    
                    n_tris = len(temp.loop_triangles)
                    tri_loops = np.empty(n_tris * 3, dtype=np.int32)
                    temp.loop_triangles.foreach_get("loops", tri_loops)
                    tri_loops = tri_loops.reshape((-1, 3))
                    
                    face_colors = loop_colors[tri_loops].mean(axis=1)[:, :3]
                    if color_layer.data_type == 'BYTE_COLOR':
                        face_colors = srgb_to_linear_numpy(face_colors)
                else:
                    face_colors = np.ones((len(temp.loop_triangles), 3), dtype=np.float32)
            
            t_idx = np.empty(len(temp.loop_triangles) * 3, dtype=np.int32)
            temp.loop_triangles.foreach_get("vertices", t_idx)
            all_raw_tris = t_idx.reshape((-1, 3))

            face_visibility = _collect_face_visibility(temp)
        finally:
            eval_obj.to_mesh_clear()

        h_target = scene.stablegen_print_model_height
        LH = scene.stablegen_print_layer_height
        n_layers = max(1, int(math.ceil(h_target / LH)))
        
        face_colors_srgb = face_colors
        q = 8
        face_colors_q = np.clip(np.round(face_colors_srgb * 255.0 / q) * q, 0, 255).astype(np.int32)
        unique_keys = [tuple(c) for c in face_colors_q]
        unique_colors_set = sorted(list(set(unique_keys)))
        unique_colors_arr = np.array(unique_colors_set, dtype=np.float32) / 255.0
        
        mode, channel_indices = classify_filaments(palette_rgbs)
        n_channels = len(channel_indices)
        
        cyan_idx = -1
        if mode != 'solid':
            for ci, p_idx in enumerate(channel_indices):
                if scene.stablegen_print_palette[p_idx].name == "Cyan":
                    cyan_idx = ci
                    break

        z_min, z_max = all_raw_verts[:, 2].min(), all_raw_verts[:, 2].max()
        h_blender = z_max - z_min
        if h_blender < 1e-5:
            h_blender = 1.0
        scale_factor = h_target / h_blender

        is_dithered = getattr(scene, "stablegen_print_dithered", True)
        dither_method = getattr(scene, "stablegen_print_dither_method", "Z_SEQUENCE")
        single_filament_internal = getattr(scene, "stablegen_print_single_filament_internal", True)
        internal_visibility_threshold = getattr(scene, "stablegen_print_internal_visibility_threshold", 0.01)
        chroma_threshold = getattr(scene, "stablegen_print_chroma_threshold", 0.35)
        smoothing_passes = getattr(scene, "stablegen_print_smoothing", 2)

        # Collect advanced solver settings on the main thread (thread-safe)
        solver_settings = {
            "init_method": getattr(scene, "stablegen_print_solver_init", 'CLOSEST'),
            "iterations": getattr(scene, "stablegen_print_solver_steps", 80),
            "model_formula": getattr(scene, "stablegen_print_solver_formula", 'KUBELKA_MUNK'),
            "scat_weight": getattr(scene, "stablegen_print_scattering_weight", 10.0),
            "penalty_weight": getattr(scene, "stablegen_print_saturation_penalty", 1.0),
        }

        self._thread_status = {
            "progress": 0.0,
            "stage": "Starting worker thread",
            "done": False,
            "cancel_requested": False
        }
        self._thread = threading.Thread(
            target=_preview_sliced_worker,
            args=(
                all_raw_verts, all_raw_tris, face_colors_srgb, unique_keys, unique_colors_arr,
                palette_rgbs, mode, channel_indices, h_target, LH, n_layers, cyan_idx, n_channels,
                z_min, scale_factor, is_dithered, chroma_threshold, smoothing_passes, self._thread_status,
                solver_settings, dither_method, single_filament_internal, internal_visibility_threshold,
                face_visibility
            ),
            daemon=True
        )
        self._thread.start()

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type in ('PROPERTIES', 'VIEW_3D'):
                        area.tag_redraw()

            status = self._thread_status
            if status.get("cancel_requested", False):
                self.report({'INFO'}, "Preview slices generation cancelled.")
                self.cleanup(context)
                return {'FINISHED'}

            if status.get("done", False):
                if "error" in status:
                    err = status["error"]
                    self.report({'ERROR'}, f"Preview slices failed: {err}")
                    import traceback
                    traceback.print_exc()
                    self.cleanup(context)
                    return {'CANCELLED'}

                # Success
                final_verts, final_tris, final_paint = status["result"]
                self.create_preview_object(context, final_verts, final_tris, final_paint)
                self.cleanup(context)
                return {'FINISHED'}

            self._progress = status.get("progress", 0.0)
            self._stage = status.get("stage", "Processing")

        return {'PASS_THROUGH'}

    def cleanup(self, context):
        if hasattr(self, "_timer") and self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def create_preview_object(self, context, final_verts, final_tris, final_paint):
        obj = self._obj
        scene = context.scene
        
        import numpy as np
        palette_rgbs_linear = np.array([item.color for item in scene.stablegen_print_palette], dtype=np.float32)
        palette_rgbs = [tuple(c) for c in linear_to_srgb_numpy(palette_rgbs_linear)]

        # Create the preview object
        preview_mesh = bpy.data.meshes.new("SG_SlicedPreviewMesh")
        preview_mesh.from_pydata(final_verts, [], final_tris)
        preview_mesh.update()
        
        color_layer = preview_mesh.color_attributes.new("SG_PrintPreview", 'BYTE_COLOR', 'CORNER')
        loop_colors = np.empty(len(preview_mesh.loops) * 4, dtype=np.float32)
        
        for face_idx, face in enumerate(preview_mesh.polygons):
            filament_idx = final_paint[face_idx]
            srgb = palette_rgbs[filament_idx]
            for loop_idx in face.loop_indices:
                loop_colors[loop_idx * 4 : loop_idx * 4 + 3] = srgb
                loop_colors[loop_idx * 4 + 3] = 1.0
                
        color_layer.data.foreach_set("color", loop_colors)
        
        preview_obj = bpy.data.objects.new("SG_SlicedPreview", preview_mesh)
        preview_obj.matrix_world = obj.matrix_world
        preview_obj["original_name"] = obj.name
        
        context.collection.objects.link(preview_obj)
        
        obj.hide_viewport = True
        obj.hide_set(True)
        
        for select_obj in context.selected_objects:
            select_obj.select_set(False)
        preview_obj.select_set(True)
        context.view_layer.objects.active = preview_obj
        
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        try:
                            # Switch shading type to SOLID to display color attributes
                            space.shading.type = 'SOLID'
                            space.shading.color_type = 'VERTEX'
                            try:
                                space.shading.light = 'FLAT'
                            except Exception:
                                pass
                            if hasattr(preview_mesh, "color_attributes"):
                                preview_mesh.color_attributes.active = color_layer
                            if hasattr(preview_mesh, "attributes"):
                                preview_mesh.attributes.active_color = color_layer
                        except TypeError:
                            # Fallback to Blender 4.x: set to ATTRIBUTE and set attribute_name
                            try:
                                space.shading.type = 'SOLID'
                                space.shading.color_type = 'ATTRIBUTE'
                                space.shading.attribute_name = "SG_PrintPreview"
                                try:
                                    space.shading.light = 'FLAT'
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        except Exception:
                            pass
                        
        self.report({'INFO'}, "Sliced preview generated.")


class StableGenPreviewSlicedCancel(bpy.types.Operator):
    """Cancel the active sliced preview generation."""
    bl_idname = "object.stablegen_preview_sliced_cancel"
    bl_label = "Cancel Preview Slices"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        op = next(
            (op for win in context.window_manager.windows
             for op in win.modal_operators
             if op.bl_idname == 'OBJECT_OT_stablegen_preview_sliced'),
            None
        )
        if op and hasattr(op, "_thread_status") and op._thread_status:
            op._thread_status["cancel_requested"] = True
            self.report({'INFO'}, "Cancellation requested...")
        return {'FINISHED'}


# ── helpers ──────────────────────────────────────────────────────────────

def _rgb_to_hsv_numpy(rgb):
    """Vectorized RGB to HSV conversion in NumPy."""
    import numpy as np
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_val = np.max(rgb, axis=1)
    min_val = np.min(rgb, axis=1)
    delta = max_val - min_val
    
    h = np.zeros_like(max_val)
    mask = delta > 0.0001
    
    # Red is max
    mask_r = mask & (max_val == r)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    
    # Green is max
    mask_g = mask & (max_val == g)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    
    # Blue is max
    mask_b = mask & (max_val == b)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    
    h = (h / 6.0) % 1.0
    s = np.zeros_like(max_val)
    s[mask] = delta[mask] / max_val[mask]
    v = max_val
    
    return np.stack([h, s, v], axis=1)


def linear_to_srgb_numpy(linear_arr):
    """Vectorized conversion from linear RGB to sRGB in NumPy."""
    import numpy as np
    srgb = np.where(
        linear_arr <= 0.0031308,
        linear_arr * 12.92,
        1.055 * np.power(np.maximum(linear_arr, 0.0), 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb, 0.0, 1.0)


def srgb_to_linear_numpy(srgb_arr):
    """Vectorized conversion from sRGB to linear RGB in NumPy."""
    import numpy as np
    linear = np.where(
        srgb_arr <= 0.04045,
        srgb_arr / 12.92,
        np.power((srgb_arr + 0.055) / 1.055, 2.4)
    )
    return np.clip(linear, 0.0, 1.0)


def classify_filaments(palette_rgbs):
    """Classifies a list of physical filament RGBs into subtractive preset roles.
    
    Returns (mode, channel_indices) where:
      mode: 'cmy', 'wcmy', 'kcmy', 'wkcmy', '2c', or 'solid'
      channel_indices: list of indices mapping role order to palette_rgbs indices
    """
    n = len(palette_rgbs)
    if n == 0:
        return 'solid', []
        
    targets = {
        'W': (1.0, 1.0, 1.0),
        'K': (0.0, 0.0, 0.0),
        'C': (0.0, 1.0, 1.0),
        'M': (1.0, 0.0, 1.0),
        'Y': (1.0, 1.0, 0.0)
    }
    
    def dist(rgb1, rgb2):
        return sum((rgb1[i] - rgb2[i])**2 for i in range(3))
        
    matched = {}
    remaining_indices = list(range(n))
    
    for role in ['W', 'K', 'C', 'M', 'Y']:
        if not remaining_indices:
            break
        target = targets[role]
        best_idx = min(remaining_indices, key=lambda idx: dist(palette_rgbs[idx], target))
        threshold = 0.05 if role in ['W', 'K'] else 0.25
        if dist(palette_rgbs[best_idx], target) < threshold:
            matched[role] = best_idx
            remaining_indices.remove(best_idx)
            
    has_c = 'C' in matched
    has_m = 'M' in matched
    has_y = 'Y' in matched
    has_w = 'W' in matched
    has_k = 'K' in matched

    # If there are spools in the palette that didn't match a standard role,
    # force the general solver to ensure no colors are ignored!
    if len(matched) < n:
        return 'solid', list(range(n))
    
    if has_c and has_m and has_y:
        if has_w and has_k:
            return 'wkcmy', [matched['W'], matched['K'], matched['C'], matched['M'], matched['Y']]
        elif has_w:
            return 'wcmy', [matched['W'], matched['C'], matched['M'], matched['Y']]
        elif has_k:
            return 'kcmy', [matched['K'], matched['C'], matched['M'], matched['Y']]
        else:
            return 'cmy', [matched['C'], matched['M'], matched['Y']]
            
    if n == 2:
        return '2c', [0, 1]
        
    return 'solid', list(range(n))


def decompose_subtractive(r, g, b, mode, palette_rgbs, channel_indices):
    """Calculates physical filament fractions using Primed3D's subtractive mixing models.
    
    r, g, b are standard sRGB values in [0, 255].
    """
    if mode == 'cmy':
        return [1.0 - r/255.0, 1.0 - g/255.0, 1.0 - b/255.0]
    elif mode == 'wcmy':
        cd = 1.0 - r/255.0
        md = 1.0 - g/255.0
        yd = 1.0 - b/255.0
        kd = min(cd, md, yd)
        wd = 1.0 - max(cd, md, yd)
        k3 = kd / 3.0
        return [wd, cd - kd + k3, md - kd + k3, yd - kd + k3]
    elif mode == 'kcmy':
        cd = 1.0 - r/255.0
        md = 1.0 - g/255.0
        yd = 1.0 - b/255.0
        k_max = min(cd, md, yd)
        chroma = max(cd, md, yd) - k_max
        k_thresh = max(0.0, (k_max - 0.35) / 0.65)
        k_scale = (1.0 - chroma * 0.7) * k_thresh
        kd = k_max * k_scale
        return [kd, cd - kd, md - kd, yd - kd]
    elif mode == 'wkcmy':
        cd = 1.0 - r/255.0
        md = 1.0 - g/255.0
        yd = 1.0 - b/255.0
        k_max = min(cd, md, yd)
        chroma = max(cd, md, yd) - k_max
        k_scale = 1.0 - chroma * 0.7
        kd = k_max * k_scale
        wd = 1.0 - max(cd, md, yd)
        return [wd, kd, cd - kd, md - kd, yd - kd]
    elif mode == '2c':
        spool_a = palette_rgbs[channel_indices[0]]
        spool_b = palette_rgbs[channel_indices[1]]
        ar, ag, ab_ = int(round(spool_a[0]*255)), int(round(spool_a[1]*255)), int(round(spool_a[2]*255))
        br, bg, bb = int(round(spool_b[0]*255)), int(round(spool_b[1]*255)), int(round(spool_b[2]*255))
        d_r, d_g, d_b = br - ar, bg - ag, bb - ab_
        dot2 = d_r*d_r + d_g*d_g + d_b*d_b
        if dot2 < 1:
            return [0.5, 0.5]
        t = max(0.0, min(1.0, ((r - ar) * d_r + (g - ag) * d_g + (b - ab_) * d_b) / dot2))
        return [1.0 - t, t]
    return []


def solve_color_mixing_numpy(fil_rgbs, target_rgbs, iterations=None, init_method=None, model_formula=None, scat_weight=None, penalty_weight=None):
    """Vectorized Frank-Wolfe optimization solver for arbitrary filament mixing.
    
    Supports Linear Additive, Linear Subtractive, and physical Kubelka-Munk models
    with configurable convergence steps, initialization method, and scattering weights.
    """
    import numpy as np
    import bpy
    
    U = target_rgbs.shape[0]
    N = fil_rgbs.shape[0]
    if N == 0:
        return np.zeros((U, 0), dtype=np.float32)
        
    # Read dynamic configuration from Blender scene if context is available
    scene = getattr(bpy.context, "scene", None)
    
    if init_method is None:
        init_method = getattr(scene, "stablegen_print_solver_init", 'CLOSEST') if scene else 'CLOSEST'
    if iterations is None:
        iterations = getattr(scene, "stablegen_print_solver_steps", 80) if scene else 80
    if model_formula is None:
        model_formula = getattr(scene, "stablegen_print_solver_formula", 'KUBELKA_MUNK') if scene else 'KUBELKA_MUNK'
    if scat_weight is None:
        scat_weight = getattr(scene, "stablegen_print_scattering_weight", 10.0) if scene else 10.0
    if penalty_weight is None:
        penalty_weight = getattr(scene, "stablegen_print_saturation_penalty", 1.0) if scene else 1.0


    # 1. Precompute absorption K and scattering S for Kubelka-Munk
    fil_val = fil_rgbs.max(axis=1)
    fil_min = fil_rgbs.min(axis=1)
    fil_sat = np.zeros_like(fil_val)
    mask = fil_val > 1e-5
    fil_sat[mask] = (fil_val[mask] - fil_min[mask]) / fil_val[mask]
    
    # Precompute target saturation for one-sided saturation penalty
    target_val = target_rgbs.max(axis=1)
    target_min = target_rgbs.min(axis=1)
    target_sat = np.zeros_like(target_val)
    t_mask = target_val > 1e-5
    target_sat[t_mask] = (target_val[t_mask] - target_min[t_mask]) / target_val[t_mask]
    
    # Scattering: base of 1.0, up to scat_weight for bright desaturated spools
    S_filaments = 1.0 + scat_weight * (1.0 - fil_sat) * fil_val  # (N,)
    S_filaments = S_filaments[:, None]  # (N, 1)
    
    R_clip = np.clip(fil_rgbs, 0.001, 0.999)
    K_filaments = S_filaments * ((1.0 - R_clip)**2) / (2.0 * R_clip)  # (N, 3)
    
    # Target values to match
    target_rgbs = np.clip(target_rgbs, 0.001, 0.999)
    
    # Initialize fractions
    if init_method == 'CLOSEST':
        # Compute reflectances of single filaments to find the best initial guess
        if model_formula == 'KUBELKA_MUNK':
            ratio_fil = K_filaments / S_filaments  # (N, 3)
            R_fil = 1.0 + ratio_fil - np.sqrt(ratio_fil**2 + 2.0 * ratio_fil)  # (N, 3)
        else:
            R_fil = fil_rgbs
            
        dists = np.sum((R_fil[:, None, :] - target_rgbs[None, :, :])**2, axis=2)  # (N, U)
        
        # Apply one-sided saturation penalty to initial distances
        if penalty_weight > 0.0:
            diff = target_sat[None, :] - fil_sat[:, None]
            penalties = penalty_weight * target_sat[None, :] * np.maximum(0.0, diff)
            dists = dists + penalties
            
        best_init = np.argmin(dists, axis=0)  # (U,)
        
        f = np.zeros((U, N), dtype=np.float32)
        f[np.arange(U), best_init] = 1.0
    else:  # EQUAL
        f = np.full((U, N), 1.0 / N, dtype=np.float32)
        
    for step in range(iterations):
        gamma = 2.0 / (step + 3)
        errors = np.empty((N, U), dtype=np.float32)
        
        for i in range(N):
            f_cand = f * (1.0 - gamma)
            f_cand[:, i] += gamma
            
            if model_formula == 'KUBELKA_MUNK':
                K_mix = np.dot(f_cand, K_filaments)  # (U, 3)
                S_mix = np.dot(f_cand, S_filaments)  # (U, 3)
                ratio = K_mix / S_mix
                R_cand = 1.0 + ratio - np.sqrt(ratio**2 + 2.0 * ratio)
            elif model_formula == 'ADDITIVE':
                R_cand = np.dot(f_cand, fil_rgbs)
            else:  # SUBTRACTIVE
                R_cand = 1.0 - np.dot(f_cand, 1.0 - fil_rgbs)
                
            base_err = np.sum((R_cand - target_rgbs)**2, axis=1)
            
            # Apply one-sided saturation penalty
            if penalty_weight > 0.0:
                pen = penalty_weight * target_sat * np.maximum(0.0, target_sat - fil_sat[i])
                errors[i] = base_err + pen
            else:
                errors[i] = base_err
            
        best_i = np.argmin(errors, axis=0)  # (U,)
        
        f *= (1.0 - gamma)
        f[np.arange(U), best_i] += gamma
        
    # Apply FDM-specific pruning to eliminate tiny noisy fractions below 8%
    threshold = 0.08
    row_sums = f.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    f_frac = f / row_sums
    f_frac[f_frac < threshold] = 0.0
    
    new_sums = f_frac.sum(axis=1, keepdims=True)
    zero_mask = (new_sums == 0).ravel()
    if np.any(zero_mask):
        max_indices = np.argmax(f, axis=1)
        f_frac[zero_mask, max_indices[zero_mask]] = 1.0
        new_sums[zero_mask] = 1.0
        
    return f_frac / new_sums


def lv(a, b, z):
    """Linear interpolation of vertex coordinates at intersection height z."""
    # Lexicographically sort endpoints to ensure deterministic interpolation order on shared edges
    if (a[2], a[0], a[1]) > (b[2], b[0], b[1]):
        a, b = b, a
    d = b[2] - a[2]
    if abs(d) < 1e-12:
        return [a[0], a[1], z]
    t = (z - a[2]) / d
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), z]


def cP(poly, z, ab):
    """Clips a polygon against a horizontal plane z. Keep vertices on correct side (ab)."""
    o = []
    n_pts = len(poly)
    for i in range(n_pts):
        c = poly[i]
        n = poly[(i + 1) % n_pts]
        cI = (c[2] >= z - 1e-8) if ab else (c[2] <= z + 1e-8)
        nI = (n[2] >= z - 1e-8) if ab else (n[2] <= z + 1e-8)
        if cI:
            o.append(c)
            if not nI:
                o.append(lv(c, n, z))
        elif nI:
            o.append(lv(c, n, z))
    return o


def cPs(ps, z, ab):
    """Clips a list of polygons against plane z."""
    o = []
    for p in ps:
        c = cP(p, z, ab)
        if len(c) >= 3:
            o.append(c)
    return o


def cTZ(tri, zB, zT):
    """Clips a triangle into horizontal band [zB, zT]. Returns sub-triangles."""
    ps = [tri]
    ps = cPs(ps, zB, True)
    ps = cPs(ps, zT, False)
    o = []
    for p in ps:
        if len(p) < 3:
            continue
        for i in range(1, len(p) - 1):
            o.append([p[0], p[i], p[i+1]])
    return o


def _match_colors_hsv(face_rgbs, palette_rgbs, chroma_threshold=0.35):
    """Perceptual color matching using HSV space distance with warped saturation and category gates."""
    import numpy as np
    hsv_faces = _rgb_to_hsv_numpy(face_rgbs)
    hsv_palette = _rgb_to_hsv_numpy(np.array(palette_rgbs, dtype=np.float32))
    
    h1, s1, v1 = hsv_faces[:, 0, None], hsv_faces[:, 1, None], hsv_faces[:, 2, None]  # (N, 1)
    h2, s2, v2 = hsv_palette[None, :, 0], hsv_palette[None, :, 1], hsv_palette[None, :, 2]  # (1, K)
    
    # Warp Saturation to reflect human chromatic boundaries
    # This maps e.g. S=0.6 to S_warped=0.93 (chromatic), but S=0.25 to S_warped=0.58 (neutral)
    def warp_s(s):
        return 1.0 - (1.0 - s) ** 3
        
    s1_w = warp_s(s1)
    s2_w = warp_s(s2)
    
    # Circular Hue difference
    dh = np.abs(h1 - h2)
    dh = np.minimum(dh, 1.0 - dh)
    
    ds = s1_w - s2_w
    dv = v1 - v2
    
    # Highly prioritize Hue matching for chromatic colors
    # We use a large hue weight if both colors are somewhat chromatic
    # but scale it down if either color is truly neutral (achromatic)
    chromatic_gate = np.minimum(s1, s2)
    hue_weight = 15.0 * np.clip(chromatic_gate / 0.15, 0.0, 1.0)
    
    # We also penalize matching between different saturation classes (chromatic vs neutral)
    # class 1: S >= chroma_threshold (chromatic), class 0: S < chroma_threshold (neutral)
    c1 = (s1 >= chroma_threshold).astype(np.float32)
    c2 = (s2 >= chroma_threshold).astype(np.float32)
    class_penalty = 1.0 * np.abs(c1 - c2)
    
    # HSV distance formula
    dists = hue_weight * (dh ** 2) + 1.0 * (ds ** 2) + 0.5 * (dv ** 2) + class_penalty
    return np.argmin(dists, axis=1)


def _smooth_colors_vectorized(tris, paint_colors, V, K, iterations=1):
    """Majority vote smoothing on mesh faces using shared vertices."""
    import numpy as np
    if iterations <= 0 or len(tris) == 0:
        return paint_colors
        
    T = len(tris)
    smoothed = paint_colors.copy()
    
    for _ in range(iterations):
        # Create one-hot counts for each face
        face_one_hot = np.zeros((T, K), dtype=np.float32)
        face_one_hot[np.arange(T), smoothed - 1] = 1.0
        
        # Accumulate face color one-hots at vertices
        v_color_counts = np.zeros((V, K), dtype=np.float32)
        np.add.at(v_color_counts, tris[:, 0], face_one_hot)
        np.add.at(v_color_counts, tris[:, 1], face_one_hot)
        np.add.at(v_color_counts, tris[:, 2], face_one_hot)
        
        # Sum vertex counts for each triangle
        tri_votes = (v_color_counts[tris[:, 0]] + 
                     v_color_counts[tris[:, 1]] + 
                     v_color_counts[tris[:, 2]])
        
        # Determine majority color (1-based index)
        smoothed = np.argmax(tri_votes, axis=1) + 1
        
    return smoothed


def _get_triangulated_mesh(obj, apply_transforms):
    """Return a copy of *obj* triangulated, in world-space if requested.

    Returns ``(mesh, eval_obj)``.  The caller **must** call
    ``eval_obj.to_mesh_clear()`` to free the temporary mesh.
    """
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deps)
    temp = eval_obj.to_mesh()
    temp.calc_loop_triangles()
    if apply_transforms:
        mat = obj.matrix_world
        temp.transform(mat)
    return temp, eval_obj


def _find_texture_image(obj):
    """Locate the best colour texture image in the object's material.

    The bake pipeline appends a ``{name}_baked`` material — iterate in
    *reverse* so that the baked material is examined first.

    Order of preference within each material:
      1. ``TEX_IMAGE`` connected to Principled BSDF Base Color.
      2. Any ``TEX_IMAGE`` node present in the material.
    """
    if not obj.data.materials:
        return None
    for mat in reversed(obj.data.materials):
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bc = node.inputs.get("Base Color")
                    if bc and bc.links:
                        src = bc.links[0].from_node
                        if src.type == 'TEX_IMAGE' and src.image:
                            return src.image
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    return node.image
    return None


def _sample_image_pixel(image, u, v):
    """Return RGBA tuple sampled from *image* at normalized UV *u*, *v*."""
    if not image or image.size[0] == 0 or image.size[1] == 0:
        return (0.5, 0.5, 0.5, 1.0)
    w, h = image.size
    u = u - math.floor(u)
    v = v - math.floor(v)
    px = min(int(u * w), w - 1)
    py = min(int(v * h), h - 1)
    idx = (py * w + px) * 4
    pixels = image.pixels
    if idx + 3 >= len(pixels):
        return (0.5, 0.5, 0.5, 1.0)
    return (pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3])


def _kmeans_quantize(colors, k, max_iter=30):
    """Quantize a list of RGBA *colors* into *k* clusters via k-means.

    Returns ``(assignments, centroids)`` where *assignments* is an
    ``int`` index per input colour and *centroids* is a list of
    ``(r, g, b)`` tuples.
    """
    n = len(colors)
    if n == 0:
        return [], []
    k = max(1, min(k, n))
    rgb = [c[:3] for c in colors]

    # k-means++ initialisation
    centroids = [random.choice(rgb)]
    for _ in range(1, k):
        dists = []
        for c in rgb:
            d = min(sum((c[j] - cen[j]) ** 2 for j in range(3)) for cen in centroids)
            dists.append(d)
        total = sum(dists)
        if total == 0:
            centroids.append(random.choice(rgb))
            continue
        r = random.random() * total
        cumulative = 0.0
        chosen = rgb[-1]
        for i, d in enumerate(dists):
            cumulative += d
            if cumulative >= r:
                chosen = rgb[i]
                break
        centroids.append(chosen)

    assignments = [0] * n
    for _ in range(max_iter):
        changed = False
        for i, c in enumerate(rgb):
            best_d = sum((c[j] - centroids[0][j]) ** 2 for j in range(3))
            best = 0
            for ci in range(1, k):
                d = sum((c[j] - centroids[ci][j]) ** 2 for j in range(3))
                if d < best_d:
                    best_d = d
                    best = ci
            if assignments[i] != best:
                assignments[i] = best
                changed = True
        if not changed:
            break
        sums = [(0.0, 0.0, 0.0) for _ in range(k)]
        counts = [0] * k
        for i, c in enumerate(rgb):
            ci = assignments[i]
            s = sums[ci]
            sums[ci] = (s[0] + c[0], s[1] + c[1], s[2] + c[2])
            counts[ci] += 1
        centroids = []
        for ci in range(k):
            if counts[ci]:
                centroids.append(tuple(v / counts[ci] for v in sums[ci]))
            else:
                centroids.append((0.5, 0.5, 0.5))
    return assignments, centroids


def _rgb_to_hex(rgb):
    """Convert a ``(r, g, b)`` float triple to an sRGB hex string."""
    def _clamp(v):
        return max(0, min(255, int(round(v * 255))))
    return f"#{_clamp(rgb[0]):02x}{_clamp(rgb[1]):02x}{_clamp(rgb[2]):02x}"


# ── 3MF XML construction ─────────────────────────────────────────────────

def _xml_with_xmlns(element, xmlns_uri, extra_ns=None):
    """Serialize an Element to a UTF-8 XML string with ``xmlns``.

    *extra_ns* is an optional ``(prefix, uri)`` pair added as an
    additional  ``xmlns:prefix="uri"`` declaration on the root.
    """
    raw = ET.tostring(element, encoding='unicode')
    tag = element.tag.split('}')[-1]
    pos = raw.index(f'<{tag}')
    gt = raw.index('>', pos)
    ns_decl = f' xmlns="{xmlns_uri}"'
    if extra_ns:
        ns_decl += f' xmlns:{extra_ns[0]}="{extra_ns[1]}"'
    out = raw[:gt] + ns_decl + raw[gt:]
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + out


def _build_model_settings_element(parent_id, child_ids_and_extruders):
    """Build a ``<config>`` Element tree for model_settings.config.
    
    Structure:
    <config>
      <object id="PARENT_ID">
        <part id="CHILD_ID" subtype="normal_part">
          <metadata key="name" value="Part_{CHILD_ID}"/>
          <!-- No extruder key: paint_color on triangles maps directly to
               global filament IDs. Setting an explicit extruder here causes
               the slicer to offset all paint_color values through the part's
               base extruder, which remaps every triangle to Filament 1. -->
        </part>
        ...
      </object>
    </config>
    """
    config = ET.Element('config')
    obj_el = ET.SubElement(config, 'object')
    obj_el.set('id', str(parent_id))
    
    for child_id, _extruder in child_ids_and_extruders:
        part_el = ET.SubElement(obj_el, 'part')
        part_el.set('id', str(child_id))
        part_el.set('subtype', 'normal_part')
        
        meta_name = ET.SubElement(part_el, 'metadata')
        meta_name.set('key', 'name')
        meta_name.set('value', f'Part_{child_id}')
        # Intentionally omit extruder metadata — paint_color already encodes
        # the global filament index directly via (filament_id << 2).
        
    return config



def _xml_to_string(element):
    """Serialize an Element to a UTF-8 XML string."""
    raw = ET.tostring(element, encoding='unicode')
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + raw


def _build_model_element(verts_arr, tris_arr, paint_arr, palette, global_to_extruder=None):
    """Build a ``<model>`` XML string using a single object wrapped in components.

    This ensures that the mesh is exported as a single, perfectly manifold object,
    while maintaining the component wrapper structure expected by OrcaSlicer/Bambu Studio
    to prevent circular reference crashes in model_settings.config.
    """
    import numpy as np

    resources_parts = []
    
    # ── Geometry Mesh Object (ID 1) ──
    geom_obj_id = 1
    resources_parts.append(f'<object id="{geom_obj_id}" type="model">')
    
    # Format vertices
    verts_strs = [f'<vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>' for v in verts_arr]
    
    # Format triangles, adding the paint_color attribute if assigned
    def get_paint_color_hex(ext_id):
        if ext_id == 0:
            return "0"
        elif ext_id == 1:
            return "4"
        elif ext_id == 2:
            return "8"
        else:
            n = ext_id - 3
            chunks = []
            while n >= 15:
                chunks.append("F")
                n -= 15
            chunks.append(f"{n:X}")
            return "".join(reversed(chunks)) + "C"

    tris_strs = []
    for i, tri in enumerate(tris_arr):
        # 1-based index in global_palette
        seq_ext = int(paint_arr[i])
        if global_to_extruder and seq_ext in global_to_extruder:
            ext_id = global_to_extruder[seq_ext]
        else:
            ext_id = seq_ext
        
        pc_str = get_paint_color_hex(ext_id)
        tris_strs.append(f'<triangle v1="{tri[0]}" v2="{tri[1]}" v3="{tri[2]}" paint_color="{pc_str}"/>')
        
    resources_parts.append(f'<mesh><vertices>{"".join(verts_strs)}</vertices><triangles>{"".join(tris_strs)}</triangles></mesh>')
    resources_parts.append('</object>')

    # ── Parent Assembly Object with Component (ID 2) ──
    parent_obj_id = 2
    resources_parts.append(f'<object id="{parent_obj_id}" type="model"><components>')
    resources_parts.append(f'<component objectid="{geom_obj_id}"/>')
    resources_parts.append('</components></object>')

    resources_xml = "".join(resources_parts)
    build_xml = f'<build><item objectid="{parent_obj_id}"/></build>'

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">'
        f'<resources>{resources_xml}</resources>'
        f'{build_xml}'
        '</model>'
    )

    parent_id = parent_obj_id
    child_ids_and_extruders = [(geom_obj_id, 1)]
    return model_xml, parent_id, child_ids_and_extruders


def _write_3mf_archive(filepath, model_element_or_str, model_settings_element=None, project_settings_json=None, prusa_config=None, prusa_model_config=None):
    """Package the model XML into a 3MF (ZIP) archive."""
    if isinstance(model_element_or_str, str):
        model_str = model_element_or_str
    else:
        model_str = _xml_with_xmlns(model_element_or_str, NS_3MF,
                                    extra_ns=('m', NS_MAT))

    ct = ET.Element('Types')
    ET.SubElement(ct, 'Default', {
        'Extension': 'rels',
        'ContentType': 'application/vnd.openxmlformats-package.relationships+xml',
    })
    ET.SubElement(ct, 'Default', {
        'Extension': 'model',
        'ContentType': 'application/vnd.ms-package.3dmanufacturing-3dmodel+xml',
    })
    if model_settings_element is not None or project_settings_json is not None:
        ET.SubElement(ct, 'Default', {
            'Extension': 'config',
            'ContentType': 'application/xml',
        })
    ct_str = _xml_with_xmlns(ct, NS_CT)

    rels = ET.Element('Relationships')
    ET.SubElement(rels, 'Relationship', {
        'Target': '/3D/3dmodel.model',
        'Id': 'rel1',
        'Type': 'http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel',
    })
    rels_str = _xml_with_xmlns(rels, NS_REL)

    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', ct_str)
        zf.writestr('_rels/.rels', rels_str)
        zf.writestr('3D/3dmodel.model', model_str)
        if model_settings_element is not None:
            settings_str = _xml_to_string(model_settings_element)
            zf.writestr('Metadata/model_settings.config', settings_str)
        if project_settings_json is not None:
            zf.writestr('Metadata/project_settings.config', project_settings_json)
            slice_info_str = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<config>\n'
                '  <header>\n'
                '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
                '    <header_item key="X-BBL-Client-Version" value=""/>\n'
                '  </header>\n'
                '</config>'
            )
            zf.writestr('Metadata/slice_info.config', slice_info_str)
        if prusa_config is not None:
            zf.writestr('Metadata/Slic3r_PE.config', prusa_config)
        if prusa_model_config is not None:
            zf.writestr('Metadata/Slic3r_PE_model.config', prusa_model_config)


def _collect_face_colors(mesh, image):
    """Sample texture colour at the centre of each loop triangle using optimized NumPy vectorization.

    Prefers the ``BakeUV`` channel (created by the bake pipeline);
    falls back to the active UV layer, then to any available UV layer.
    """
    import numpy as np

    if not mesh.loop_triangles or not mesh.uv_layers:
        return np.ones((len(mesh.loop_triangles), 4), dtype=np.float32) * 0.5

    uv_layer = mesh.uv_layers.get("BakeUV")
    if uv_layer is None:
        uv_layer = mesh.uv_layers.active
    if uv_layer is None and mesh.uv_layers:
        uv_layer = mesh.uv_layers[0]
    if uv_layer is None:
        return np.ones((len(mesh.loop_triangles), 4), dtype=np.float32) * 0.5

    # Force load image pixels if not already loaded in memory
    try:
        if len(image.pixels) > 0:
            _ = image.pixels[0]
    except Exception as e:
        print(f"[StableGen 3MF] Warning: could not force load image pixels: {e}")

    # 1. Fetch entire image pixels to numpy array once (extremely fast)
    w, h = image.size
    pixels = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape((h, w, 4))

    # 2. Fetch all UVs
    n_loops = len(uv_layer.data)
    uv_data = np.empty(n_loops * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_data)
    uvs = uv_data.reshape((-1, 2))

    # 3. Fetch all loop triangle indices
    n_tris = len(mesh.loop_triangles)
    tri_loops = np.empty(n_tris * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", tri_loops)
    tri_loops = tri_loops.reshape((-1, 3))

    # 4. Compute average UV per triangle vectorially
    avg_uvs = uvs[tri_loops].mean(axis=1)  # (n_tris, 2)

    # 5. Map UVs to pixel coordinates
    u = avg_uvs[:, 0]
    v = avg_uvs[:, 1]
    
    # Handle wrapping
    u = u - np.floor(u)
    v = v - np.floor(v)
    
    px = np.clip((u * w).astype(np.int32), 0, w - 1)
    py = np.clip((v * h).astype(np.int32), 0, h - 1)

    # 6. Sample colors at once using advanced indexing
    face_colors = pixels[py, px].copy()  # (n_tris, 4)

    # Determine color space of the image.
    # If the colorspace settings are 'Non-Color' or 'Raw' and the image is not a float/HDR texture,
    # the pixels in image.pixels are raw sRGB values, not linear.
    # In all other cases (e.g. standard 'sRGB', or 'Linear'), image.pixels are linear Rec. 709/sRGB floats.
    is_linear = True
    cs_name = "unknown"
    if hasattr(image, "colorspace_settings"):
        cs_name = image.colorspace_settings.name
        cs_name_lower = cs_name.lower()
        if cs_name_lower in ('non-color', 'raw'):
            is_float = getattr(image, "is_float", False)
            if not is_float:
                is_linear = False

    print(f"[StableGen 3MF] Texture '{image.name}' color space settings: '{cs_name}', interpreted as {'Linear' if is_linear else 'sRGB'}")

    if not is_linear:
        # Convert sRGB raw image pixels to linear space for consistent downstream behavior
        face_colors[:, :3] = srgb_to_linear_numpy(face_colors[:, :3])

    # Debug: print color range
    uniq = set(tuple(round(float(c), 2) for c in col[:3]) for col in face_colors[:1000]) # Sample a subset for performance
    print(f"[StableGen 3MF] Sampled {len(face_colors)} faces, found {len(uniq)}+ unique color values")

    return face_colors



def _keep_largest_island(bm):
    """Keep only the largest connected face component of ``bm`` (by surface area).

    Two faces are considered connected when they share at least one edge. All
    non-largest components are deleted with ``context='FACES'`` and the loose
    edges/verts left behind are also removed.

    Returns ``(n_components, n_removed_faces, removed_area)`` for diagnostics.
    """
    from collections import defaultdict
    import bmesh

    if not bm.faces:
        return 0, 0, 0.0

    adj = defaultdict(list)
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) < 2:
            continue
        a = lf[0]
        for b in lf[1:]:
            adj[a].append(b)
            adj[b].append(a)

    unvisited = set(bm.faces)
    components = []

    while unvisited:
        seed = next(iter(unvisited))
        unvisited.discard(seed)
        comp_faces = [seed]
        comp_area = seed.calc_area()
        stack = [seed]
        while stack:
            f = stack.pop()
            for nb in adj.get(f, ()):
                if nb in unvisited:
                    unvisited.discard(nb)
                    comp_faces.append(nb)
                    comp_area += nb.calc_area()
                    stack.append(nb)
        components.append((comp_area, comp_faces))

    n_components = len(components)
    if n_components <= 1:
        return n_components, 0, 0.0

    largest_idx = max(range(n_components), key=lambda i: components[i][0])
    to_delete = []
    removed_area = 0.0
    for i, (area, faces_list) in enumerate(components):
        if i == largest_idx:
            continue
        to_delete.extend(faces_list)
        removed_area += area

    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context='FACES')
        loose_edges = [e for e in bm.edges if not e.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')
        loose_verts = [v for v in bm.verts if not v.link_edges]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    return n_components, len(to_delete), removed_area


def _collect_face_visibility(mesh):
    """Collect per-face visibility from ``_SG_VisWeight_*`` mesh attributes.

    Sums all per-vertex visibility weight attributes (one per texturing camera)
    and averages per triangle via ``loop_triangles``.  Returns a NumPy float32
    array of shape ``(n_tris,)`` or ``None`` when no visibility attributes exist.
    """
    import numpy as np

    vis_attrs = [attr for attr in mesh.attributes
                 if attr.name.startswith("_SG_VisWeight_") and attr.domain == 'POINT']
    if not vis_attrs:
        return None

    n_verts = len(mesh.vertices)
    total = np.zeros(n_verts, dtype=np.float32)
    for attr in vis_attrs:
        vals = np.empty(n_verts, dtype=np.float32)
        attr.data.foreach_get("value", vals)
        total += vals

    n_tris = len(mesh.loop_triangles)
    tri_verts = np.empty(n_tris * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_verts)
    tri_verts = tri_verts.reshape((-1, 3))

    face_vis = total[tri_verts].mean(axis=1)
    return face_vis


def _canonical_face_cycle(vertex_ids):
    """Return a rotation-stable, winding-sensitive face identifier."""
    ids = tuple(int(vertex_id) for vertex_id in vertex_ids)
    if not ids:
        return ids
    minimum = min(ids)
    return min(
        ids[index:] + ids[:index]
        for index, vertex_id in enumerate(ids)
        if vertex_id == minimum
    )


def _capture_source_face_windings(bm):
    """Tag source vertices and capture every source face's directed cycle.

    BMesh operations may delete/reindex elements, and a fill can recreate the
    exact same triangle with the opposite winding.  A temporary vertex layer
    gives surviving/reused vertices stable source identities so that matching
    output faces can be restored without relying on mutable BMesh indices.
    """
    layer_name = ".sg_source_vertex_id"
    suffix = 0
    while bm.verts.layers.int.get(layer_name) is not None:
        suffix += 1
        layer_name = f".sg_source_vertex_id_{suffix}"
    source_vertex_layer = bm.verts.layers.int.new(layer_name)

    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    for vertex in bm.verts:
        # Zero remains reserved for vertices created after this snapshot.
        vertex[source_vertex_layer] = vertex.index + 1

    source_windings = {
        _canonical_face_cycle(
            vertex[source_vertex_layer] for vertex in face.verts
        )
        for face in bm.faces
    }
    return source_vertex_layer, source_windings


def _restore_matching_source_face_windings(
        bm, source_vertex_layer, source_windings):
    """Restore source winding on output faces with identical source vertices.

    Returns ``(matching_faces, flipped_faces)``.  Genuinely new faces and
    faces whose vertices were changed by welding/remeshing are left alone.
    """
    matching_faces = 0
    flipped_faces = 0
    for face in bm.faces:
        vertex_ids = [
            int(vertex[source_vertex_layer]) for vertex in face.verts
        ]
        if not vertex_ids or any(vertex_id <= 0 for vertex_id in vertex_ids):
            continue

        winding = _canonical_face_cycle(vertex_ids)
        if winding in source_windings:
            matching_faces += 1
            continue

        reverse_winding = _canonical_face_cycle(reversed(vertex_ids))
        if reverse_winding in source_windings:
            face.normal_flip()
            matching_faces += 1
            flipped_faces += 1

    return matching_faces, flipped_faces


def _mesh_topology_counts(mesh):
    """Return the print-relevant topology defects in *mesh*.

    The counts use geometric mesh vertex indices, not render-loop vertices, so
    UV seams do not appear as cracks.  A printable surface must have exactly
    two faces per edge, no loose edges, and no zero-area faces.
    """
    face_counts = {
        tuple(sorted(edge.vertices[:])): 0
        for edge in mesh.edges
    }
    for polygon in mesh.polygons:
        for edge_key in polygon.edge_keys:
            key = tuple(sorted(edge_key))
            face_counts[key] = face_counts.get(key, 0) + 1

    counts = tuple(face_counts.values())
    return {
        "boundary": sum(count == 1 for count in counts),
        "non_manifold": sum(count > 2 for count in counts),
        "loose": sum(count == 0 for count in counts),
        "degenerate": sum(
            len(set(polygon.vertices[:])) < 3
            or polygon.area <= 1.0e-12
            for polygon in mesh.polygons
        ),
    }


def _require_printable_mesh(mesh, *, label="Mesh", require_closed=True):
    """Raise with actionable counts when *mesh* is unsafe for a slicer."""
    topology = _mesh_topology_counts(mesh)
    blocking = {
        "non-manifold edges": topology["non_manifold"],
        "loose edges": topology["loose"],
        "degenerate faces": topology["degenerate"],
    }
    if require_closed:
        blocking = {
            "open boundary edges": topology["boundary"],
            **blocking,
        }
    problems = [
        f"{count} {name}"
        for name, count in blocking.items()
        if count
    ]
    if problems:
        raise RuntimeError(
            f"{label} is not print-safe: " + ", ".join(problems) + ". "
            "StableGen stopped before writing a broken slicer file."
        )
    return topology


def _make_solid_mesh_object(
        obj, fill_gaps=True, visible_faces=None, keep_largest_island=False,
        face_visibility=None, *, use_evaluated_mesh=True, raycast_count=None,
        weld_distance=0.0, robust_fill=True, smooth_caps=True):
    import bmesh
    from mathutils.bvhtree import BVHTree
    import numpy as np
    import mathutils
    import math
    import bpy
    import time

    t_start = time.perf_counter()
    if use_evaluated_mesh:
        temp, eval_obj = _get_triangulated_mesh(
            obj, apply_transforms=False
        )
    else:
        # General post-processing must operate on the base/rest mesh. Using
        # evaluated_get().to_mesh() here would bake the current armature pose
        # and leave the Armature modifier attached, causing double deformation.
        temp = obj.data.copy()
        eval_obj = None
        temp.calc_loop_triangles()
    try:
        bm = bmesh.new()
        bm.from_mesh(temp)
        source_vertex_layer, source_face_windings = (
            _capture_source_face_windings(bm)
        )
        
        # The 3D-print export historically welds coincident evaluated geometry.
        # General post-processing must not globally weld detached character
        # shells (clothing/body, armor panels, eyes, etc.), because doing so
        # creates branching boundaries and overlapping caps.
        if weld_distance and weld_distance > 0.0:
            bmesh.ops.remove_doubles(
                bm, verts=bm.verts, dist=float(weld_distance)
            )
        
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        
        color_lay = bm.loops.layers.color.get("SG_PrintPreview")
        
        def get_avg_color(loops, color_lay):
            if not color_lay:
                return (0.5, 0.5, 0.5, 1.0)
            r, g, b, a = 0.0, 0.0, 0.0, 0.0
            n = len(loops)
            for loop in loops:
                c = loop[color_lay]
                r += c[0]
                g += c[1]
                b += c[2]
                if len(c) > 3:
                    a += c[3]
                else:
                    a += 1.0
            return (r / n, g / n, b / n, a / n)
            
        t0 = time.perf_counter()
        face_info = []
        for f in bm.faces:
            center = f.calc_center_median()
            avg_color = get_avg_color(f.loops, color_lay)
            face_info.append((center, avg_color, f.index))
        print(f"[StableGen Make Solid] Collected face info in {time.perf_counter() - t0:.4f}s")
            
        if visible_faces is None:
            t0 = time.perf_counter()
            bvh = BVHTree.FromBMesh(bm)
            print(f"[StableGen Make Solid] Built BVH in {time.perf_counter() - t0:.4f}s")
            
            verts_co = np.empty(len(temp.vertices) * 3, dtype=np.float32)
            temp.vertices.foreach_get("co", verts_co)
            verts_co = verts_co.reshape((-1, 3))
            
            xmin, ymin, zmin = verts_co.min(axis=0)
            xmax, ymax, zmax = verts_co.max(axis=0)
            
            cx = (xmin + xmax) * 0.5
            cy = (ymin + ymax) * 0.5
            cz = (zmin + zmax) * 0.5
            
            radius = max(xmax - xmin, ymax - ymin) * 0.5 * 1.25
            z_top = zmax + (zmax - zmin) * 0.25
            z_bottom = zmin
            
            # Precompute cylinder angle offsets (ordered by proximity to 0.0)
            effective_raycast_count = raycast_count
            if effective_raycast_count is None:
                effective_raycast_count = 12
                if hasattr(bpy, "context") and bpy.context and hasattr(bpy.context, "scene") and bpy.context.scene:
                    effective_raycast_count = getattr(
                        bpy.context.scene,
                        "stablegen_print_raycast_count",
                        12,
                    )
            effective_raycast_count = max(4, int(effective_raycast_count))
            
            angle_offsets = [0.0]
            step = 2.0 * math.pi / effective_raycast_count
            i = 1
            while len(angle_offsets) < effective_raycast_count:
                angle_offsets.append(i * step)
                if len(angle_offsets) < effective_raycast_count:
                    angle_offsets.append(-i * step)
                i += 1

            t0 = time.perf_counter()
            visible_faces = set()
            
            for f in bm.faces:
                C = f.calc_center_median()
                is_visible = False
                
                # Check top point first (useful fallback for upper surface concavities)
                P_top = mathutils.Vector((C.x, C.y, z_top))
                direction = C - P_top
                dist = direction.length
                if dist > 1e-6:
                    direction.normalize()
                    hit_loc, hit_normal, hit_idx, hit_dist = bvh.ray_cast(P_top, direction)
                    if hit_loc is not None and hit_idx == f.index:
                        is_visible = True
                
                if not is_visible:
                    dx = C.x - cx
                    dy = C.y - cy
                    dist_xy = math.sqrt(dx*dx + dy*dy)
                    if dist_xy < 1e-6:
                        dx, dy = 1.0, 0.0
                        dist_xy = 1.0
                    nx = dx / dist_xy
                    ny = dy / dist_xy
                    base_angle = math.atan2(ny, nx)
                    
                    heights = [C.z, (C.z + z_bottom) * 0.5, z_bottom, (C.z + z_top) * 0.5, z_top]
                    
                    for h in heights:
                        pz = max(h, z_bottom)
                        for offset in angle_offsets:
                            angle = base_angle + offset
                            px = cx + radius * math.cos(angle)
                            py = cy + radius * math.sin(angle)
                            P = mathutils.Vector((px, py, pz))
                            
                            direction = C - P
                            dist = direction.length
                            if dist < 1e-6:
                                continue
                            direction.normalize()
                            
                            hit_loc, hit_normal, hit_idx, hit_dist = bvh.ray_cast(P, direction)
                            if hit_loc is not None and hit_idx == f.index:
                                is_visible = True
                                break
                        if is_visible:
                            break
                                
                if is_visible:
                    visible_faces.add(f.index)
            print(f"[StableGen Make Solid] Checked face visibility ({len(bm.faces)} faces) in {time.perf_counter() - t0:.4f}s")
                
        t0 = time.perf_counter()
        faces_to_delete = [f for f in bm.faces if f.index not in visible_faces]
        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
        
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        
        loose_edges = [e for e in bm.edges if not e.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')
        loose_verts = [v for v in bm.verts if not v.link_edges]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
            
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        print(f"[StableGen Make Solid] Deleted invisible geometry in {time.perf_counter() - t0:.4f}s")
        
        if keep_largest_island:
            t0 = time.perf_counter()
            bm.faces.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            n_comp, n_rm, area_rm = _keep_largest_island(bm)
            if n_rm > 0:
                bm.faces.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.verts.ensure_lookup_table()
            print(f"[StableGen Make Solid] Kept largest island: {n_comp} components found, "
                  f"removed {n_rm} faces ({area_rm:.4f} area) in {time.perf_counter() - t0:.4f}s")

        # Build centroid → visibility dict for surviving faces (before fill_gaps)
        vis_centers = None
        if face_visibility is not None:
            vis_centers = {}
            for f in bm.faces:
                if f.index < len(face_visibility):
                    key = tuple(round(c, 4) for c in f.calc_center_median())
                    vis_centers[key] = float(face_visibility[f.index])

        if fill_gaps:
            t0 = time.perf_counter()
            boundary_edges = [e for e in bm.edges if e.is_boundary]
            print(f"[StableGen Make Solid] Identified {len(boundary_edges)} boundary edges in {time.perf_counter() - t0:.4f}s")

            # Mark newly created faces explicitly. Centroid comparison is not
            # reliable on dense meshes because unrelated small triangles can
            # quantize to the same rounded center.
            cap_layer = bm.faces.layers.int.get("_SG_SolidCap")
            if cap_layer is None:
                cap_layer = bm.faces.layers.int.new("_SG_SolidCap")
            source_faces = set(bm.faces)

            t0 = time.perf_counter()
            new_faces = []

            if robust_fill and boundary_edges:
                # Fill each genuine boundary cycle independently. The legacy
                # arbitrary transition walk can cross between touching loops
                # and create overlapping faces. A branching component is
                # rejected rather than silently damaging the source object.
                remaining = set(boundary_edges)
                boundary_set = set(boundary_edges)
                components = []
                while remaining:
                    seed = remaining.pop()
                    component = {seed}
                    stack = [seed]
                    while stack:
                        edge = stack.pop()
                        for vert in edge.verts:
                            for linked in vert.link_edges:
                                if linked in remaining and linked in boundary_set:
                                    remaining.remove(linked)
                                    component.add(linked)
                                    stack.append(linked)
                    components.append(component)

                for component in components:
                    component_vertices = {
                        vert for edge in component for vert in edge.verts
                    }
                    branching = [
                        vert for vert in component_vertices
                        if sum(
                            edge in component for edge in vert.link_edges
                        ) != 2
                    ]
                    if branching:
                        raise RuntimeError(
                            "cannot safely cap a branching boundary "
                            f"({len(component)} edges, "
                            f"{len(branching)} branching vertices)"
                        )
                    fill_result = bmesh.ops.triangle_fill(
                        bm,
                        edges=list(component),
                        use_beauty=True,
                        use_dissolve=False,
                    )
                    component_faces = [
                        geom for geom in fill_result.get("geom", ())
                        if isinstance(geom, bmesh.types.BMFace)
                        and geom not in source_faces
                    ]
                    if not component_faces:
                        raise RuntimeError(
                            "failed to create a cap for a valid boundary loop"
                        )
                    new_faces.extend(component_faces)
            else:
                # Preserve the historical 3D-print fill behavior. The general
                # post-processing operator uses the validated triangle-fill
                # path above.
                directed_transitions = []
                for edge in boundary_edges:
                    face = edge.link_faces[0]
                    target_loop = next(
                        (loop for loop in face.loops if loop.edge == edge),
                        None,
                    )
                    if target_loop:
                        directed_transitions.append((
                            target_loop.link_loop_next.vert,
                            target_loop.vert,
                            edge,
                        ))

                from collections import defaultdict
                vert_to_transitions = defaultdict(list)
                for transition in directed_transitions:
                    vert_to_transitions[transition[0]].append(transition)

                directed_edges_set = set(directed_transitions)
                while directed_edges_set:
                    start_trans = directed_edges_set.pop()
                    loop_verts = [start_trans[0]]
                    visited_verts_set = {start_trans[0]}
                    vert_current = start_trans[1]

                    while True:
                        if vert_current in visited_verts_set:
                            index = loop_verts.index(vert_current)
                            cycle = loop_verts[index:]
                            if len(cycle) >= 3:
                                try:
                                    new_faces.append(bm.faces.new(cycle))
                                except Exception:
                                    pass
                            loop_verts = loop_verts[:index]
                            visited_verts_set = set(loop_verts)

                        loop_verts.append(vert_current)
                        visited_verts_set.add(vert_current)
                        next_trans = next(
                            (
                                transition
                                for transition in vert_to_transitions[
                                    vert_current
                                ]
                                if transition in directed_edges_set
                            ),
                            None,
                        )
                        if next_trans is None:
                            if len(loop_verts) >= 3:
                                try:
                                    new_faces.append(
                                        bm.faces.new(loop_verts)
                                    )
                                except Exception:
                                    pass
                            break
                        directed_edges_set.remove(next_trans)
                        vert_current = next_trans[1]

            print(f"[StableGen Make Solid] Created {len(new_faces)} cap faces in {time.perf_counter() - t0:.4f}s")

            # Use a temporary sentinel so triangulation can be identified
            # without depending on BMesh face identity surviving the operation.
            for face in new_faces:
                face[cap_layer] = 2

            t0 = time.perf_counter()
            if new_faces and not robust_fill:
                try:
                    bmesh.ops.triangulate(
                        bm,
                        faces=new_faces,
                        quad_method='BEAUTY',
                        ngon_method='BEAUTY',
                    )
                except Exception as error:
                    print(
                        "[StableGen Make Solid] Warning: "
                        f"BMesh triangulation failed: {error}"
                    )
            final_new_faces = [
                face for face in bm.faces if face[cap_layer] == 2
            ]
            for face in final_new_faces:
                face[cap_layer] = 1
            print(f"[StableGen Make Solid] Finalized {len(final_new_faces)} cap triangles in {time.perf_counter() - t0:.4f}s")

            # Inherit the surrounding face material instead of accepting
            # BMesh's default material slot 0. Direct boundary-edge votes are
            # preferred; vertex adjacency covers interior fill triangles.
            for cap_face in final_new_faces:
                material_votes = {}
                for edge in cap_face.edges:
                    for adjacent in edge.link_faces:
                        if adjacent in source_faces and adjacent.is_valid:
                            material = adjacent.material_index
                            material_votes[material] = (
                                material_votes.get(material, 0.0)
                                + max(edge.calc_length(), 1.0e-8) * 4.0
                            )
                if not material_votes:
                    for vert in cap_face.verts:
                        for adjacent in vert.link_faces:
                            if adjacent in source_faces and adjacent.is_valid:
                                material = adjacent.material_index
                                material_votes[material] = (
                                    material_votes.get(material, 0.0) + 1.0
                                )
                if material_votes:
                    cap_face.material_index = max(
                        material_votes,
                        key=material_votes.get,
                    )
                cap_face.smooth = bool(smooth_caps)

            t0 = time.perf_counter()

            # New cap loops otherwise receive zero/default custom-data values.
            # Copy UV, color, and generic corner attributes from the nearest
            # surviving loop on the same boundary vertex. This keeps existing
            # ProjectionUV attributes usable when solidifying after texturing.
            new_face_set = set(final_new_faces)
            layer_kinds = (
                "uv", "color", "float_color", "float",
                "float_vector", "int", "string",
            )
            for nf in final_new_faces:
                for dst_loop in nf.loops:
                    src_loop = next(
                        (
                            loop for loop in dst_loop.vert.link_loops
                            if loop.face not in new_face_set
                        ),
                        None,
                    )
                    if src_loop is None:
                        continue
                    for kind in layer_kinds:
                        layers = getattr(bm.loops.layers, kind, None)
                        if layers is None:
                            continue
                        for layer in layers.values():
                            try:
                                if kind == "uv":
                                    dst_loop[layer].uv = src_loop[layer].uv
                                else:
                                    dst_loop[layer] = src_loop[layer]
                            except (AttributeError, TypeError, ValueError):
                                # Not every Blender version exposes every
                                # generic layer through an assignable BMesh API.
                                pass
                    
            if color_lay and final_new_faces:
                valid_face_info = [info for info in face_info if info[2] in visible_faces]
                if valid_face_info:
                    for nf in final_new_faces:
                        nf_center = nf.calc_center_median()
                        best_dist = float('inf')
                        best_color = (0.5, 0.5, 0.5, 1.0)
                        for orig_center, orig_color, _ in valid_face_info:
                            d = (orig_center - nf_center).length_squared
                            if d < best_dist:
                                best_dist = d
                                best_color = orig_color
                                
                        for loop in nf.loops:
                            loop[color_lay] = best_color
            print(f"[StableGen Make Solid] Mapped colors in {time.perf_counter() - t0:.4f}s")
                            
        matching_faces, flipped_faces = _restore_matching_source_face_windings(
            bm,
            source_vertex_layer,
            source_face_windings,
        )
        bm.verts.layers.int.remove(source_vertex_layer)
        if flipped_faces:
            print(
                "[StableGen Make Solid] Restored source winding on "
                f"{flipped_faces}/{matching_faces} matching faces"
            )

        # Update normals in BMesh
        try:
            bm.normal_update()
        except Exception as e:
            print(f"[StableGen Make Solid] Warning: could not update normals: {e}")
        
        solid_mesh = bpy.data.meshes.new(f"{obj.name}_solid")
        bm.to_mesh(solid_mesh)
        bm.free()

        # BMesh exposes face booleans through its integer custom-data layer.
        # Convert the internal 0/1 marker to a proper mesh Boolean attribute so
        # Geometry Nodes, shaders, diagnostics, and future post-processing can
        # consume it without guessing the storage convention.
        cap_attribute = solid_mesh.attributes.get("_SG_SolidCap")
        if (
            cap_attribute is not None
            and cap_attribute.data_type != 'BOOLEAN'
        ):
            cap_values = [
                bool(item.value) for item in cap_attribute.data
            ]
            solid_mesh.attributes.remove(cap_attribute)
            cap_attribute = solid_mesh.attributes.new(
                name="_SG_SolidCap",
                type='BOOLEAN',
                domain='FACE',
            )
            cap_attribute.data.foreach_set("value", cap_values)

        if vis_centers is not None:
            import numpy as np
            n_faces = len(solid_mesh.polygons)
            vis_vals = np.zeros(n_faces, dtype=np.float32)
            for i in range(n_faces):
                c = solid_mesh.polygons[i].center
                key = (round(c.x, 4), round(c.y, 4), round(c.z, 4))
                vis_vals[i] = vis_centers.get(key, 0.0)
            attr = solid_mesh.attributes.new(name="_SG_FaceVisibility", type='FLOAT', domain='FACE')
            attr.data.foreach_set("value", vis_vals.tolist())
            print(f"[StableGen Make Solid] Stored _SG_FaceVisibility on {n_faces} faces")

        solid_mesh.calc_loop_triangles()
        try:
            _require_printable_mesh(
                solid_mesh,
                label=f"Make Solid result for {obj.name}",
                require_closed=bool(fill_gaps),
            )
        except Exception:
            bpy.data.meshes.remove(solid_mesh)
            raise
        print(f"[StableGen Make Solid] Total execution time: {time.perf_counter() - t_start:.4f}s")
        return solid_mesh
    finally:
        if eval_obj is not None:
            eval_obj.to_mesh_clear()
        elif temp and temp.name in bpy.data.meshes:
            bpy.data.meshes.remove(temp)


def _extract_mesh_data(obj, apply_transforms, palette_rgbs, image):
    """Extract triangulated mesh data with per-triangle extruder IDs.

    Returns ``(verts, tris, paint_colors, palette)`` where:
      *verts*        — NumPy float32 array (V, 3)
      *tris*         — NumPy int32 array (T, 3)
      *paint_colors* — NumPy int32 array (T,) (1-based, 0 = no assignment)
      *palette*      — list of ``(hex_str, (r,g,b))``
    """
    import numpy as np
    import bpy

    scene = bpy.context.scene
    make_solid = getattr(scene, "stablegen_print_make_solid", False)
    chroma_threshold = getattr(scene, "stablegen_print_chroma_threshold", 0.35)

    if make_solid:
        if not obj.data.color_attributes.get("SG_PrintPreview"):
            raise ValueError(f"Make Solid requires the 'SG_PrintPreview' color attribute. Run 'Preview Colors' first on {obj.name}.")
        temp = _make_solid_mesh_object(obj)
        eval_obj = None
    else:
        temp, eval_obj = _get_triangulated_mesh(obj, apply_transforms)

    try:
        _require_printable_mesh(
            temp,
            label=obj.name,
            require_closed=True,
        )
        n_verts = len(temp.vertices)
        verts = np.empty(n_verts * 3, dtype=np.float32)
        temp.vertices.foreach_get("co", verts)
        verts = verts.reshape((-1, 3))

        color_layer = temp.color_attributes.get("SG_PrintPreview")

        if palette_rgbs and (color_layer or image):
            if image and not make_solid:
                face_colors = _collect_face_colors(temp, image)
            elif color_layer:
                n_loops = len(temp.loops)
                loop_colors = np.empty(n_loops * 4, dtype=np.float32)
                color_layer.data.foreach_get("color", loop_colors)
                loop_colors = loop_colors.reshape((-1, 4))
                
                n_tris = len(temp.loop_triangles)
                tri_loops = np.empty(n_tris * 3, dtype=np.int32)
                temp.loop_triangles.foreach_get("loops", tri_loops)
                tri_loops = tri_loops.reshape((-1, 3))
                
                face_colors = loop_colors[tri_loops].mean(axis=1)
                if color_layer.data_type == 'BYTE_COLOR':
                    face_colors[:, :3] = srgb_to_linear_numpy(face_colors[:, :3])
            else:
                face_colors = _collect_face_colors(temp, image)
            
            face_colors_rgb = face_colors[:, :3]
            face_colors_rgb_srgb = linear_to_srgb_numpy(face_colors_rgb)
            material_indices = _match_colors_hsv(face_colors_rgb_srgb, palette_rgbs, chroma_threshold=chroma_threshold)
            
            smoothing_passes = getattr(scene, "stablegen_print_smoothing", 2)
            if smoothing_passes > 0:
                n_tris = len(temp.loop_triangles)
                tri_verts = np.empty(n_tris * 3, dtype=np.int32)
                temp.loop_triangles.foreach_get("vertices", tri_verts)
                tri_verts = tri_verts.reshape((-1, 3))
                
                material_indices = _smooth_colors_vectorized(
                    tri_verts, material_indices + 1, len(temp.vertices), len(palette_rgbs), iterations=smoothing_passes
                ) - 1
                
            centroids = palette_rgbs
        else:
            material_indices = np.zeros(len(temp.loop_triangles), dtype=np.int32)
            centroids = [(0.7, 0.7, 0.7)]

        palette = []
        hex_lookup = {}
        for col in centroids:
            hex_c = _rgb_to_hex(col)
            if hex_c not in hex_lookup:
                hex_lookup[hex_c] = len(palette)
                palette.append((hex_c, col))

        map_arr = np.zeros(len(centroids), dtype=np.int32)
        for ci, col in enumerate(centroids):
            hex_c = _rgb_to_hex(col)
            map_arr[ci] = hex_lookup.get(hex_c, 0) + 1

        paint_colors = map_arr[material_indices]

        n_tris = len(temp.loop_triangles)
        tris = np.empty(n_tris * 3, dtype=np.int32)
        temp.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape((-1, 3))
    finally:
        if eval_obj is not None:
            eval_obj.to_mesh_clear()
        else:
            bpy.data.meshes.remove(temp)

    return verts, tris, paint_colors, palette


# ── Operator: selection / state helpers ──────────────────────────────────

class _SelectionSnapshot:
    def __init__(self, context):
        self.selected = list(context.selected_objects)
        self.active = context.view_layer.objects.active

    def restore(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        for o in self.selected:
            try:
                o.select_set(True)
            except ReferenceError:
                pass
        if self.active:
            try:
                context.view_layer.objects.active = self.active
            except ReferenceError:
                pass


def _get_mesh_objects(context, scope):
    if scope == 'ALL_MESH':
        return [o for o in context.view_layer.objects
                if o.type == 'MESH' and not o.hide_get()]
    return [o for o in context.selected_objects if o.type == 'MESH']


def _poll_common(context):
    addon_prefs = get_addon_prefs(context)
    if not addon_prefs or not os.path.exists(addon_prefs.output_dir):
        return "Output directory not set or does not exist (check addon preferences)"
    if sg_modal_active(context):
        return "Another operation is in progress"
    return None


# ── Textured OBJ for image-map slicers ──────────────────────────────────

def _safe_export_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value or "mesh"


def _base_color_texture_binding(obj):
    """Return ``(image, uv_name)`` for the best baked colour texture."""
    if not obj.data.materials:
        return None, None

    fallback = None
    for material in reversed(obj.data.materials):
        if not material or not material.use_nodes:
            continue
        nodes = material.node_tree.nodes
        candidates = []
        for node in nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            base_color = node.inputs.get("Base Color")
            if base_color and base_color.links:
                image_node = base_color.links[0].from_node
                if image_node.type == 'TEX_IMAGE' and image_node.image:
                    candidates.append(image_node)
        candidates.extend(
            node for node in nodes
            if node.type == 'TEX_IMAGE' and node.image
            and node not in candidates
        )

        for image_node in candidates:
            uv_name = None
            vector = image_node.inputs.get("Vector")
            if vector and vector.links:
                vector_node = vector.links[0].from_node
                if vector_node.type == 'UVMAP':
                    uv_name = vector_node.uv_map or None
            binding = (image_node.image, uv_name)
            if image_node in candidates[:1]:
                return binding
            if fallback is None:
                fallback = binding
    return fallback or (None, None)


def _textured_print_uv_layer(mesh, requested_name=None):
    if requested_name and mesh.uv_layers.get(requested_name):
        return mesh.uv_layers.get(requested_name)
    if mesh.uv_layers.get("BakeUV"):
        return mesh.uv_layers.get("BakeUV")
    render_layer = next(
        (
            layer for layer in mesh.uv_layers
            if getattr(layer, "active_render", False)
        ),
        None,
    )
    return render_layer or mesh.uv_layers.active


def _collect_textured_print_record(
        obj, *, apply_transforms=True, make_solid=False,
        raycast_count=12, keep_largest_island=False):
    """Extract shared positions and per-corner UVs for a slicer-safe OBJ."""
    import numpy as np

    image, requested_uv = _base_color_texture_binding(obj)
    if image is None:
        raise RuntimeError(
            f"{obj.name} has no baked Base Color image. Run Bake Textures "
            "before textured print export."
        )

    eval_obj = None
    owns_mesh = False
    if make_solid:
        mesh = _make_solid_mesh_object(
            obj,
            fill_gaps=True,
            keep_largest_island=keep_largest_island,
            use_evaluated_mesh=True,
            raycast_count=raycast_count,
            weld_distance=0.0,
            robust_fill=True,
            smooth_caps=True,
        )
        owns_mesh = True
        if apply_transforms:
            mesh.transform(obj.matrix_world)
    else:
        mesh, eval_obj = _get_triangulated_mesh(obj, apply_transforms)

    try:
        mesh.calc_loop_triangles()
        _require_printable_mesh(mesh, label=obj.name, require_closed=True)
        uv_layer = _textured_print_uv_layer(mesh, requested_uv)
        if uv_layer is None:
            raise RuntimeError(
                f"{obj.name} has no UV map for its baked texture."
            )

        vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", vertices)
        vertices = vertices.reshape((-1, 3))

        triangle_vertices = np.empty(
            len(mesh.loop_triangles) * 3,
            dtype=np.int64,
        )
        mesh.loop_triangles.foreach_get("vertices", triangle_vertices)
        triangle_vertices = triangle_vertices.reshape((-1, 3))

        triangle_loops = np.empty(
            len(mesh.loop_triangles) * 3,
            dtype=np.int64,
        )
        mesh.loop_triangles.foreach_get("loops", triangle_loops)
        triangle_loops = triangle_loops.reshape((-1, 3))

        loop_uvs = np.empty(len(uv_layer.data) * 2, dtype=np.float64)
        uv_layer.data.foreach_get("uv", loop_uvs)
        loop_uvs = loop_uvs.reshape((-1, 2))

        return {
            "name": obj.name,
            "vertices": vertices,
            "triangle_vertices": triangle_vertices,
            "triangle_uvs": loop_uvs[triangle_loops],
            "image": image,
            "uv_name": uv_layer.name,
        }
    finally:
        if eval_obj is not None:
            eval_obj.to_mesh_clear()
        elif owns_mesh and mesh.name in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)


def _copy_print_texture(image, output_dir, stem):
    """Copy or losslessly save *image* beside the OBJ and return its name."""
    source_path = ""
    try:
        source_path = bpy.path.abspath(image.filepath_from_user())
    except Exception:
        source_path = bpy.path.abspath(image.filepath or "")

    can_copy = (
        image.source == 'FILE'
        and not image.packed_file
        and not image.is_dirty
        and source_path
        and os.path.isfile(source_path)
    )
    if can_copy:
        extension = os.path.splitext(source_path)[1].lower() or ".png"
        texture_name = f"{stem}_texture{extension}"
        destination = os.path.join(output_dir, texture_name)
        if os.path.normcase(os.path.abspath(source_path)) != os.path.normcase(
                os.path.abspath(destination)):
            shutil.copy2(source_path, destination)
        return texture_name

    texture_name = f"{stem}_texture.png"
    destination = os.path.join(output_dir, texture_name)
    image.save_render(destination, scene=bpy.context.scene)
    if not os.path.isfile(destination):
        raise RuntimeError(f"Could not save baked texture {image.name}")
    return texture_name


def _write_textured_print_obj(filepath, records):
    """Write OBJ with shared geometry indices and independent UV indices.

    GLTF must split vertices at UV seams. OBJ has independent ``v`` and ``vt``
    indices, which lets image-map slicers see one connected geometric shell.
    """
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)
    stem = _safe_export_name(os.path.splitext(os.path.basename(filepath))[0])
    mtl_name = f"{stem}.mtl"
    mtl_path = os.path.join(output_dir, mtl_name)

    material_records = []
    for index, record in enumerate(records):
        item_stem = f"{stem}_{index + 1}"
        texture_name = _copy_print_texture(
            record["image"], output_dir, item_stem)
        material_records.append((f"SG_Print_{index + 1}", texture_name))

    with open(mtl_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("# StableGen textured print material\n")
        for material_name, texture_name in material_records:
            stream.write(f"newmtl {material_name}\n")
            stream.write("Ka 1.000000 1.000000 1.000000\n")
            stream.write("Kd 1.000000 1.000000 1.000000\n")
            stream.write("d 1.000000\nillum 1\n")
            stream.write(f"map_Kd {texture_name}\n\n")

    vertex_offset = 0
    uv_offset = 0
    total_triangles = 0
    with open(filepath, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("# StableGen textured print OBJ\n")
        stream.write(f"mtllib {mtl_name}\n")
        for record, material_record in zip(records, material_records):
            stream.write(f"o {_safe_export_name(record['name'])}\n")
            for x, y, z in record["vertices"]:
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")

            # Deliberately keep one texture coordinate per triangle corner.
            # Geometry still references the original shared position indices.
            for triangle_uvs in record["triangle_uvs"]:
                for u, v in triangle_uvs:
                    stream.write(f"vt {u:.9g} {v:.9g}\n")

            stream.write(f"usemtl {material_record[0]}\n")
            for triangle_index, triangle in enumerate(
                    record["triangle_vertices"]):
                vertex_indices = triangle + vertex_offset + 1
                first_uv = uv_offset + triangle_index * 3 + 1
                stream.write(
                    "f "
                    f"{vertex_indices[0]}/{first_uv} "
                    f"{vertex_indices[1]}/{first_uv + 1} "
                    f"{vertex_indices[2]}/{first_uv + 2}\n"
                )

            vertex_offset += len(record["vertices"])
            uv_offset += len(record["triangle_uvs"]) * 3
            total_triangles += len(record["triangle_vertices"])

    return {
        "filepath": filepath,
        "mtl_path": mtl_path,
        "vertices": vertex_offset,
        "triangles": total_triangles,
    }


class ExportTexturedPrintOBJ(bpy.types.Operator):
    """Export a watertight UV-textured OBJ for image-map slicers."""
    bl_idname = "object.export_textured_print_obj"
    bl_label = "Export Textured OBJ"
    bl_description = (
        "Export baked texture and shared-position OBJ for image-map slicers; "
        "refuses open or non-manifold geometry"
    )
    bl_options = {'REGISTER'}

    export_scope: bpy.props.EnumProperty(
        name="Objects",
        items=[
            ('SELECTED', 'Selected', 'Export selected mesh objects'),
            ('ALL_MESH', 'All Mesh', 'Export all visible mesh objects'),
        ],
        default='SELECTED',
    )
    apply_transforms: bpy.props.BoolProperty(
        name="Apply Transforms",
        description=(
            "Write world-space position, rotation and scale without changing "
            "the Blender objects"
        ),
        default=True,
    )

    @classmethod
    def poll(cls, context):
        message = _poll_common(context)
        if message:
            cls.poll_message_set(message)
            return False
        return True

    def draw(self, context):
        self.layout.prop(self, "export_scope")
        self.layout.prop(self, "apply_transforms")
        if getattr(context.scene, "stablegen_print_make_solid", False):
            self.layout.label(
                text="Make Solid will run on a temporary export copy",
                icon='MOD_REMESH',
            )
        else:
            self.layout.label(
                text="Existing mesh must already be watertight",
                icon='INFO',
            )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        meshes = _get_mesh_objects(context, self.export_scope)
        if not meshes:
            self.report({'ERROR'}, "No mesh objects to export.")
            return {'CANCELLED'}

        scene = context.scene
        make_solid = getattr(scene, "stablegen_print_make_solid", False)
        records = []
        try:
            for obj in meshes:
                records.append(_collect_textured_print_record(
                    obj,
                    apply_transforms=self.apply_transforms,
                    make_solid=make_solid,
                    raycast_count=getattr(
                        scene, "stablegen_print_raycast_count", 12),
                    keep_largest_island=getattr(
                        scene,
                        "stablegen_print_keep_largest_island",
                        False,
                    ),
                ))

            output_dir = get_dir_path(context, "baked")
            base_name = _safe_export_name(meshes[0].name)
            filepath = os.path.join(
                output_dir, f"{base_name}_textured_print.obj")
            result = _write_textured_print_obj(filepath, records)
        except Exception as error:
            self.report({'ERROR'}, f"Textured print export failed: {error}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"Textured print OBJ exported to {result['filepath']} "
            f"({result['triangles']} triangles)",
        )
        return {'FINISHED'}


# ── ExportSTL ────────────────────────────────────────────────────────────

class ExportSTL(bpy.types.Operator):
    """Export selected mesh objects as STL for 3D printing."""
    bl_idname = "object.export_stl"
    bl_label = "Export STL"
    bl_options = {'REGISTER'}

    export_scope: bpy.props.EnumProperty(
        name="Objects",
        items=[
            ('SELECTED', 'Selected', 'Export selected mesh objects'),
            ('ALL_MESH', 'All Mesh', 'Export all visible mesh objects'),
        ],
        default='SELECTED',
    )
    apply_transforms: bpy.props.BoolProperty(
        name="Apply Transforms",
        description="Apply location, rotation and scale before export",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        msg = _poll_common(context)
        if msg:
            cls.poll_message_set(msg)
            return False
        return True

    def draw(self, context):
        self.layout.prop(self, "export_scope")
        self.layout.prop(self, "apply_transforms")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        meshes = _get_mesh_objects(context, self.export_scope)
        if not meshes:
            self.report({'ERROR'}, "No mesh objects to export.")
            return {'CANCELLED'}

        # Verify SG_PrintPreview if Make Solid is enabled
        scene = context.scene
        if getattr(scene, "stablegen_print_make_solid", False):
            for o in meshes:
                if not o.data.color_attributes.get("SG_PrintPreview"):
                    self.report({'ERROR'}, f"Make Solid requires the 'SG_PrintPreview' color attribute. Run 'Preview Colors' first on {o.name}.")
                    return {'CANCELLED'}
        else:
            try:
                for obj in meshes:
                    mesh, eval_obj = _get_triangulated_mesh(
                        obj, apply_transforms=False)
                    try:
                        _require_printable_mesh(
                            mesh,
                            label=obj.name,
                            require_closed=True,
                        )
                    finally:
                        eval_obj.to_mesh_clear()
            except Exception as error:
                self.report({'ERROR'}, f"STL export stopped: {error}")
                return {'CANCELLED'}

        snap = _SelectionSnapshot(context)
        output_dir = get_dir_path(context, "baked")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{meshes[0].name}.stl")

        bpy.ops.object.select_all(action='DESELECT')
        for o in meshes:
            o.select_set(True)
        context.view_layer.objects.active = meshes[0]

        if self.apply_transforms:
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True)

        solidified_mappings = {}
        try:
            if getattr(scene, "stablegen_print_make_solid", False):
                for o in meshes:
                    solid_mesh = _make_solid_mesh_object(o)
                    solidified_mappings[o] = (o.data, solid_mesh)
                    o.data = solid_mesh

            if hasattr(bpy.ops.wm, 'stl_export'):
                bpy.ops.wm.stl_export(
                    filepath=filepath,
                    global_scale=1.0,
                )
            else:
                bpy.ops.export_mesh.stl(
                    filepath=filepath,
                    use_selection=True,
                    ascii=False,
                    global_scale=1.0,
                )
            self.report({'INFO'}, f"STL exported to {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"STL export failed: {e}")
            import traceback
            traceback.print_exc()
            snap.restore(context)
            return {'CANCELLED'}
        finally:
            for o, (orig_mesh, solid_mesh) in solidified_mappings.items():
                o.data = orig_mesh
                bpy.data.meshes.remove(solid_mesh)

        snap.restore(context)
        return {'FINISHED'}


# ── Asynchronous Exporter Workers ───────────────────────────────────────

def _prepare_mesh_for_raycasting(obj):
    import bmesh
    import numpy as np
    temp, eval_obj = _get_triangulated_mesh(obj, apply_transforms=False)
    bm = bmesh.new()
    bm.from_mesh(temp)
    # Keep vertex and face indices identical to the later Make Solid pass.
    # A tolerance weld here used to create branching boundaries and also made
    # the worker's visible-face indices refer to different geometry.
    
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    coords = [v.co.copy() for v in bm.verts]
    faces = [[v.index for v in f.verts] for f in bm.faces]
    
    verts_co = np.array(coords, dtype=np.float32)
    xmin, ymin, zmin = verts_co.min(axis=0)
    xmax, ymax, zmax = verts_co.max(axis=0)
    
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    
    radius = max(xmax - xmin, ymax - ymin) * 0.5 * 1.25
    z_top = zmax + (zmax - zmin) * 0.25
    z_bottom = zmin
    
    bm.free()
    eval_obj.to_mesh_clear()
    
    return {
        "coords": coords,
        "faces": faces,
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "z_top": z_top,
        "z_bottom": z_bottom
    }


def _raycast_visibility_worker(mesh_tasks, angle_offsets, status_dict):
    import time
    import math
    import mathutils
    from mathutils.bvhtree import BVHTree
    import numpy as np

    try:
        results = []
        n_tasks = len(mesh_tasks)
        for task_idx, task in enumerate(mesh_tasks):
            if status_dict.get("cancel_requested", False):
                return
            
            coords = task["coords"]
            faces = task["faces"]
            cx = task["cx"]
            cy = task["cy"]
            radius = task["radius"]
            z_top = task["z_top"]
            z_bottom = task["z_bottom"]
            
            status_dict["stage"] = f"Building BVH tree ({task_idx + 1}/{n_tasks})"
            status_dict["progress"] = (task_idx / n_tasks) * 100.0 + 5.0 / n_tasks
            
            bvh = BVHTree.FromPolygons(coords, faces)
            
            status_dict["stage"] = f"Raycasting visibility check ({task_idx + 1}/{n_tasks})"
            
            visible_faces = set()
            n_faces = len(faces)
            progress_step = max(1, n_faces // 50)
            
            for face_idx, f_verts in enumerate(faces):
                if status_dict.get("cancel_requested", False):
                    return
                
                if face_idx % progress_step == 0:
                    pct = (task_idx / n_tasks) * 100.0 + ((10.0 + (face_idx / n_faces) * 80.0) / n_tasks)
                    status_dict["progress"] = pct
                
                v1, v2, v3 = coords[f_verts[0]], coords[f_verts[1]], coords[f_verts[2]]
                cx_f = (v1[0] + v2[0] + v3[0]) / 3.0
                cy_f = (v1[1] + v2[1] + v3[1]) / 3.0
                cz_f = (v1[2] + v2[2] + v3[2]) / 3.0
                C = mathutils.Vector((cx_f, cy_f, cz_f))
                
                is_visible = False
                
                P_top = mathutils.Vector((cx_f, cy_f, z_top))
                direction = C - P_top
                dist = direction.length
                if dist > 1e-6:
                    direction.normalize()
                    hit_loc, hit_normal, hit_idx, hit_dist = bvh.ray_cast(P_top, direction)
                    if hit_loc is not None and hit_idx == face_idx:
                        is_visible = True
                        
                if not is_visible:
                    dx = cx_f - cx
                    dy = cy_f - cy
                    dist_xy = math.sqrt(dx*dx + dy*dy)
                    if dist_xy < 1e-6:
                        dx, dy = 1.0, 0.0
                        dist_xy = 1.0
                    nx = dx / dist_xy
                    ny = dy / dist_xy
                    base_angle = math.atan2(ny, nx)
                    
                    heights = [cz_f, (cz_f + z_bottom) * 0.5, z_bottom, (cz_f + z_top) * 0.5, z_top]
                    for h in heights:
                        pz = max(h, z_bottom)
                        for offset in angle_offsets:
                            angle = base_angle + offset
                            px = cx + radius * math.cos(angle)
                            py = cy + radius * math.sin(angle)
                            P = mathutils.Vector((px, py, pz))
                            
                            direction = C - P
                            dist = direction.length
                            if dist < 1e-6:
                                continue
                            direction.normalize()
                            
                            hit_loc, hit_normal, hit_idx, hit_dist = bvh.ray_cast(P, direction)
                            if hit_loc is not None and hit_idx == face_idx:
                                is_visible = True
                                break
                        if is_visible:
                            break
                            
                if is_visible:
                    visible_faces.add(face_idx)
            
            results.append(visible_faces)
            
        status_dict["visible_faces_results"] = results
        status_dict["progress"] = 100.0
        status_dict["stage"] = "Visibility check completed"
        status_dict["done"] = True
    except Exception as e:
        status_dict["error"] = e
        status_dict["done"] = True


def _export_3mf_worker(is_dithered, filepath, palette_rgbs, solid_data, dithered_data, status_dict):
    import time
    import math
    import json
    import numpy as np
    import xml.etree.ElementTree as ET

    try:
        if not is_dithered:
            status_dict["stage"] = "Building model XML"
            status_dict["progress"] = 10.0
            
            all_verts_arr = solid_data["all_verts_arr"]
            all_tris_arr = solid_data["all_tris_arr"]
            all_paint_arr = solid_data["all_paint_arr"]
            global_palette = solid_data["global_palette"]
            global_to_extruder = solid_data["global_to_extruder"]
            parent_id = solid_data["parent_id"]
            child_ids_and_extruders = solid_data["child_ids_and_extruders"]
            project_settings_json = solid_data["project_settings_json"]
            prusa_config = solid_data["prusa_config"]
            prusa_model_config = solid_data["prusa_model_config"]

            model, parent_id, child_ids_and_extruders = _build_model_element(
                all_verts_arr, all_tris_arr, all_paint_arr, global_palette,
                global_to_extruder=global_to_extruder
            )
            
            if status_dict.get("cancel_requested", False):
                return
                
            status_dict["stage"] = "Writing 3MF package"
            status_dict["progress"] = 50.0
            model_settings = _build_model_settings_element(parent_id, child_ids_and_extruders)
            
            _write_3mf_archive(filepath, model, model_settings, project_settings_json, prusa_config, prusa_model_config)
            
            status_dict["progress"] = 100.0
            status_dict["stage"] = "Completed"
            status_dict["done"] = True
            status_dict["total_triangles"] = len(all_tris_arr)
            return

        # DITHERED MODE
        all_raw_verts = dithered_data["all_raw_verts"]
        all_raw_tris = dithered_data["all_raw_tris"]
        all_raw_face_colors = dithered_data["all_raw_face_colors"]
        h_target = dithered_data["h_target"]
        LH = dithered_data["LH"]
        n_layers = dithered_data["n_layers"]
        unique_keys = dithered_data["unique_keys"]
        unique_colors_arr = dithered_data["unique_colors_arr"]
        mode = dithered_data["mode"]
        channel_indices = dithered_data["channel_indices"]
        channels_order = dithered_data["channels_order"]
        mesh_name = dithered_data["mesh_name"]
        cyan_idx = dithered_data["cyan_idx"]
        dither_method = dithered_data.get("dither_method", "Z_SEQUENCE")
        single_filament_internal = dithered_data.get("single_filament_internal", False)
        internal_visibility_threshold = dithered_data.get("internal_visibility_threshold", 0.01)
        face_visibility = dithered_data.get("face_visibility", None)

        status_dict["stage"] = "Scaling geometry"
        status_dict["progress"] = 5.0

        z_min, z_max = all_raw_verts[:, 2].min(), all_raw_verts[:, 2].max()
        h_blender = z_max - z_min
        if h_blender < 1e-5:
            h_blender = 1.0
        scale_factor = h_target / h_blender

        scaled_verts = all_raw_verts.copy()
        scaled_verts[:, 0] *= scale_factor
        scaled_verts[:, 1] *= scale_factor
        scaled_verts[:, 2] = (scaled_verts[:, 2] - z_min) * scale_factor

        # Solve filament demands
        status_dict["stage"] = "Calculating filament demands"
        status_dict["progress"] = 15.0

        n_channels = len(channel_indices)
        if mode == 'solid':
            fil_rgbs = np.array(palette_rgbs, dtype=np.float32)
            solver_settings = dithered_data.get("solver_settings", {})
            iters = solver_settings.get("iterations", 80)
            init_m = solver_settings.get("init_method", 'CLOSEST')
            formula = solver_settings.get("model_formula", 'KUBELKA_MUNK')
            s_weight = solver_settings.get("scat_weight", 10.0)
            p_weight = solver_settings.get("penalty_weight", 1.0)
            
            demands = solve_color_mixing_numpy(
                1.0 - fil_rgbs, 1.0 - unique_colors_arr,
                iterations=iters,
                init_method=init_m,
                model_formula=formula,
                scat_weight=s_weight,
                penalty_weight=p_weight
            )
        else:
            demands = np.empty((len(unique_colors_arr), n_channels), dtype=np.float32)
            for idx, color in enumerate(unique_colors_arr):
                r, g, b = round(color[0] * 255.0), round(color[1] * 255.0), round(color[2] * 255.0)
                demands[idx] = decompose_subtractive(r, g, b, mode, palette_rgbs, channel_indices)

        unique_colors_set = [tuple(round(c * 255.0) for c in color) for color in unique_colors_arr]

        if dither_method == 'BLUE_NOISE':
            status_dict["stage"] = "Building blue noise dither data"
            status_dict["progress"] = 25.0

            blue_noise_tile = _generate_blue_noise_tile(128)
            bn_size = blue_noise_tile.shape[0]
            bn_scale = 4.0
            _off_rng = np.random.RandomState(98765)
            layer_ox = _off_rng.randint(0, bn_size, size=n_layers).tolist()
            layer_oy = _off_rng.randint(0, bn_size, size=n_layers).tolist()

            color_cumdemands = {}
            for idx, color_key in enumerate(unique_colors_set):
                if status_dict.get("cancel_requested", False):
                    return
                row = demands[idx]
                total = sum(max(0.0, d) for d in row)
                if total < 0.001:
                    color_cumdemands[color_key] = None
                else:
                    cum = []
                    s = 0.0
                    for d in row:
                        s += max(0.0, d) / total
                        cum.append(s)
                    cum[-1] = 1.0
                    color_cumdemands[color_key] = cum
        else:
            status_dict["stage"] = "Building dither sequences"
            status_dict["progress"] = 25.0

            color_seqs = {}
            for idx, color_key in enumerate(unique_colors_set):
                if status_dict.get("cancel_requested", False):
                    return
                row = demands[idx]
                total = sum(max(0.0, d) for d in row)
                seq = [0] * n_layers

                if total < 0.001:
                    if cyan_idx >= 0:
                        nc = min(3, n_channels - cyan_idx)
                        for li in range(n_layers):
                            seq[li] = cyan_idx + (li % nc)
                    else:
                        for li in range(n_layers):
                            seq[li] = li % n_channels
                else:
                    frac = [max(0.0, d) / total for d in row]
                    placed = [0] * n_channels
                    for li in range(n_layers):
                        best_ch = 0
                        best_d = -float('inf')
                        for ch in range(n_channels):
                            if frac[ch] < 0.001:
                                continue
                            d = frac[ch] * (li + 1) - placed[ch]
                            tie_break = d + ch * 1e-6 + (1e-7 if li % n_channels == ch else 0.0)
                            if tie_break > best_d:
                                best_d = tie_break
                                best_ch = ch
                        seq[li] = best_ch
                        placed[best_ch] += 1
                color_seqs[color_key] = seq

        # Slicing and clipping
        status_dict["stage"] = "Clipping layers"
        status_dict["progress"] = 35.0

        clipped_tris = []
        ZOFF = LH / 2

        n_raw_tris = len(all_raw_tris)
        progress_step = max(1, n_raw_tris // 50)

        for fi in range(n_raw_tris):
            if status_dict.get("cancel_requested", False):
                return

            if fi % progress_step == 0:
                pct = 35.0 + (fi / n_raw_tris) * 35.0
                status_dict["progress"] = pct

            t_idx = all_raw_tris[fi]
            v1, v2, v3 = scaled_verts[t_idx[0]], scaled_verts[t_idx[1]], scaled_verts[t_idx[2]]
            fz_min = min(v1[2], v2[2], v3[2])
            fz_max = max(v1[2], v2[2], v3[2])

            tri = [list(v1), list(v2), list(v3)]

            li_start = max(0, int(math.floor((fz_min - ZOFF) / LH)))
            li_end = min(n_layers - 1, int(math.ceil((fz_max - ZOFF) / LH)))

            color_key = unique_keys[fi]

            if single_filament_internal and face_visibility is not None and fi < len(face_visibility) and face_visibility[fi] < internal_visibility_threshold:
                internal_fi = n_channels - 1
                if li_start == li_end or li_start >= n_layers:
                    cz_c = (fz_min + fz_max) / 2
                    li2 = max(0, min(n_layers - 1, int(math.floor((cz_c - ZOFF) / LH))))
                    clipped_tris.append((tri, internal_fi))
                else:
                    for li in range(li_start, li_end + 1):
                        if li >= n_layers:
                            break
                        zB = ZOFF + li * LH
                        zT = ZOFF + (li + 1) * LH
                        for st in cTZ(tri, zB, zT):
                            clipped_tris.append((st, internal_fi))
                continue

            if dither_method == 'BLUE_NOISE':
                cum = color_cumdemands.get(color_key)
                if li_start == li_end or li_start >= n_layers:
                    cz = (fz_min + fz_max) / 2
                    li2 = max(0, min(n_layers - 1, int(math.floor((cz - ZOFF) / LH))))
                    for ft, fi in _bn_assign_tri(tri, layer_ox[li2], layer_oy[li2],
                                                  blue_noise_tile, bn_size, bn_scale,
                                                  cum, n_channels, li2):
                        clipped_tris.append((ft, fi))
                else:
                    for li in range(li_start, li_end + 1):
                        if li >= n_layers:
                            break
                        zB = ZOFF + li * LH
                        zT = ZOFF + (li + 1) * LH
                        sub_tris = cTZ(tri, zB, zT)
                        ox = layer_ox[li]
                        oy = layer_oy[li]
                        for st in sub_tris:
                            for ft, fi in _bn_assign_tri(st, ox, oy,
                                                          blue_noise_tile, bn_size, bn_scale,
                                                          cum, n_channels, li):
                                clipped_tris.append((ft, fi))
            else:
                seq = color_seqs[color_key]
                if li_start == li_end or li_start >= n_layers:
                    cz = (fz_min + fz_max) / 2
                    li2 = max(0, min(n_layers - 1, int(math.floor((cz - ZOFF) / LH))))
                    clipped_tris.append((tri, seq[li2]))
                else:
                    for li in range(li_start, li_end + 1):
                        if li >= n_layers:
                            break
                        zB = ZOFF + li * LH
                        zT = ZOFF + (li + 1) * LH
                        sub_tris = cTZ(tri, zB, zT)
                        for st in sub_tris:
                            clipped_tris.append((st, seq[li]))

        # Weld vertices
        status_dict["stage"] = "Welding vertices"
        status_dict["progress"] = 70.0

        weld_map = {}
        welded_verts = []
        welded_tris = []
        paint_arr = []
        
        n_clipped = len(clipped_tris)
        progress_step_w = max(1, n_clipped // 50)

        for w_idx, (tri, filament_idx) in enumerate(clipped_tris):
            if status_dict.get("cancel_requested", False):
                return
                
            if w_idx % progress_step_w == 0:
                pct = 70.0 + (w_idx / n_clipped) * 15.0  # maps to 70% - 85%
                status_dict["progress"] = pct

            tri_idx = []
            for pt in tri:
                key = (round(pt[0], 6), round(pt[1], 6), round(pt[2], 6))
                if key not in weld_map:
                    weld_map[key] = len(welded_verts)
                    welded_verts.append(pt)
                tri_idx.append(weld_map[key])
                
            if tri_idx[0] != tri_idx[1] and tri_idx[1] != tri_idx[2] and tri_idx[0] != tri_idx[2]:
                welded_tris.append(tri_idx)
                paint_arr.append(filament_idx)

        all_verts_arr = np.array(welded_verts, dtype=np.float32)
        all_tris_arr = np.array(welded_tris, dtype=np.int32)
        all_paint_arr = np.array(paint_arr, dtype=np.int32)

        # Build colorgroup and global mapping
        global_palette = []
        for p_idx in channels_order:
            color = palette_rgbs[p_idx]
            global_palette.append((_rgb_to_hex(color), color))

        global_to_extruder = {}
        for i in range(len(channels_order)):
            global_to_extruder[i + 1] = channels_order[i] + 1

        # Slicer settings config
        # Slicer settings config (must preserve original palette order so paint IDs align with extruders)
        ch_colors6 = [_rgb_to_hex(color).upper() for color in palette_rgbs]
        project_settings_json = json.dumps({"filament_colour": ch_colors6}, indent=4)
        
        prusa_extruder_colors = ";".join(ch_colors6)
        prusa_nozzles = ",".join(["0.4"] * len(palette_rgbs))
        prusa_config = f"; extruder_colour = {prusa_extruder_colors}\n; filament_colour = {prusa_extruder_colors}\n; nozzle_diameter = {prusa_nozzles}\n"
        
        safe_n = mesh_name.replace('&', '&amp;').replace('"', '&quot;')
        prusa_model_config = f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n <object id="2" instances_count="1">\n  <metadata type="object" key="name" value="{safe_n}"/>\n  <volume firstid="0" lastid="{len(all_tris_arr)-1}">\n   <metadata type="volume" key="name" value="{safe_n}"/>\n   <metadata type="volume" key="volume_type" value="ModelPart"/>\n   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n   <metadata type="volume" key="extruder" value="0"/>\n  </volume>\n </object>\n</config>'

        shifted_paint_arr = all_paint_arr + 1

        status_dict["stage"] = "Generating 3D model XML"
        status_dict["progress"] = 85.0

        model, parent_id, child_ids_and_extruders = _build_model_element(
            all_verts_arr, all_tris_arr, shifted_paint_arr, global_palette,
            global_to_extruder=global_to_extruder
        )
        model_settings = _build_model_settings_element(parent_id, child_ids_and_extruders)

        status_dict["stage"] = "Writing 3MF package"
        status_dict["progress"] = 90.0

        _write_3mf_archive(filepath, model, model_settings, project_settings_json, prusa_config, prusa_model_config)

        status_dict["progress"] = 100.0
        status_dict["stage"] = "Completed"
        status_dict["done"] = True
        status_dict["total_triangles"] = len(all_tris_arr)
    except Exception as e:
        status_dict["error"] = e
        status_dict["done"] = True


# ── Export3MF ────────────────────────────────────────────────────────────

class Export3MF(bpy.types.Operator):
    """Export selected mesh objects as 3MF for multi-color 3D printing."""
    bl_idname = "object.export_3mf"
    bl_label = "Export 3MF"
    bl_options = {'REGISTER'}

    export_scope: bpy.props.EnumProperty(
        name="Objects",
        items=[
            ('SELECTED', 'Selected', 'Export selected mesh objects'),
            ('ALL_MESH', 'All Mesh', 'Export all visible mesh objects'),
        ],
        default='SELECTED',
    )
    apply_transforms: bpy.props.BoolProperty(
        name="Apply Transforms",
        description="Apply location, rotation and scale before export",
        default=True,
    )

    _timer = None
    _thread = None
    _thread_status = None
    _snap = None
    _progress = 0.0
    _stage = ""
    _temp_solid_meshes = None
    _solidified_mappings = None
    _phase = ""
    _prepared_tasks = None
    _meshes = None
    _filepath = ""
    _palette_rgbs = None
    _is_dithered = False

    @classmethod
    def poll(cls, context):
        msg = _poll_common(context)
        if msg:
            cls.poll_message_set(msg)
            return False
        return True

    def draw(self, context):
        self.layout.prop(self, "export_scope")
        self.layout.prop(self, "apply_transforms")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        meshes = _get_mesh_objects(context, self.export_scope)
        if not meshes:
            self.report({'ERROR'}, "No mesh objects to export.")
            return {'CANCELLED'}

        scene = context.scene
        if not scene.stablegen_print_palette:
            prepopulate_palette(scene)
            
        import numpy as np
        palette_rgbs_linear = np.array([item.color for item in scene.stablegen_print_palette], dtype=np.float32)
        palette_rgbs = [tuple(c) for c in linear_to_srgb_numpy(palette_rgbs_linear)]
        if not palette_rgbs:
            self.report({'ERROR'}, "Print color palette is empty.")
            return {'CANCELLED'}

        # Verify SG_PrintPreview if Make Solid is enabled
        make_solid = getattr(scene, "stablegen_print_make_solid", False)
        if make_solid:
            for o in meshes:
                if not o.data.color_attributes.get("SG_PrintPreview"):
                    self.report({'ERROR'}, f"Make Solid requires the 'SG_PrintPreview' color attribute. Run 'Preview Colors' first on {o.name}.")
                    return {'CANCELLED'}

        self._snap = _SelectionSnapshot(context)
        self._meshes = meshes
        self._palette_rgbs = palette_rgbs
        self._is_dithered = getattr(scene, "stablegen_print_dithered", True)
        self._temp_solid_meshes = []
        self._solidified_mappings = {}
        
        output_dir = get_dir_path(context, "baked")
        os.makedirs(output_dir, exist_ok=True)
        self._filepath = os.path.join(output_dir, f"{meshes[0].name}.3mf")

        if self.apply_transforms:
            bpy.ops.object.select_all(action='DESELECT')
            for o in meshes:
                o.select_set(True)
            context.view_layer.objects.active = meshes[0]
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        self._progress = 0.0
        self._stage = "Initializing"

        if make_solid:
            self._phase = 'RAYCASTING'
            self._stage = "Preparing raycast tasks"
            
            self._prepared_tasks = []
            try:
                for obj in meshes:
                    task = _prepare_mesh_for_raycasting(obj)
                    self._prepared_tasks.append(task)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to prepare meshes for raycasting: {e}")
                self._snap.restore(context)
                return {'CANCELLED'}
                
            import math
            raycast_count = getattr(context.scene, "stablegen_print_raycast_count", 12)
            angle_offsets = [0.0]
            step = 2.0 * math.pi / raycast_count
            i = 1
            while len(angle_offsets) < raycast_count:
                angle_offsets.append(i * step)
                if len(angle_offsets) < raycast_count:
                    angle_offsets.append(-i * step)
                i += 1
            
            import threading
            self._thread_status = {
                "progress": 0.0,
                "stage": "Starting visibility check",
                "done": False,
                "cancel_requested": False
            }
            self._thread = threading.Thread(
                target=_raycast_visibility_worker,
                args=(self._prepared_tasks, angle_offsets, self._thread_status),
                daemon=True
            )
            self._thread.start()
        else:
            self._phase = 'EXPORTING'
            self._stage = "Extracting mesh data"
            
            success = self._start_exporting_worker(context)
            if not success:
                self._snap.restore(context)
                return {'CANCELLED'}

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {'RUNNING_MODAL'}

    def _start_exporting_worker(self, context):
        scene = context.scene
        orig_make_solid = scene.stablegen_print_make_solid
        scene.stablegen_print_make_solid = False
        
        try:
            import numpy as np
            import threading
            import math
            
            solid_data = {}
            dithered_data = {}
            
            if not self._is_dithered:
                # ── SOLID MODE Extraction ──
                all_verts = []
                all_tris = []
                all_paint = []
                global_palette = []

                for obj in self._meshes:
                    image = _find_texture_image(obj)
                    verts, tris, paint_colors, palette = _extract_mesh_data(
                        obj, apply_transforms=False,
                        palette_rgbs=self._palette_rgbs, image=image,
                    )
                    
                    base_vi = sum(v.shape[0] for v in all_verts) if all_verts else 0
                    all_verts.append(verts)
                    all_tris.append(tris + base_vi)
                    all_paint.append(paint_colors)

                    for hex_c, rgb in palette:
                        if hex_c not in {c for c, _ in global_palette}:
                            global_palette.append((hex_c, rgb))

                if not all_tris:
                    self.report({'ERROR'}, "No valid mesh data extracted.")
                    return False

                all_verts_arr = np.concatenate(all_verts, axis=0)
                all_tris_arr = np.concatenate(all_tris, axis=0)
                all_paint_arr = np.concatenate(all_paint, axis=0)

                global_to_extruder = {}
                for seq_ext in range(1, len(global_palette) + 1):
                    hex_c = global_palette[seq_ext - 1][0]
                    p_idx = -1
                    for idx, item in enumerate(scene.stablegen_print_palette):
                        if _rgb_to_hex(item.color).upper() == hex_c.upper():
                            p_idx = idx
                            break
                    global_to_extruder[seq_ext] = p_idx + 1 if p_idx != -1 else seq_ext

                import json
                ch_colors6 = [_rgb_to_hex(rgb).upper() for rgb in self._palette_rgbs]
                project_settings_json = json.dumps({"filament_colour": ch_colors6}, indent=4)
                
                prusa_extruder_colors = ";".join(ch_colors6)
                prusa_nozzles = ",".join(["0.4"] * len(self._palette_rgbs))
                prusa_config = f"; extruder_colour = {prusa_extruder_colors}\n; filament_colour = {prusa_extruder_colors}\n; nozzle_diameter = {prusa_nozzles}\n"
                
                safe_n = self._meshes[0].name.replace('&', '&amp;').replace('"', '&quot;')
                prusa_model_config = f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n <object id="2" instances_count="1">\n  <metadata type="object" key="name" value="{safe_n}"/>\n  <volume firstid="0" lastid="{len(all_tris_arr)-1}">\n   <metadata type="volume" key="name" value="{safe_n}"/>\n   <metadata type="volume" key="volume_type" value="ModelPart"/>\n   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n   <metadata type="volume" key="extruder" value="0"/>\n  </volume>\n </object>\n</config>'

                geom_obj_id = 1
                parent_obj_id = 2
                
                solid_data = {
                    "all_verts_arr": all_verts_arr,
                    "all_tris_arr": all_tris_arr,
                    "all_paint_arr": all_paint_arr,
                    "global_palette": global_palette,
                    "global_to_extruder": global_to_extruder,
                    "parent_id": parent_obj_id,
                    "child_ids_and_extruders": [(geom_obj_id, 1)],
                    "project_settings_json": project_settings_json,
                    "prusa_config": prusa_config,
                    "prusa_model_config": prusa_model_config
                }
            else:
                # ── DITHERED MODE Extraction ──
                raw_verts_list = []
                raw_tris_list = []
                raw_face_colors_list = []
                raw_face_vis_list = []
                
                for obj in self._meshes:
                    image = _find_texture_image(obj)
                    temp, eval_obj = _get_triangulated_mesh(obj, apply_transforms=False)
                        
                    try:
                        _require_printable_mesh(
                            temp,
                            label=obj.name,
                            require_closed=True,
                        )
                        n_v = len(temp.vertices)
                        v_co = np.empty(n_v * 3, dtype=np.float32)
                        temp.vertices.foreach_get("co", v_co)
                        v_co = v_co.reshape((-1, 3))
                        
                        if image:
                            face_colors = _collect_face_colors(temp, image)[:, :3]
                        else:
                            color_layer = temp.color_attributes.get("SG_PrintPreview")
                            if color_layer:
                                n_loops = len(temp.loops)
                                loop_colors = np.empty(n_loops * 4, dtype=np.float32)
                                color_layer.data.foreach_get("color", loop_colors)
                                loop_colors = loop_colors.reshape((-1, 4))
                                
                                n_tris = len(temp.loop_triangles)
                                tri_loops = np.empty(n_tris * 3, dtype=np.int32)
                                temp.loop_triangles.foreach_get("loops", tri_loops)
                                tri_loops = tri_loops.reshape((-1, 3))
                                
                                face_colors = loop_colors[tri_loops].mean(axis=1)[:, :3]
                                if color_layer.data_type == 'BYTE_COLOR':
                                    face_colors = srgb_to_linear_numpy(face_colors)
                            else:
                                face_colors = np.ones((len(temp.loop_triangles), 3), dtype=np.float32)
                        
                        t_idx = np.empty(len(temp.loop_triangles) * 3, dtype=np.int32)
                        temp.loop_triangles.foreach_get("vertices", t_idx)
                        t_idx = t_idx.reshape((-1, 3))
                        
                        base_vi = sum(v.shape[0] for v in raw_verts_list) if raw_verts_list else 0
                        raw_verts_list.append(v_co)
                        raw_tris_list.append(t_idx + base_vi)
                        raw_face_colors_list.append(face_colors)

                        fv_attr = temp.attributes.get("_SG_FaceVisibility")
                        if fv_attr is not None and fv_attr.domain == 'FACE':
                            n_polys = len(temp.polygons)
                            poly_vis = np.zeros(n_polys, dtype=np.float32)
                            fv_attr.data.foreach_get("value", poly_vis.tolist())
                            n_tris_fv = len(temp.loop_triangles)
                            tri_poly = np.empty(n_tris_fv, dtype=np.int32)
                            temp.loop_triangles.foreach_get("polygon_index", tri_poly)
                            raw_face_vis_list.append(poly_vis[tri_poly])
                        else:
                            fv = _collect_face_visibility(temp)
                            raw_face_vis_list.append(fv if fv is not None else np.full(len(t_idx), -1.0, dtype=np.float32))
                    finally:
                        eval_obj.to_mesh_clear()
                            
                all_raw_verts = np.concatenate(raw_verts_list, axis=0)
                all_raw_tris = np.concatenate(raw_tris_list, axis=0)
                all_raw_face_colors = np.concatenate(raw_face_colors_list, axis=0)
                all_raw_face_vis = np.concatenate(raw_face_vis_list, axis=0)
                has_face_vis = bool(np.any(all_raw_face_vis >= 0.0))
                
                h_target = scene.stablegen_print_model_height
                LH = scene.stablegen_print_layer_height
                n_layers = max(1, int(math.ceil(h_target / LH)))
                
                face_colors_srgb = all_raw_face_colors
                q = 8
                face_colors_q = np.clip(np.round(face_colors_srgb * 255.0 / q) * q, 0, 255).astype(np.int32)
                unique_keys = [tuple(c) for c in face_colors_q]
                unique_colors_set = sorted(list(set(unique_keys)))
                unique_colors_arr = np.array(unique_colors_set, dtype=np.float32) / 255.0
                
                mode, channel_indices = classify_filaments(self._palette_rgbs)
                
                if mode == 'solid':
                    channels_order = list(range(len(self._palette_rgbs)))
                else:
                    channels_order = channel_indices
                    
                cyan_idx = -1
                if mode != 'solid':
                    for ci, p_idx in enumerate(channel_indices):
                        if scene.stablegen_print_palette[p_idx].name == "Cyan":
                            cyan_idx = ci
                            break
                            
                dithered_data = {
                    "all_raw_verts": all_raw_verts,
                    "all_raw_tris": all_raw_tris,
                    "all_raw_face_colors": all_raw_face_colors,
                    "h_target": h_target,
                    "LH": LH,
                    "n_layers": n_layers,
                    "unique_keys": unique_keys,
                    "unique_colors_arr": unique_colors_arr,
                    "mode": mode,
                    "channel_indices": channel_indices,
                    "channels_order": channels_order,
                    "mesh_name": self._meshes[0].name,
                    "cyan_idx": cyan_idx,
                    "dither_method": getattr(scene, "stablegen_print_dither_method", "Z_SEQUENCE"),
                    "single_filament_internal": getattr(scene, "stablegen_print_single_filament_internal", True),
                    "internal_visibility_threshold": getattr(scene, "stablegen_print_internal_visibility_threshold", 0.01),
                    "face_visibility": all_raw_face_vis if has_face_vis else None,
                    "solver_settings": {
                        "init_method": getattr(scene, "stablegen_print_solver_init", 'CLOSEST'),
                        "iterations": getattr(scene, "stablegen_print_solver_steps", 80),
                        "model_formula": getattr(scene, "stablegen_print_solver_formula", 'KUBELKA_MUNK'),
                        "scat_weight": getattr(scene, "stablegen_print_scattering_weight", 10.0),
                        "penalty_weight": getattr(scene, "stablegen_print_saturation_penalty", 1.0),
                    }
                }

            self._thread_status = {
                "progress": 0.0,
                "stage": "Starting export worker",
                "done": False,
                "cancel_requested": False
            }
            self._thread = threading.Thread(
                target=_export_3mf_worker,
                args=(self._is_dithered, self._filepath, self._palette_rgbs, solid_data, dithered_data, self._thread_status),
                daemon=True
            )
            self._thread.start()
            return True

        except Exception as e:
            self.report({'ERROR'}, f"Failed to start export thread: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            scene.stablegen_print_make_solid = orig_make_solid

    def modal(self, context, event):
        if event.type == 'TIMER':
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type in ('PROPERTIES', 'VIEW_3D'):
                        area.tag_redraw()

            status = self._thread_status
            if status.get("cancel_requested", False):
                self.report({'INFO'}, "Export cancelled by user.")
                self.cleanup(context)
                return {'FINISHED'}

            if status.get("done", False):
                if "error" in status:
                    err = status["error"]
                    self.report({'ERROR'}, f"3MF export failed: {err}")
                    import traceback
                    print(f"[StableGen 3MF] Error details:")
                    if isinstance(err, Exception):
                        traceback.print_exception(type(err), err, err.__traceback__)
                    else:
                        print(err)
                    self.cleanup(context)
                    return {'CANCELLED'}
                
                if self._phase == 'RAYCASTING':
                    self._phase = 'EXPORTING'
                    self._stage = "Raycasting completed. Starting export..."
                    
                    visible_faces_results = status["visible_faces_results"]
                    fill_gaps = getattr(context.scene, "stablegen_print_fill_gaps", True)
                    keep_largest = getattr(context.scene, "stablegen_print_keep_largest_island", True)
                    
                    try:
                        for idx, obj in enumerate(self._meshes):
                            vis_faces = visible_faces_results[idx]
                            face_vis = None
                            if getattr(context.scene, "stablegen_print_single_filament_internal", True):
                                temp_ms, eval_ms = _get_triangulated_mesh(obj, apply_transforms=False)
                                try:
                                    face_vis = _collect_face_visibility(temp_ms)
                                finally:
                                    eval_ms.to_mesh_clear()
                            solid_mesh = _make_solid_mesh_object(obj, fill_gaps=fill_gaps, keep_largest_island=keep_largest, visible_faces=vis_faces, face_visibility=face_vis)
                            self._temp_solid_meshes.append(solid_mesh)
                            
                            self._solidified_mappings[obj] = (obj.data, solid_mesh)
                            obj.data = solid_mesh
                            
                        success = self._start_exporting_worker(context)
                        if not success:
                            self.cleanup(context)
                            return {'CANCELLED'}
                            
                    except Exception as e:
                        self.report({'ERROR'}, f"Failed to solidify meshes: {e}")
                        import traceback
                        traceback.print_exc()
                        self.cleanup(context)
                        return {'CANCELLED'}
                else:
                    n_extruders = len(self._palette_rgbs)
                    total_tris = status.get("total_triangles", 0)
                    self.report({'INFO'},
                                f"3MF exported to {self._filepath} "
                                f"({n_extruders} extruder{'' if n_extruders == 1 else 's'}, "
                                f"{total_tris} triangles)")
                    self.cleanup(context)
                    return {'FINISHED'}

            self._progress = status.get("progress", 0.0)
            self._stage = status.get("stage", "Processing")

        return {'PASS_THROUGH'}

    def cleanup(self, context):
        if hasattr(self, "_snap") and self._snap:
            self._snap.restore(context)

        if hasattr(self, "_solidified_mappings") and self._solidified_mappings:
            for obj, (orig_mesh, solid_mesh) in self._solidified_mappings.items():
                obj.data = orig_mesh
                try:
                    bpy.data.meshes.remove(solid_mesh)
                except Exception as e:
                    print(f"[StableGen 3MF] Warning: could not remove solid mesh {solid_mesh.name}: {e}")

        if hasattr(self, "_timer") and self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class Export3MFCancel(bpy.types.Operator):
    """Cancel the active 3MF export process."""
    bl_idname = "object.export_3mf_cancel"
    bl_label = "Cancel 3MF Export"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        export_op = next(
            (op for win in context.window_manager.windows
             for op in win.modal_operators
             if op.bl_idname == 'OBJECT_OT_export_3mf'),
            None
        )
        if export_op:
            if hasattr(export_op, "_thread_status") and export_op._thread_status:
                export_op._thread_status["cancel_requested"] = True
                self.report({'INFO'}, "Requesting cancellation of 3MF export...")
            else:
                self.report({'WARNING'}, "Export operator has no active thread status.")
        else:
            self.report({'WARNING'}, "No active 3MF export operator found.")
        return {'FINISHED'}
