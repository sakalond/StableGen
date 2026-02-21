""" This file contains the operators and panels for the StableGen addon """
# disable import-error because pylint doesn't recognize the blenders internal modules
import os
import bpy, bmesh  # pylint: disable=import-error
import numpy as np
import math
import time
import mathutils
from mathutils import Matrix
import blf
from .utils import get_file_path, get_dir_path, remove_empty_dirs, get_compositor_node_tree, configure_output_node_paths, get_eevee_engine_id
import gpu
from gpu_extras.batch import batch_for_shader
import tempfile
import shutil
from PIL import Image

# Import wheels
import cv2 
import imageio
import imageio_ffmpeg

def purge_orphans():
    """
    Purge unused datablocks (images, materials, node groups, etc).
    Uses Blender's Outliner orphan purge.
    """
    try:
        # Blender 3.x/4.x: call recursively a few times to fully purge
        for _ in range(5):
            result = bpy.ops.outliner.orphans_purge(do_recursive=True)
            # If it reports "CANCELLED" or does nothing, we're done
            if 'CANCELLED' in result:
                break
    except Exception as e:
        print(f"[StableGen] Orphan purge failed: {e}")

def apply_vignette_to_mask(mask_file_path, feather_width=0.15, gamma=1.0, blur=True):
    """
    Soften hard edges in a grayscale visibility mask.

    Instead of only darkening a thin border at the image edges, this applies
    a Gaussian blur whose radius is proportional to the image size. That means
    *any* 0→1 transition in the mask (occlusion edges, camera frustum edges,
    etc.) becomes a smooth ramp, which the shader can use for a soft blend.

    feather_width: fraction of min(image_w, image_h) used as blur radius.
                   0.0 = no blur, 0.5 = very soft edges.
    gamma: optional gamma applied to the blurred mask (1.0 = none).
    blur: whether to apply Gaussian blur.
    """
    log_prefix = "[StableGen] Vignette:"

    # -------------------------------------------------------------------------
    # Basic guards
    # -------------------------------------------------------------------------
    if feather_width <= 0.0:
        return

    if not isinstance(mask_file_path, str):
        print(f"{log_prefix} mask_file_path must be a string, got {type(mask_file_path)}")
        return

    if not os.path.exists(mask_file_path):
        print(f"{log_prefix} mask file not found: {mask_file_path}")
        return

    img = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{log_prefix} failed to read mask: {mask_file_path}")
        return

    # -------------------------------------------------------------------------
    # Blur-based feathering
    # -------------------------------------------------------------------------
    h, w = img.shape[:2]
    base = img.astype(np.float32) / 255.0

    # Radius as a fraction of the smallest dimension
    min_dim = float(min(h, w))
    radius = max(1.0, feather_width * min_dim)

    # Kernel size must be odd and at least 3
    ksize = int(max(3, int(radius) | 1))

    if blur:
        blurred = cv2.GaussianBlur(base, (ksize, ksize), 0)
    else:
        blurred = base

    if gamma != 1.0:
        blurred = np.power(blurred, gamma)

    result = np.clip(blurred, 0.0, 1.0)
    result_u8 = (result * 255.0).astype(np.uint8)
    cv2.imwrite(mask_file_path, result_u8)

    print(
        f"{log_prefix} soft-edge blur applied to mask: {mask_file_path} "
        f"(ksize={ksize}, fw={feather_width}, gamma={gamma}, blur={blur})"
    )


def create_edge_feathered_mask(mask_path, feather_width=30):
    """Create an edge-feathered version of a visibility mask for projection blending.

    Uses a distance transform on the binary mask so that:
    - Interior pixels (far from any edge) → 1.0  (full new texture)
    - Edge pixels (near visibility boundary) → ramp 0→1 over *feather_width* px
    - Invisible pixels                      → 0.0  (keep original texture)

    The result is saved next to the original with an ``_edgefeather`` suffix.
    Returns the output path, or *None* on failure.
    """
    if not isinstance(mask_path, str) or not os.path.exists(mask_path):
        print(f"[StableGen] Edge-feather: mask not found: {mask_path}")
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[StableGen] Edge-feather: failed to read mask: {mask_path}")
        return None

    # Threshold to hard binary (visible = white, invisible = black)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Distance transform: each white pixel → distance to nearest black pixel
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Normalise into [0, 1] ramp over feather_width pixels
    feathered = np.clip(dist / max(feather_width, 1), 0.0, 1.0)

    feathered_u8 = (feathered * 255).astype(np.uint8)

    output_path = mask_path.replace('.png', '_edgefeather.png')
    cv2.imwrite(output_path, feathered_u8)
    print(f"[StableGen] Edge-feather mask saved: {output_path} (width={feather_width}px)")
    return output_path


def render_edge_feather_mask(context, to_export, camera, camera_index, feather_width=30, softness=1.0):
    """Render a geometry silhouette from *camera* and apply distance-transform
    edge feathering.

    All target objects render as white (Emission), non-target mesh objects are
    hidden, and the world is set to black.  The resulting binary silhouette is
    distance-transformed so that interior pixels = 1.0, boundary pixels ramp
    0→1 over *feather_width* pixels, and background = 0.0.

    The final mask is saved to ``inpaint/visibility/render{camera_index}_edgefeather.png``.
    Returns the output path, or *None* on failure.
    """
    output_dir = get_dir_path(context, "inpaint")["visibility"]
    os.makedirs(output_dir, exist_ok=True)
    raw_file = f"render{camera_index}_geomask"

    # ── Save original state ─────────────────────────────────────────────────
    original_camera = context.scene.camera
    original_engine = context.scene.render.engine
    original_transparent = context.scene.render.film_transparent
    original_samples = context.scene.cycles.samples
    original_view_transform = bpy.context.scene.view_settings.view_transform

    world = context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        context.scene.world = world
    original_use_nodes = world.use_nodes
    original_color = world.color.copy()
    original_bg_node_color = None
    original_bg_node_strength = None
    if world.use_nodes and world.node_tree:
        for wn in world.node_tree.nodes:
            if wn.type == 'BACKGROUND':
                original_bg_node_color = tuple(wn.inputs["Color"].default_value)
                original_bg_node_strength = wn.inputs["Strength"].default_value
                break

    # ── Camera ──────────────────────────────────────────────────────────────
    context.scene.camera = camera

    # ── Replace target-object materials with white Emission ─────────────────
    saved_materials = {}
    saved_active_materials = {}
    temp_materials = []
    for obj in to_export:
        saved_materials[obj] = list(obj.data.materials)
        saved_active_materials[obj] = obj.active_material

        mat = bpy.data.materials.new(name="_SG_EdgeFeather_Temp")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        emission = mat.node_tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 1.0
        out_node = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(emission.outputs[0], out_node.inputs["Surface"])

        obj.data.materials.clear()
        obj.data.materials.append(mat)
        temp_materials.append(mat)

    # ── Hide non-target mesh objects ────────────────────────────────────────
    hidden_restore = {}
    for obj in context.scene.objects:
        if obj.type == 'MESH' and obj not in to_export:
            hidden_restore[obj] = obj.hide_render
            obj.hide_render = True

    # ── World → black background ────────────────────────────────────────────
    if bpy.app.version >= (5, 0, 0):
        world.use_nodes = True
        if world.node_tree:
            for wn in world.node_tree.nodes:
                if wn.type == 'BACKGROUND':
                    wn.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                    wn.inputs["Strength"].default_value = 1.0
                    break
    else:
        world.color = (0, 0, 0)
        world.use_nodes = False

    # ── Render settings ─────────────────────────────────────────────────────
    context.scene.render.engine = 'CYCLES'
    context.scene.render.film_transparent = False
    context.scene.cycles.samples = 1
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.view_transform = 'Raw'

    view_layer = context.view_layer
    view_layer.use_pass_emit = True
    view_layer.use_pass_environment = True

    # ── Compositor ──────────────────────────────────────────────────────────
    context.scene.use_nodes = True
    node_tree = get_compositor_node_tree(context.scene)
    comp_nodes = node_tree.nodes
    comp_links = node_tree.links
    comp_nodes.clear()

    render_layers = comp_nodes.new('CompositorNodeRLayers')
    try:
        mix_node = comp_nodes.new('CompositorNodeMixRGB')
    except Exception:
        mix_node = comp_nodes.new('ShaderNodeMixRGB')
    mix_node.blend_type = 'ADD'
    mix_node.inputs[0].default_value = 1
    output_node = comp_nodes.new('CompositorNodeOutputFile')
    configure_output_node_paths(output_node, output_dir, raw_file)

    if bpy.app.version < (5, 0, 0):
        comp_links.new(render_layers.outputs['Emit'], mix_node.inputs[1])
        comp_links.new(render_layers.outputs['Env'], mix_node.inputs[2])
    else:
        comp_links.new(render_layers.outputs['Emission'], mix_node.inputs[1])
        comp_links.new(render_layers.outputs['Environment'], mix_node.inputs[2])
    comp_links.new(mix_node.outputs[0], output_node.inputs[0])

    # ── Render ──────────────────────────────────────────────────────────────
    bpy.ops.render.render(write_still=True)

    # ── Determine actual output path ────────────────────────────────────────
    frame_suffix = "0001" if bpy.app.version < (5, 0, 0) else ""
    raw_path = os.path.join(output_dir, f"{raw_file}{frame_suffix}.png")

    # ── Restore materials ───────────────────────────────────────────────────
    for obj, mats in saved_materials.items():
        obj.data.materials.clear()
        if saved_active_materials[obj]:
            obj.data.materials.append(saved_active_materials[obj])
        for m in mats:
            if m != saved_active_materials[obj]:
                obj.data.materials.append(m)
    for mat in temp_materials:
        if mat and mat.name in bpy.data.materials:
            bpy.data.materials.remove(mat)

    # ── Restore non-target visibility ───────────────────────────────────────
    for obj, was_hidden in hidden_restore.items():
        obj.hide_render = was_hidden

    # ── Restore render / world settings ─────────────────────────────────────
    context.scene.camera = original_camera
    context.scene.render.engine = original_engine
    context.scene.render.film_transparent = original_transparent
    context.scene.cycles.samples = original_samples
    bpy.context.scene.view_settings.view_transform = original_view_transform
    world.use_nodes = original_use_nodes
    world.color = original_color
    if original_bg_node_color is not None and world.node_tree:
        for wn in world.node_tree.nodes:
            if wn.type == 'BACKGROUND':
                wn.inputs["Color"].default_value = original_bg_node_color
                wn.inputs["Strength"].default_value = original_bg_node_strength
                break

    # ── Distance-transform edge feathering ──────────────────────────────────
    if not os.path.exists(raw_path):
        print(f"[StableGen] Edge-feather: raw mask not found after render: {raw_path}")
        return None

    mask_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"[StableGen] Edge-feather: failed to read raw mask: {raw_path}")
        return None

    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    linear = np.clip(dist / max(feather_width, 1), 0.0, 1.0).astype(np.float32)

    # Gaussian-blur the linear ramp to round off the kinks at both ends.
    # The linear ramp has sharp slope discontinuities at dist=0 (edge) and
    # dist=feather_width (interior plateau).  A small Gaussian blur smooths
    # these transitions without shifting the ramp position or compressing the
    # transition zone the way smoothstep does.
    #   softness == 0   → raw linear ramp (kinks preserved)
    #   softness == 1   → moderate rounding (sigma ≈ 25% of feather width)
    #   softness  > 1   → stronger rounding / wider smooth zone
    if softness > 0.01:
        blur_sigma = softness * feather_width * 0.25
        feathered = cv2.GaussianBlur(linear, (0, 0), sigmaX=blur_sigma)
        feathered = np.clip(feathered, 0.0, 1.0)
    else:
        feathered = linear

    feathered_u8 = (np.clip(feathered, 0.0, 1.0) * 255).astype(np.uint8)

    ef_path = os.path.join(output_dir, f"render{camera_index}_edgefeather.png")
    cv2.imwrite(ef_path, feathered_u8)

    # Clean up raw mask
    try:
        os.remove(raw_path)
    except OSError:
        pass

    print(f"[StableGen] Edge-feather mask saved: {ef_path} (width={feather_width}px, softness={softness:.2f})")
    return ef_path


def apply_uv_inpaint_texture(context, obj, baked_image_path):
    """
    Apply a UV inpainted/baked texture to the active material.

    Priority:
      1) StableGen projection chain: replace LAST MixRGB with "Projection" in name (Color2).
      2) Fallback: traverse from Material Output and find a suitable MIX_RGB (Color2 unlinked
         or linked from TEX_IMAGE), then replace that (Color2).

    In both cases:
      - Insert TexImage + UVMap (first non-ProjectionUV layer) feeding Color2.
      - Remove existing Color2 links.
      - If replacing a projection input, also remove the old TexImage node and orphaned image.
    """

    mat = obj.active_material
    if not mat or not mat.use_nodes or not mat.node_tree:
        print("[StableGen] No active material or no node tree.")
        return

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # -----------------------------
    # Load baked image
    # -----------------------------
    try:
        img = bpy.data.images.load(baked_image_path)
    except Exception as e:
        print("[StableGen] Failed to load baked image", baked_image_path, e)
        return

    # -----------------------------
    # Helpers
    # -----------------------------
    def pick_uv_name():
        uv_layers = getattr(obj.data, "uv_layers", None)
        if not uv_layers or len(uv_layers) == 0:
            return None
        for uv_layer in uv_layers:
            if not uv_layer.name.startswith("ProjectionUV") and uv_layer.name != "_SG_ProjectionBuffer":
                return uv_layer.name
        return uv_layers.active.name

    def clear_color2_links(mix_node, cleanup_teximage=False):
        """Remove links to Color2. Optionally remove upstream TexImage nodes & orphaned images."""
        for link in list(mix_node.inputs["Color2"].links):
            src = link.from_node
            links.remove(link)

            if cleanup_teximage and src and src.type == "TEX_IMAGE":
                old_img = src.image
                try:
                    nodes.remove(src)
                except Exception:
                    pass
                if old_img and old_img.users == 0:
                    try:
                        bpy.data.images.remove(old_img)
                    except Exception:
                        pass

    def inject_tex_uv_into_color2(mix_node):
        """Create TexImage+UVMap and connect into mix_node Color2."""
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.label = "BakedProjection"
        tex.location = (mix_node.location[0] - 300, mix_node.location[1])

        uv = nodes.new("ShaderNodeUVMap")
        uv_name = pick_uv_name()
        if uv_name:
            uv.uv_map = uv_name
        uv.location = (tex.location[0] - 300, tex.location[1] - 200)

        links.new(uv.outputs["UV"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], mix_node.inputs["Color2"])

    # ============================================================
    # 1) StableGen projection chain path
    # ============================================================
    proj_mix_nodes = [n for n in nodes if n.type == "MIX_RGB" and "Projection" in n.name]
    if proj_mix_nodes:
        mix_node = sorted(proj_mix_nodes, key=lambda n: n.name)[-1]

        # Remove previous projection input, clean up old TexImage nodes
        clear_color2_links(mix_node, cleanup_teximage=True)

        # Inject baked texture
        inject_tex_uv_into_color2(mix_node)

        print(f"[StableGen] Injected baked projection into projection chain ({mix_node.name}) on {obj.name}")
        return

    # ============================================================
    # 2) Fallback traversal path (your original logic)
    # ============================================================
    output_node = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    if not output_node or not output_node.inputs["Surface"].links:
        print("[StableGen] No Material Output surface link found for fallback.")
        return

    before_output = output_node.inputs["Surface"].links[0].from_node

    if before_output.type == "BSDF_PRINCIPLED":
        # Follow: Output.Surface -> Principled -> Base Color -> upstream node
        base_color = before_output.inputs.get("Base Color")
        if base_color and base_color.links:
            current_node = base_color.links[0].from_node
        else:
            current_node = None
    else:
        current_node = before_output

    mix_node = None
    visited = set()

    while current_node and current_node.as_pointer() not in visited:
        visited.add(current_node.as_pointer())

        if current_node.type == "MIX_RGB":
            c2 = current_node.inputs.get("Color2")
            if c2:
                if (not c2.is_linked) or (c2.is_linked and c2.links and c2.links[0].from_node.type == "TEX_IMAGE"):
                    mix_node = current_node
                    break

        c2 = current_node.inputs.get("Color2")
        if c2 and c2.links:
            current_node = c2.links[0].from_node
        else:
            current_node = None

    if not mix_node:
        print("[StableGen] No suitable fallback MixRGB node found.")
        return

    # Remove any existing links on Color2 (fallback does NOT aggressively delete nodes/images)
    clear_color2_links(mix_node, cleanup_teximage=False)

    # Insert baked texture + UVMap
    inject_tex_uv_into_color2(mix_node)

    print(f"[StableGen] Injected baked projection into fallback chain ({mix_node.name}) on {obj.name}")


def flatten_projection_material_for_refine(context, obj, baked_image_path):
    """
    Replace the StableGen ProjectionMaterial on this object with a minimal
    baked-base material that still matches the expectations of:
      - export_emit_image / _setup_emit_material
      - project_image local_edit logic

    Final graph:
        UV Map -> Baked Image -> MixRGB -> Principled BSDF -> Output
    """
    import os
    try:
        img = bpy.data.images.load(baked_image_path)
    except Exception as e:
        print(f"[StableGen] Failed to load baked image for {obj.name}: {baked_image_path} ({e})")
        return

    # -------------------------------------------------------------------------
    # 1) Find or create a suitable material
    # -------------------------------------------------------------------------
    target_mat = None
    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.name.startswith("ProjectionMaterial"):
            target_mat = mat
            break

    if target_mat is None:
        target_mat = obj.active_material

    if target_mat is None:
        target_mat = bpy.data.materials.new(name="ProjectionMaterial")
        obj.data.materials.append(target_mat)

    obj.active_material = target_mat
    if obj.active_material_index < 0:
        if target_mat not in obj.data.materials:
            obj.data.materials.append(target_mat)
        obj.active_material_index = obj.material_slots.find(target_mat.name)

    # -------------------------------------------------------------------------
    # 2) Build: UV -> Tex -> MixRGB -> Principled -> Output
    # -------------------------------------------------------------------------
    target_mat.use_nodes = True
    nodes = target_mat.node_tree.nodes
    links = target_mat.node_tree.links

    # Clear old nodes
    for node in list(nodes):
        nodes.remove(node)

    # Output
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (800, 0)

    # Principled
    principled_node = nodes.new("ShaderNodeBsdfPrincipled")
    principled_node.location = (500, 0)
    principled_node.inputs["Roughness"].default_value = 1.0

    # MixRGB that _setup_emit_material expects to sit before the Principled
    mix_node = nodes.new("ShaderNodeMixRGB")
    mix_node.location = (200, 0)
    mix_node.use_clamp = True
    # Fac=0 -> output = Color1 (baked tex)
    mix_node.inputs["Fac"].default_value = 0.0

    # Baked texture
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.location = (-100, 0)
    tex_node.image = img

    # UV map
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.location = (-400, -150)

    # -------------------------------------------------------------------------
    # 3) Choose a stable UV map (avoid ProjectionUV)
    # -------------------------------------------------------------------------
    uv_name = None
    for uv in obj.data.uv_layers:
        if not uv.name.startswith("ProjectionUV") and uv.name != "_SG_ProjectionBuffer":
            uv_name = uv.name
            break

    if not uv_name:
        uv_names = [uv.name for uv in obj.data.uv_layers]
        if "BakeUV" in uv_names:
            uv_name = "BakeUV"
        else:
            uv_layer = obj.data.uv_layers.new(name="BakeUV")
            uv_name = uv_layer.name

    uv_node.uv_map = uv_name

    # -------------------------------------------------------------------------
    # 4) Wire up the graph
    # -------------------------------------------------------------------------
    links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
    links.new(tex_node.outputs["Color"], mix_node.inputs["Color1"])

    links.new(mix_node.outputs["Color"], principled_node.inputs["Base Color"])

    # Drive emission too (Blender 4.x uses "Emission Color")
    if "Emission Color" in principled_node.inputs:
        links.new(mix_node.outputs["Color"], principled_node.inputs["Emission Color"])
    elif "Emission" in principled_node.inputs:
        links.new(mix_node.outputs["Color"], principled_node.inputs["Emission"])

    if "Emission Strength" in principled_node.inputs:
        principled_node.inputs["Emission Strength"].default_value = 1.0

    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])


    print(f"[StableGen] Flattened ProjectionMaterial for '{obj.name}' to baked texture '{os.path.basename(baked_image_path)}'")

def export_emit_image(context, to_export, camera_id=None, bg_color=(0.5, 0.5, 0.5), view_transform='Standard', fallback_color=(0,0,0)):
        """
        Exports a emit-only render of the scene from a camera's perspective.
        :param context: Blender context.
        :param camera_id: ID of the camera.
        :return: None
        """
        print("Exporting emit render")
        # Set animation frame to 1
        bpy.context.scene.frame_set(1)

        # Store original materials and create temporary ones
        original_materials = {}
        original_active_material = {}
        temporary_materials = {}

        # Check if there is BSDF applied

        # We need to temporarily disconnect BDSF nodes and connect their inputs directly to the output

        for obj in to_export:
            # Store original materials
            original_materials[obj] = list(obj.data.materials)
            original_active_material[obj] = obj.active_material

            # Copy active material and switch to it
            mat = obj.active_material
            if not mat:
                continue
            mat_copy = mat.copy()

            # Clear materials and assign temp material
            obj.data.materials.clear()
            obj.data.materials.append(mat_copy)

            # Store the temporary material for later deletion
            temporary_materials[obj] = mat_copy

            # Enable use of nodes
            mat_copy.use_nodes = True
            nodes = mat_copy.node_tree.nodes
            links = mat_copy.node_tree.links

            # Find the output node
            output = None
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    output = node
                    break

            if not output or not output.inputs[0].links:
                continue

                # Check the type of the node which connects to output
            before_output = output.inputs[0].links[0].from_node
            if before_output.type == 'BSDF_PRINCIPLED':
                # Find the last color mix node
                color_mix = output.inputs[0].links[0].from_node.inputs[0].links[0].from_node
                # Set color 2 to fallback color
                if not "visibility" in str(camera_id):
                    color_mix.inputs["Color2"].default_value = (fallback_color[0], fallback_color[1], fallback_color[2], 1.0)
            else:
                # Already a color mix node
                color_mix = before_output
                # Set color 2 to fallback color
                if not "visibility" in str(camera_id):
                    color_mix.inputs["Color2"].default_value = (fallback_color[0], fallback_color[1], fallback_color[2], 1.0)

                # Blender 5.0+: Wrap color output with Emission shader so the Emit pass picks it up
                if bpy.app.version >= (5, 0, 0):
                    emission_node = nodes.new("ShaderNodeEmission")
                    emission_node.location = (output.location.x - 200, output.location.y)
                    links.new(color_mix.outputs[0], emission_node.inputs["Color"])
                    links.new(emission_node.outputs[0], output.inputs["Surface"])
                continue

            # Find the last color mix node
            color_mix = output.inputs[0].links[0].from_node.inputs[0].links[0].from_node
            # Connect the color mix node directly to the output
            if bpy.app.version >= (5, 0, 0):
                # Blender 5.0+: Wrap in Emission shader for Emit pass
                emission_node = nodes.new("ShaderNodeEmission")
                emission_node.location = (output.location.x - 200, output.location.y)
                links.new(color_mix.outputs[0], emission_node.inputs["Color"])
                links.new(emission_node.outputs[0], output.inputs["Surface"])
            else:
                links.new(color_mix.outputs[0], output.inputs[0])

        output_dir = get_dir_path(context, "inpaint")["visibility"] if "visibility" in str(camera_id) else get_dir_path(context, "inpaint")["render"]
        output_file = f"render{camera_id}" if camera_id is not None else "render"

        # Store and set world settings
        world = context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            context.scene.world = world

        # Store original settings
        original_engine = context.scene.render.engine
        original_film_transparent = context.scene.render.film_transparent
        original_use_nodes = world.use_nodes
        original_color = world.color.copy()
        original_bg_node_color = None
        original_bg_node_strength = None
        if world.use_nodes and world.node_tree:
            for wn in world.node_tree.nodes:
                if wn.type == 'BACKGROUND':
                    original_bg_node_color = tuple(wn.inputs["Color"].default_value)
                    original_bg_node_strength = wn.inputs["Strength"].default_value
                    break

        # Set world background color
        if bpy.app.version >= (5, 0, 0):
            # Blender 5.0+: Use shader nodes for world background (required for Environment pass)
            world.use_nodes = True
            if world.node_tree:
                for wn in world.node_tree.nodes:
                    if wn.type == 'BACKGROUND':
                        wn.inputs["Color"].default_value = (*bg_color[:3], 1.0)
                        wn.inputs["Strength"].default_value = 1.0
                        break
        else:
            world.color = bg_color
            world.use_nodes = False

        # Switch to CYCLES render engine (needed for emission pass rendering)
        context.scene.render.engine = 'CYCLES'
        # Force CPU + OSL only for Blender < 5.1 (native Raycast nodes don't need it)
        if bpy.app.version < (5, 1, 0):
            if hasattr(context.scene.cycles, 'shading_system'):
                context.scene.cycles.shading_system = True
            else:
                context.scene.cycles.use_osl = True
            context.scene.cycles.device = 'CPU'
        context.scene.render.film_transparent = False
        # Change color management to standard
        bpy.context.scene.display_settings.display_device = 'sRGB'
        bpy.context.scene.view_settings.view_transform = view_transform
        context.scene.cycles.samples = 1  # Minimum samples for speed
        # Configure view layer settings for diffuse-only
        view_layer = context.view_layer
        view_layer.use_pass_diffuse_color = False
        view_layer.use_pass_diffuse_direct = False
        view_layer.use_pass_diffuse_indirect = False

        # Disable all other passes
        view_layer.use_pass_ambient_occlusion = False
        view_layer.use_pass_shadow = False
        view_layer.use_pass_emit = True
        view_layer.use_pass_environment = True


        # Set up compositor nodes
        context.scene.use_nodes = True
        node_tree = get_compositor_node_tree(context.scene)
        nodes = node_tree.nodes
        links = node_tree.links

        # Clear existing nodes
        nodes.clear()

        # Create nodes
        render_layers = nodes.new('CompositorNodeRLayers')
        try:
            mix_node = nodes.new('CompositorNodeMixRGB')
        except:
            mix_node = nodes.new('ShaderNodeMixRGB')
        mix_node.blend_type = 'ADD'
        mix_node.inputs[0].default_value = 1
        output_node = nodes.new('CompositorNodeOutputFile')
        configure_output_node_paths(output_node, output_dir, output_file)

        # Connect emission to output
        # Blender 5.0+ renamed pass names: Emit -> Emission, Env -> Environment
        if bpy.app.version < (5, 0, 0):
            links.new(render_layers.outputs['Emit'], mix_node.inputs[1])
            links.new(render_layers.outputs['Env'], mix_node.inputs[2])
        else:
            links.new(render_layers.outputs['Emission'], mix_node.inputs[1])
            links.new(render_layers.outputs['Environment'], mix_node.inputs[2])
        links.new(mix_node.outputs[0], output_node.inputs[0])

        # Render
        bpy.ops.render.render(write_still=True)

        # Post-processing for visibility masks
        if "visibility" in str(camera_id):
            # Determine the actual file path (Blender 4.x appends 0001, 5.x does not)
            if bpy.app.version >= (5, 0, 0):
                final_path = os.path.join(output_dir, f"{output_file}.png")
            else:
                final_path = os.path.join(output_dir, f"{output_file}0001.png")

            if context.scene.visibility_vignette and (context.scene.generation_method == 'local_edit' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'local_edit')):
                # Smooth edge feathering, no blocky mask
                apply_vignette_to_mask(
                    final_path,
                    feather_width=context.scene.visibility_vignette_width,
                    gamma=1.0,
                    blur=context.scene.visibility_vignette_blur
                )
            elif context.scene.mask_blocky:
                # Only do blocky mask if vignette is OFF
                expanded_mask = expand_mask_to_blocks(final_path, block_size=8)
                if expanded_mask is not None:
                    expanded_mask_u8 = (expanded_mask * 255).astype(np.uint8)
                    cv2.imwrite(final_path, expanded_mask_u8)

        # Restore original settings
        context.scene.render.engine = original_engine
        context.scene.render.film_transparent = original_film_transparent
        world.use_nodes = original_use_nodes
        world.color = original_color
        if original_bg_node_color is not None and world.node_tree:
            for wn in world.node_tree.nodes:
                if wn.type == 'BACKGROUND':
                    wn.inputs["Color"].default_value = original_bg_node_color
                    wn.inputs["Strength"].default_value = original_bg_node_strength
                    break
        bpy.context.scene.view_settings.view_transform = 'Standard'

        # Restore original materials
        for obj, materials in original_materials.items():
            obj.data.materials.clear()
            # First append the original active material
            if original_active_material[obj]:
                obj.data.materials.append(original_active_material[obj])
            for mat in materials:
                if mat != original_active_material[obj]:
                    obj.data.materials.append(mat)

        # Clean up temporary materials
        for _, temp_mat in temporary_materials.items():
            if temp_mat and temp_mat.name in bpy.data.materials:
                bpy.data.materials.remove(temp_mat)

        print(f"Emmision render saved to: {os.path.join(output_dir, output_file)}.png")


def export_render(context, camera_id=None, output_dir=None, filename=None):
    """
    Renders the scene from a camera's perspective using Workbench.
    Creates temporary materials for consistent rendering.
    :param context: Blender context.
    :param camera_id: ID of the camera for the output filename.
    :param output_dir: Optional output directory.
    :param filename: Optional filename (without component/frame suffix).
    :return: None
    """
    print("Exporting render using Workbench")

    # Store original materials and create temporary ones
    original_materials = {}
    original_active_material = {}
    for obj in context.view_layer.objects:
        if obj.type == 'MESH':
            # Store original materials
            original_materials[obj] = list(obj.data.materials)
            original_active_material[obj] = obj.active_material

            # Create temporary material
            temp_mat = bpy.data.materials.new(name="TempRenderMaterial")
            temp_mat.use_nodes = True # Even Workbench uses nodes for basic color
            nodes = temp_mat.node_tree.nodes
            links = temp_mat.node_tree.links

            # Clear default nodes
            for node in nodes:
                nodes.remove(node)

            # Create basic material output and diffuse BSDF (Workbench respects Base Color)
            mat_output = nodes.new('ShaderNodeOutputMaterial')
            diffuse = nodes.new('ShaderNodeBsdfDiffuse') # Simple diffuse color
            diffuse.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0) # Default grey
            diffuse.inputs['Roughness'].default_value = 0.5
            links.new(diffuse.outputs['BSDF'], mat_output.inputs['Surface'])

            # Clear materials and assign temp material
            obj.data.materials.clear()
            obj.data.materials.append(temp_mat)

    # Set animation frame to 1
    context.scene.frame_set(1)

    # Setup output path
    if output_dir is None:
        output_dir = get_dir_path(context, "misc")
    
    if filename is None:
        output_file = f"render{camera_id}" if camera_id is not None else "render"
    else:
        output_file = filename

    # Store original render settings
    original_engine = context.scene.render.engine
    original_workbench_settings = {
        'lighting': context.scene.display.shading.light,
        'color_type': context.scene.display.shading.color_type
    }
    original_render_filepath = context.scene.render.filepath
    original_image_settings = context.scene.render.image_settings.file_format

    # Switch to WORKBENCH render engine and configure settings
    context.scene.render.engine = 'BLENDER_WORKBENCH'
    # Configure Workbench for a flat, consistent look if needed
    context.scene.display.shading.light = 'STUDIO'
    context.scene.display.shading.color_type = 'SINGLE'

    render_layer = context.view_layer
    original_combined = render_layer.use_pass_combined

    # Enable combined pass for Workbench
    render_layer.use_pass_combined = True

    # Set up output nodes (Compositor setup remains the same)
    context.scene.use_nodes = True
    node_tree = get_compositor_node_tree(context.scene)
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    render_layers = nodes.new('CompositorNodeRLayers')
    output_node = nodes.new('CompositorNodeOutputFile')
    configure_output_node_paths(output_node, output_dir, output_file)
    links.new(render_layers.outputs['Image'], output_node.inputs[0])

    # Render
    bpy.ops.render.render(write_still=True)

    # Restore original materials
    for obj, materials in original_materials.items():
        obj.data.materials.clear()
        # First append the original active material
        if original_active_material[obj]:
            obj.data.materials.append(original_active_material[obj])
        for mat in materials:
            if mat != original_active_material[obj]:
                obj.data.materials.append(mat)

    # Restore original render settings
    render_layer.use_pass_combined = original_combined

    # Clean up temporary materials
    # Use a while loop to safely remove materials while iterating
    temp_mats = [m for m in bpy.data.materials if m.name.startswith("TempRenderMaterial")]
    for mat in temp_mats:
        bpy.data.materials.remove(mat)


    # Restore original render settings
    context.scene.render.engine = original_engine
    context.scene.display.shading.light = original_workbench_settings['lighting']
    context.scene.display.shading.color_type = original_workbench_settings['color_type']
    context.scene.render.filepath = original_render_filepath
    context.scene.render.image_settings.file_format = original_image_settings


    print(f"Render saved to: {os.path.join(output_dir, output_file)}0001.png") # Blender adds frame number


def export_viewport(context, camera_id=None, output_dir=None, filename=None):
    """
    Renders the scene using viewport OpenGL render to include overlays.
    :param context: Blender context.
    :param camera_id: ID of the camera for the output filename.
    :param output_dir: Optional output directory.
    :param filename: Optional filename (without component/frame suffix).
    :return: None
    """
    print("Exporting render using Viewport (OpenGL)")

    # Setup output path
    if output_dir is None:
        output_dir = get_dir_path(context, "misc")

    if filename is None:
        output_file = f"render{camera_id}" if camera_id is not None else "render"
    else:
        output_file = filename

    # Store original render settings
    original_engine = context.scene.render.engine
    original_render_filepath = context.scene.render.filepath
    original_image_settings = context.scene.render.image_settings.file_format

    # Switch to WORKBENCH render engine for consistent viewport shading
    context.scene.render.engine = 'BLENDER_WORKBENCH'

    # Find a viewport
    viewport_area = None
    viewport_region = None
    viewport_space = None
    viewport_region_3d = None
    viewport_window = None

    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                viewport_area = area
                viewport_window = window
                for region in area.regions:
                    if region.type == 'WINDOW':
                        viewport_region = region
                        break
                viewport_space = area.spaces.active
                viewport_region_3d = viewport_space.region_3d if viewport_space else None
                break
        if viewport_area:
            break

    if not (viewport_area and viewport_region and viewport_space and viewport_region_3d and viewport_window):
        print("Viewport render failed: no VIEW_3D area found.")
        context.scene.render.engine = original_engine
        return

    # Store viewport settings
    original_view_perspective = viewport_region_3d.view_perspective
    original_shading_type = viewport_space.shading.type
    original_overlay_show = viewport_space.overlay.show_overlays

    # Configure viewport for camera render with overlays
    viewport_region_3d.view_perspective = 'CAMERA'
    viewport_space.shading.type = 'RENDERED'
    viewport_space.overlay.show_overlays = True

    # Configure output filepath
    context.scene.render.filepath = os.path.join(output_dir, f"{output_file}.png")
    context.scene.render.image_settings.file_format = 'PNG'

    override = {
        'window': viewport_window,
        'screen': viewport_window.screen,
        'area': viewport_area,
        'region': viewport_region,
        'scene': context.scene,
        'space_data': viewport_space,
        'region_data': viewport_region_3d,
    }
    with bpy.context.temp_override(**override):
        bpy.ops.render.opengl(write_still=True, view_context=True)

    # Restore viewport settings
    viewport_region_3d.view_perspective = original_view_perspective
    viewport_space.shading.type = original_shading_type
    viewport_space.overlay.show_overlays = original_overlay_show

    # Restore original render settings
    context.scene.render.engine = original_engine
    context.scene.render.filepath = original_render_filepath
    context.scene.render.image_settings.file_format = original_image_settings

    print(f"Viewport render saved to: {os.path.join(output_dir, output_file)}.png")

def export_canny(context, camera_id=None, low_threshold=0, high_threshold=80):
    """
    Uses export_render and openCV to generate a Canny edge detection image.
    :param context: Blender context.
    :param camera_id: ID of the camera for the output filename.
    :param low_threshold: Low threshold for edge detection.
    :param high_threshold: High threshold for edge detection.
    :return: None
    """
    # Render the scene
    export_render(context, camera_id)

    # Load the rendered image
    output_dir_render = get_dir_path(context, "misc")
    output_dir_canny = get_dir_path(context, "controlnet")["canny"]
    # Blender 4.x appends 0001 frame suffix, 5.x does not
    frame_suffix = "0001" if bpy.app.version < (5, 0, 0) else ""
    output_file = f"render{camera_id}{frame_suffix}" if camera_id is not None else "render"
    image_path = os.path.join(output_dir_render, f"{output_file}.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Save the edge detection image
    output_file = f"canny{camera_id}{frame_suffix}" if camera_id is not None else "canny"
    cv2.imwrite(os.path.join(output_dir_canny, f"{output_file}.png"), edges)

    print(f"Canny edge detection saved to: {os.path.join(output_dir_canny, output_file)}.png")


def expand_mask_to_blocks(mask_file_path, block_size=8):
    """
    Loads a mask image from a file path and processes it so that any
    block_size x block_size grid cell containing any non-black pixel (value > 0)
    becomes fully white (1.0).

    Args:
        mask_file_path (str): The path to the mask image file.
        block_size (int): The size of the grid blocks (default: 8).

    Returns:
        np.ndarray | None: The processed mask as a NumPy array (float32, normalized [0, 1]).
                           Returns None if the file cannot be loaded or processed.
    """
    if not isinstance(mask_file_path, str):
        print(f"Error: mask_file_path must be a string. Got: {type(mask_file_path)}")
        return None
    if not os.path.exists(mask_file_path):
        print(f"Error: Mask file not found at {mask_file_path}")
        return None
    if not os.path.isfile(mask_file_path):
         print(f"Error: Path provided is not a file: {mask_file_path}")
         return None

    try:
        # Load the image using Pillow
        with Image.open(mask_file_path) as img:
            # Convert to grayscale ('L') which typically represents intensity
            mask_pil = img.convert("L")
            # Convert PIL Image to numpy array
            mask_array = np.array(mask_pil)

        # --- Normalize mask to float32 [0, 1] ---
        # Ensures comparison with 0.0 is reliable regardless of original bit depth
        if mask_array.dtype == np.uint8:
            mask_array = mask_array.astype(np.float32) / 255.0
        elif np.issubdtype(mask_array.dtype, np.integer):
            max_val = np.iinfo(mask_array.dtype).max
            mask_array = mask_array.astype(np.float32) / max_val if max_val > 0 else mask_array.astype(np.float32)
        elif not np.issubdtype(mask_array.dtype, np.floating):
            print(f"Warning: Unsupported mask dtype {mask_array.dtype} after loading. Trying to convert.")
            mask_array = mask_array.astype(np.float32) # Attempt conversion
        # Ensure it's precisely within [0, 1] after potential float conversion
        mask_array = np.clip(mask_array, 0.0, 1.0)
        # --- End normalization ---

        max_value = 1.0 # Output will be normalized float [0, 1]

        height, width = mask_array.shape
        # Create output mask initialized to zeros (black), ensure float32 type
        output_mask = np.zeros_like(mask_array, dtype=np.float32)

        # --- Core block processing logic ---
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Define block boundaries, handle image edges
                y_start = y
                y_end = min(y + block_size, height)
                x_start = x
                x_end = min(x + block_size, width)

                # Extract the current block
                block = mask_array[y_start:y_end, x_start:x_end]

                # Check if *any* pixel value in the block is greater than 0.0
                if np.any(block > 0.0):
                    # If yes, set the corresponding block in the output mask to 1.0 (white)
                    output_mask[y_start:y_end, x_start:x_end] = max_value
        # --- End of core logic ---

        return output_mask

    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_file_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing mask file {mask_file_path}: {e}")
        return None


def export_visibility(context, to_export, obj=None, camera_visibility=None, prepare_only=False):
    """     
    Exports the visibility of the mesh by temporarily altering the shading nodes.
    :param context: Blender context.
    :param filepath: Path to the output file.
    :param obj: Blender object.
    :param camera_visibility: Camera object for visibility calculation.
    :param prepare_only: If True, only prepare the visibility material without
                         baking/rendering and without restoring the original materials.
                         Used by debug tools to inspect the material in the viewport.
    :return: None
    """
    # Store original materials and create temporary ones
    original_materials = {}
    original_active_material = {}
    temporary_materials = {}

    def prepare_material(obj):
        mat = obj.active_material
        if not mat:
            return False
        
        # Store original materials
        original_materials[obj] = list(obj.data.materials)
        original_active_material[obj] = obj.active_material  # Store original active material

        # Store original active mat

        # Copy active material and switch to it
        mat = obj.active_material
        if not mat:
            return False
        mat_copy = mat.copy()

        # Clear materials and assign temp material
        obj.data.materials.clear()
        obj.data.materials.append(mat_copy)
        
        # Store temporary material for later deletion
        temporary_materials[obj] = mat_copy
        
        # Enable use of nodes
        mat_copy.use_nodes = True
        nodes = mat_copy.node_tree.nodes
        links = mat_copy.node_tree.links

        # Find the output node
        output = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output = node
                break

        if not output:
            return False
        
        if not output.inputs[0].links:
            return False
        
        # Determine which input to used based on existence of BSDF node before the output
        if output.inputs[0].links and output.inputs[0].links[0].from_node.type == 'BSDF_PRINCIPLED':
            principled = output.inputs[0].links[0].from_node
            if not principled.inputs[0].links:
                # Principled BSDF exists but nothing is connected to its
                # Base Color – this is not a projection material.
                return False
            color_mix = principled.inputs[0].links[0].from_node
            input = principled.inputs[0]
        else:
            color_mix = output.inputs[0].links[0].from_node
            input = output.inputs[0]
        # Add equal node between color mix and bsdf
        equal = nodes.new("ShaderNodeMath")
        
        # Use compare operation to filter to only 1 or 0
        
        if context.scene.generation_method == 'sequential' and context.scene.sequential_smooth:
            # Add color ramp node
            compare = nodes.new("ShaderNodeValToRGB")
            compare.color_ramp.interpolation = 'LINEAR'
            compare.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
            compare.color_ramp.elements[0].position = context.scene.sequential_factor_smooth if context.scene.generation_method == 'sequential' else 0.5
            compare.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
            compare.color_ramp.elements[1].position = context.scene.sequential_factor_smooth_2 if context.scene.generation_method == 'sequential' else 1.0
            links.new(color_mix.outputs[0], compare.inputs[0])
            links.new(compare.outputs[0], input)
        elif not context.scene.allow_modify_existing_textures or context.scene.generation_method == 'sequential':
            equal.operation = 'COMPARE'
            equal.inputs[1].default_value = 1
            equal.inputs[2].default_value = context.scene.sequential_factor if context.scene.generation_method == 'sequential' else (1e-5 if bpy.app.version >= (5, 1, 0) else 0.0) # Small epsilon needed for blender 5.1+
            equal.location = (color_mix.location[0], color_mix.location[1])
            links.new(color_mix.outputs[0], equal.inputs[0])
            links.new(equal.outputs[0], input)
        else:
            links.new(color_mix.outputs[0], input)

        while True:
            # Remove color ramp connected to fac, connect directly to color ramp's fac
            color_ramp = color_mix.inputs[0].links[0].from_node
            fac_node = color_ramp.inputs[0].links[0].from_node
            # Add subtract node
            subtract = nodes.new("ShaderNodeMath")
            subtract.operation = 'SUBTRACT'
            subtract.location = (color_ramp.location[0], color_ramp.location[1])
            subtract.inputs[0].default_value = 1
            links.new(fac_node.outputs[0], subtract.inputs[1])
            links.new(subtract.outputs[0], color_mix.inputs["Fac"])
            nodes.remove(color_ramp)
            # Disconnect color1
            links.remove(color_mix.inputs[1].links[0])
            color_mix.inputs["Color1"].default_value = (0, 0, 0, 1)
            if not (color_mix.inputs["Color2"].links and (color_mix.inputs[2].links[0].from_node.type == 'MIX_RGB')):
                color_mix.inputs["Color2"].default_value = (1, 1, 1, 1)
                break
            else:
                color_mix = color_mix.inputs["Color2"].links[0].from_node
        # If there is previous tex_image node, remove it (for cases when this function is called multiple times)
        if color_mix.inputs["Color2"].is_linked:
            nodes.remove(color_mix.inputs["Color2"].links[0].from_node)
            
        for node in nodes:
            if node.type == 'SCRIPT' and "Power" in node.inputs:
                try:
                    # Set according to weight_exponent_mask
                    node.inputs["Power"].default_value = context.scene.weight_exponent if context.scene.weight_exponent_mask else 1.0
                except Exception as e:
                    print(f"  - Warning: Failed to set Power for node '{node.name}'. Error: {e}")
            # Also handle native Raycast path (Blender 5.1+) where power is a MATH POWER node
            elif node.type == 'MATH' and node.operation == 'POWER' and node.label == 'power_weight':
                try:
                    node.inputs[1].default_value = context.scene.weight_exponent if context.scene.weight_exponent_mask else 1.0
                except Exception as e:
                    print(f"  - Warning: Failed to set Power for native node '{node.name}'. Error: {e}")

        # When the normalization chain exists (multi-camera), the node
        # chain is:  power_weight(exp=1) → base_weight → NormW(÷max) →
        #            SharpW(pow exp) → mix tree.
        # For the visibility map we need the *original* un-normalized
        # weight: pow(cos(θ), target_exp) × binary_gates.
        # We achieve this by:
        #   1. Setting power_weight to target_exp (restores original weight)
        #   2. Making SharpW a passthrough (exp=1) and rerouting its input
        #      from NormW's source (the base weight) so the DIVIDE-by-max
        #      is bypassed entirely.
        has_norm_nodes = any(
            n.type == 'MATH' and n.operation == 'DIVIDE'
            and n.label.startswith('NormW-')
            for n in nodes
        )
        if has_norm_nodes:
            target_exp = context.scene.weight_exponent if context.scene.weight_exponent_mask else 1.0
            for node in nodes:
                # Restore per-camera power to the desired exponent
                if (node.type == 'MATH' and node.operation == 'POWER'
                        and node.label == 'power_weight'):
                    try:
                        node.inputs[1].default_value = target_exp
                    except Exception as e:
                        print(f"  - Warning: Failed to set power_weight '{node.name}'. Error: {e}")
                elif node.type == 'SCRIPT' and "Power" in node.inputs:
                    try:
                        node.inputs["Power"].default_value = target_exp
                    except Exception as e:
                        print(f"  - Warning: Failed to set OSL Power '{node.name}'. Error: {e}")

                # Bypass NormW+SharpW: reroute SharpW to read from
                # NormW's source (the base weight), set exponent to 1.0
                elif (node.type == 'MATH' and node.operation == 'POWER'
                        and node.label.startswith('SharpW-')):
                    try:
                        # SharpW.inputs[0] ← NormW.outputs[0]
                        # NormW.inputs[0]  ← base_weight_output
                        # Reroute: SharpW.inputs[0] ← base_weight_output
                        if node.inputs[0].links:
                            norm_node = node.inputs[0].links[0].from_node
                            if (norm_node.label.startswith('NormW-')
                                    and norm_node.inputs[0].links):
                                base_out = norm_node.inputs[0].links[0].from_socket
                                links.remove(node.inputs[0].links[0])
                                links.new(base_out, node.inputs[0])
                        node.inputs[1].default_value = 1.0
                    except Exception as e:
                        print(f"  - Warning: Failed to bypass SharpW '{node.name}'. Error: {e}")

                # Also bypass standalone NormW nodes (when user_exponent
                # was 1.0 at build time, there's no SharpW — the NormW
                # DIVIDE node is used directly in the mix tree).
                elif (node.type == 'MATH' and node.operation == 'DIVIDE'
                        and node.label.startswith('NormW-')):
                    try:
                        # Check if this NormW feeds directly into the
                        # mix tree (no SharpW after it).
                        feeds_sharp = False
                        for link in node.outputs[0].links:
                            if (link.to_node.type == 'MATH'
                                    and link.to_node.operation == 'POWER'
                                    and link.to_node.label.startswith('SharpW-')):
                                feeds_sharp = True
                                break
                        if not feeds_sharp:
                            # This NormW feeds the mix tree directly.
                            # Reroute its downstream links to its source.
                            if node.inputs[0].links:
                                base_out = node.inputs[0].links[0].from_socket
                                for link in list(node.outputs[0].links):
                                    to_socket = link.to_socket
                                    links.remove(link)
                                    links.new(base_out, to_socket)
                    except Exception as e:
                        print(f"  - Warning: Failed to bypass NormW '{node.name}'. Error: {e}")
               
        return True
    
    # Prepare the material

    if not obj:
        # Prepare for all objects
        for obj in to_export:
            if not prepare_material(obj):
                return False
    else:
        if not prepare_material(obj):
            return False
        
    if prepare_only:
        # Debug mode: leave the visibility material applied for viewport inspection
        # Rename the temp materials so they are identifiable as debug materials
        for obj_key, temp_mat in temporary_materials.items():
            temp_mat.name = f"SG_Debug_Visibility_{obj_key.name}"
        return True

    # Bake or render the texture
    if not camera_visibility:
        output_dir = get_dir_path(context, "uv_inpaint")["visibility"]
        output_file = f"{obj.name}_baked_visibility"
        prepare_baking(context)
        bake_texture(context, obj, suffix="_visibility", texture_resolution=1024, view_transform='Raw', output_dir=output_dir)
        # Use openCV to normalize the image
        image_path = os.path.join(output_dir, f"{output_file}.png")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Normalize the image
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # Save the image
        cv2.imwrite(image_path, image)

        if context.scene.visibility_vignette and (context.scene.generation_method == 'local_edit' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'local_edit')):
            apply_vignette_to_mask(
                image_path,
                feather_width=context.scene.visibility_vignette_width,
                gamma=1.0,
                blur=context.scene.visibility_vignette_blur
            )
    else:
        # Make sure the camera is active and set to render
        cameras = [obj for obj in context.scene.objects if obj.type == 'CAMERA']
        cameras.sort(key=lambda x: x.name)
        camera_visibility_index = [i for i, camera in enumerate(cameras) if camera == camera_visibility][0]
        camera_render_index = (camera_visibility_index + 1) % len(cameras)
        camera_render = cameras[camera_render_index]
        context.scene.camera = camera_render
        export_emit_image(context, to_export, camera_id=f"{camera_render_index}_visibility", bg_color=(1, 1, 1), view_transform='Raw', fallback_color=(1,1,1))

    # Restore original materials
    for obj, materials in original_materials.items():
        obj.data.materials.clear()
        # First append the original active material
        if original_active_material[obj]:
            obj.data.materials.append(original_active_material[obj])
        for mat in materials:
            if mat != original_active_material[obj]:
                obj.data.materials.append(mat)
                
    # Clean up temporary materials
    for _, temp_mat in temporary_materials.items():
        if temp_mat and temp_mat.name in bpy.data.materials:
            bpy.data.materials.remove(temp_mat)

    return True


# =========================================================
# Camera Placement Helper Functions
# =========================================================


def _existing_camera_directions(mesh_center):
    """Return unit-direction vectors from *mesh_center* toward each existing
    camera in the scene.  Used by 'Consider existing cameras' to treat
    pre-existing cameras as already-placed directions."""
    dirs = []
    center = np.array(mesh_center, dtype=float)
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            pos = np.array(obj.location, dtype=float)
            d = pos - center
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                dirs.append(d / norm)
    return dirs


def _filter_near_existing(directions, existing_dirs, min_angle_deg=30.0):
    """Remove directions that are within *min_angle_deg* of any existing
    camera direction.  Both *directions* and *existing_dirs* should be
    lists of unit-length numpy arrays."""
    if not existing_dirs or not directions:
        return directions
    cos_thresh = math.cos(math.radians(min_angle_deg))
    existing_np = np.array(existing_dirs)          # (M, 3)
    filtered = []
    for d in directions:
        d_np = np.asarray(d, dtype=float)
        n = np.linalg.norm(d_np)
        if n < 1e-12:
            continue
        d_unit = d_np / n
        dots = existing_np @ d_unit                # (M,)
        if dots.max() < cos_thresh:
            filtered.append(d)
    return filtered

def _fibonacci_sphere_points(n):
    """Generate *n* approximately evenly-spaced unit vectors on a sphere
    using a Fibonacci spiral.  Returns list of (x, y, z) tuples."""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))
    return points


def _gather_target_meshes(context):
    """Return a list of mesh objects to cover.
    - If any selected objects are meshes, use those.
    - Otherwise, use ALL mesh objects in the scene.
    """
    selected = [o for o in context.selected_objects if o.type == 'MESH']
    if selected:
        return selected
    return [o for o in context.scene.objects if o.type == 'MESH']


def _get_mesh_face_data(objs):
    """Return world-space face (normals, areas, centers) as numpy arrays.
    *objs* can be a single object or a list of mesh objects."""
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    all_normals, all_areas, all_centers = [], [], []
    for obj in objs:
        mesh = obj.data
        mat = obj.matrix_world
        rot = mat.to_3x3()
        scale_det = abs(rot.determinant())
        n = len(mesh.polygons)
        normals = np.empty((n, 3))
        areas = np.empty(n)
        centers = np.empty((n, 3))
        for idx, poly in enumerate(mesh.polygons):
            wn = rot @ poly.normal
            wn.normalize()
            normals[idx] = (wn.x, wn.y, wn.z)
            wc = mat @ poly.center
            centers[idx] = (wc.x, wc.y, wc.z)
            areas[idx] = poly.area * (scale_det ** 0.5)
        all_normals.append(normals)
        all_areas.append(areas)
        all_centers.append(centers)
    return np.vstack(all_normals), np.concatenate(all_areas), np.vstack(all_centers)


def _filter_bottom_faces(normals, areas, centers, angle_rad):
    """Remove faces whose normals point more than *angle_rad* below the
    horizon (negative-Z).  Returns filtered (normals, areas, centers).

    *angle_rad* = 80° (≈1.40 rad) removes faces whose normal is > 80° below
    horizontal, i.e. nearly straight down.  The Z-component threshold is
    ``-cos(angle_rad)``."""
    threshold_z = -math.cos(angle_rad)
    mask = normals[:, 2] >= threshold_z  # keep faces *above* the threshold
    return normals[mask], areas[mask], centers[mask]


def _get_mesh_verts_world(objs):
    """Return world-space vertex positions as (N, 3) numpy array.
    *objs* can be a single object or a list of mesh objects."""
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    parts = []
    for obj in objs:
        mesh = obj.data
        mat = obj.matrix_world
        n = len(mesh.vertices)
        verts = np.empty((n, 3))
        for i, v in enumerate(mesh.vertices):
            wv = mat @ v.co
            verts[i] = (wv.x, wv.y, wv.z)
        parts.append(verts)
    return np.vstack(parts)


def _camera_basis(cam_dir_np):
    """Build an orthonormal camera basis from a centre-to-camera direction.

    Returns ``(right, up, d_unit)`` where
    * *right*   – camera local X (numpy unit vector)
    * *up*      – camera local Y (numpy unit vector)
    * *d_unit*  – normalised centre-to-camera direction (camera local Z)

    When the direction is nearly vertical the world-up fallback switches
    from Z to Y, giving a deterministic orientation that avoids
    ``to_track_quat`` gimbal degeneracy.
    """
    d = cam_dir_np / np.linalg.norm(cam_dir_np)
    forward = -d
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, d


def _rotation_from_basis(right, up, d_unit):
    """Build a Blender rotation (Euler) from camera basis vectors.

    The resulting rotation aligns the camera's local axes:
    * local  X → *right*
    * local  Y → *up*
    * local  Z → *d_unit*   (away from mesh; -Z is the look direction)
    """
    rot_mat = mathutils.Matrix((
        (right[0], up[0], d_unit[0]),
        (right[1], up[1], d_unit[1]),
        (right[2], up[2], d_unit[2]),
    ))
    return rot_mat.to_euler()


def _compute_silhouette_distance(verts_world, center_np, cam_dir_np, fov_x, fov_y, margin=0.10):
    """Compute the optimal distance along *cam_dir_np* so every mesh vertex
    fits inside the camera frame with *margin* breathing room.

    Uses a perspective-correct formula so objects with depth along the
    viewing direction are never clipped.

    Returns ``(distance, aim_offset)`` where *aim_offset* is a 3-element numpy
    vector that shifts from *center_np* to the visual center of the silhouette.
    The caller should aim the camera at ``center_np + aim_offset`` and place
    it at ``center_np + aim_offset + direction * distance``.
    """
    right, up, d = _camera_basis(cam_dir_np)
    forward = -d

    # Project every vertex onto the camera's right / up plane
    rel = verts_world - center_np
    proj_r = rel @ right
    proj_u = rel @ up

    r_min, r_max = float(proj_r.min()), float(proj_r.max())
    u_min, u_max = float(proj_u.min()), float(proj_u.max())

    # Visual centre of the silhouette (midpoint of extents)
    mid_r = (r_max + r_min) / 2.0
    mid_u = (u_max + u_min) / 2.0

    # Aim offset: shifts from mesh center to silhouette visual centre
    aim_offset = mid_r * right + mid_u * up

    eff_fov_x = fov_x * (1.0 - margin)
    eff_fov_y = fov_y * (1.0 - margin)

    tan_hx = math.tan(eff_fov_x / 2) if eff_fov_x > 0.02 else 1e-6
    tan_hy = math.tan(eff_fov_y / 2) if eff_fov_y > 0.02 else 1e-6

    # Perspective-correct distance: for each vertex the minimum camera
    # distance along the view direction is  |lateral| / tan(half_fov) - depth
    # where depth is signed distance from the aim point along the view ray.
    aim_point = center_np + aim_offset
    rel_aim = verts_world - aim_point
    pr = rel_aim @ right
    pu = rel_aim @ up
    # Depth: positive = in front of aim point (toward camera)
    pd = rel_aim @ d  # dot with camera direction (away from mesh)

    min_dist_r = np.abs(pr) / tan_hx + pd
    min_dist_u = np.abs(pu) / tan_hy + pd

    dist = max(float(min_dist_r.max()), float(min_dist_u.max()), 0.5)

    # --- Refine aim_offset using perspective angular centre ---
    # The orthographic midpoint doesn't account for perspective foreshortening.
    # Compute where the visual centre actually is from the camera position and
    # shift the aim to centre it, then recompute the distance.
    cam_pos = aim_point + d * dist
    rel_cam = verts_world - cam_pos
    depth = -(rel_cam @ d)  # positive = in front of camera
    depth = np.maximum(depth, 0.001)
    ang_r = np.arctan2(rel_cam @ right, depth)
    ang_u = np.arctan2(rel_cam @ up, depth)
    ang_mid_r = (float(ang_r.max()) + float(ang_r.min())) / 2.0
    ang_mid_u = (float(ang_u.max()) + float(ang_u.min())) / 2.0
    # Convert angular offset to world-space shift (at the computed distance)
    aim_offset = aim_offset + dist * math.tan(ang_mid_r) * right + dist * math.tan(ang_mid_u) * up

    # Recompute distance with refined aim
    aim_point = center_np + aim_offset
    rel_aim = verts_world - aim_point
    pr = rel_aim @ right
    pu = rel_aim @ up
    pd = rel_aim @ d
    min_dist_r = np.abs(pr) / tan_hx + pd
    min_dist_u = np.abs(pu) / tan_hy + pd
    dist = max(float(min_dist_r.max()), float(min_dist_u.max()), 0.5)

    return dist, aim_offset


def _kmeans_on_sphere(directions, weights, k, max_iter=50):
    """Spherical K-means: cluster unit vectors weighted by area.
    Returns (k, 3) numpy array of cluster-centre unit vectors."""
    n_pts = len(directions)
    if n_pts == 0 or k == 0:
        return np.zeros((max(k, 1), 3))
    k = min(k, n_pts)
    rng = np.random.default_rng(42)
    probs = weights / weights.sum()
    indices = rng.choice(n_pts, size=k, replace=False, p=probs)
    centers = directions[indices].copy()
    for _ in range(max_iter):
        dots = directions @ centers.T
        labels = np.argmax(dots, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                ws = (directions[mask] * weights[mask, np.newaxis]).sum(axis=0)
                nrm = np.linalg.norm(ws)
                new_centers[j] = ws / nrm if nrm > 0 else centers[j]
            else:
                new_centers[j] = centers[j]
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    return centers


def _compute_pca_axes(verts):
    """Return the 3 principal axes of a (N, 3) vertex array (rows of a
    3x3 array, sorted by descending eigenvalue)."""
    mean = verts.mean(axis=0)
    centered = verts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx].T  # rows = principal axes


def _greedy_coverage_directions(normals, areas, max_cameras=12,
                                coverage_target=0.95, n_candidates=200,
                                existing_dirs=None):
    """Greedy set-cover: iteratively pick the camera direction that adds the
    most newly-visible surface area (back-face culling only, no occlusion,
    for speed).  Returns (selected_directions, final_coverage_fraction)."""
    total_area = areas.sum()
    if total_area <= 0:
        return [], 0.0

    candidates = np.array(_fibonacci_sphere_points(n_candidates))

    # visibility[i, j] = True if face i faces toward candidate j
    # cos(75°) ≈ 0.26 – ignore near-grazing faces that won't texture well
    visibility = normals @ candidates.T > 0.26  # (n_faces, n_candidates)

    covered = np.zeros(len(areas), dtype=bool)
    # Pre-seed coverage from existing cameras
    if existing_dirs:
        for edir in existing_dirs:
            covered |= normals @ np.asarray(edir, dtype=float) > 0.26
    selected = []

    for _ in range(max_cameras):
        uncovered = ~covered
        # Vectorised: for each candidate sum the area of faces that are
        # both visible from it AND not yet covered
        new_vis = visibility & uncovered[:, np.newaxis]  # broadcast
        new_areas = (new_vis * areas[:, np.newaxis]).sum(axis=0)

        best_idx = int(np.argmax(new_areas))
        if new_areas[best_idx] < total_area * 0.005:   # < 0.5 % new coverage
            break

        selected.append(candidates[best_idx].copy())
        covered |= visibility[:, best_idx]

        coverage = float(areas[covered].sum() / total_area)
        if coverage >= coverage_target:
            break

    final_coverage = float(areas[covered].sum() / total_area) if covered.any() else 0.0
    return selected, final_coverage


# ──────────────────────────────────────────────────────────────────────
# BVH occlusion helpers
# ──────────────────────────────────────────────────────────────────────

def _build_bvh_trees(objs, depsgraph):
    """Build a list of BVHTree objects (one per mesh) for raycasting.

    Returns
    -------
    list[mathutils.bvhtree.BVHTree]
        One BVH per object, in world space.
    """
    from mathutils.bvhtree import BVHTree
    trees = []
    for obj in objs:
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)
        bm.transform(obj.matrix_world)
        tree = BVHTree.FromBMesh(bm)
        bm.free()
        trees.append(tree)
    return trees


def _ray_occluded(bvh_trees, origin, direction, max_dist):
    """Return True if *any* BVH tree has a hit between *origin* and
    *origin + direction * max_dist* (exclusive of the starting face).

    *origin* should already be slightly offset along the face normal to avoid
    self-intersection.
    """
    for tree in bvh_trees:
        hit_loc, _normal, _index, _dist = tree.ray_cast(
            mathutils.Vector(origin), mathutils.Vector(direction), max_dist)
        if hit_loc is not None:
            return True
    return False


def _greedy_select_from_visibility(vis, areas, max_cameras, coverage_target,
                                   candidates, existing_dirs=None,
                                   normals=None):
    """Run greedy set-cover on a pre-computed visibility matrix.

    Parameters
    ----------
    vis : ndarray (n_faces, n_candidates), bool
    areas : ndarray (n_faces,)
    max_cameras : int
    coverage_target : float
    candidates : ndarray (n_candidates, 3)
    existing_dirs : list[ndarray] or None
        Directions of existing cameras to pre-seed coverage.
    normals : ndarray (n_faces, 3) or None
        Face normals, required when *existing_dirs* is given.

    Returns
    -------
    (selected_directions, final_coverage)
    """
    total_area = areas.sum()
    if total_area <= 0:
        return [], 0.0

    covered = np.zeros(len(areas), dtype=bool)
    # Pre-seed coverage from existing cameras
    if existing_dirs and normals is not None:
        for edir in existing_dirs:
            covered |= normals @ np.asarray(edir, dtype=float) > 0.26
    selected = []

    for _ in range(max_cameras):
        uncovered = ~covered
        new_vis = vis & uncovered[:, np.newaxis]
        new_areas = (new_vis * areas[:, np.newaxis]).sum(axis=0)

        best_idx = int(np.argmax(new_areas))
        if new_areas[best_idx] < total_area * 0.005:
            break

        selected.append(candidates[best_idx].copy())
        covered |= vis[:, best_idx]

        coverage = float(areas[covered].sum() / total_area)
        if coverage >= coverage_target:
            break

    final_cov = float(areas[covered].sum() / total_area) if covered.any() else 0.0
    return selected, final_cov


# ──────────────────────────────────────────────────────────────────────
# Occlusion generators (modal-friendly, yield progress)
# ──────────────────────────────────────────────────────────────────────

def _occ_filter_faces_generator(normals, areas, centers, bvh_trees,
                                n_candidates=200):
    """Generator: determine which faces are visible from at least one
    camera direction (exterior faces).

    Yields progress floats in [0, 1].  Final result (via
    ``StopIteration.value``) is a boolean mask ``(n_faces,)`` where True
    means the face is visible from at least one direction.

    Uses early-exit per face: as soon as one unoccluded direction is found,
    the face is marked exterior and skipped for remaining candidates.
    """
    n_faces = len(normals)
    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26  # (F, C)
    exterior = np.zeros(n_faces, dtype=bool)
    epsilon = 0.001
    BATCH = 5

    for j in range(n_candidates):
        cam_dir = candidates[j]
        for i in range(n_faces):
            if exterior[i]:
                continue  # already proven exterior
            if not backface_vis[i, j]:
                continue
            origin = centers[i] + normals[i] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                exterior[i] = True
        if (j + 1) % BATCH == 0 or j == n_candidates - 1:
            yield (j + 1) / n_candidates
            # Early termination: all faces proven exterior
            if exterior.all():
                break

    return exterior


def _occ_vis_count_generator(normals, areas, centers, bvh_trees,
                              n_candidates=200):
    """Generator: count how many candidate directions can see each face.

    Like ``_occ_filter_faces_generator`` but returns per-face visibility
    *counts* instead of a boolean mask.  This enables continuous weighting
    rather than binary keep/remove filtering.

    Yields progress floats in [0, 1].  Final result (via
    ``StopIteration.value``) is an int array ``(n_faces,)`` with the number
    of unoccluded candidate directions per face.
    """
    n_faces = len(normals)
    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26  # (F, C)
    vis_count = np.zeros(n_faces, dtype=int)
    epsilon = 0.001
    BATCH = 5

    for j in range(n_candidates):
        cam_dir = candidates[j]
        for i in range(n_faces):
            if not backface_vis[i, j]:
                continue
            origin = centers[i] + normals[i] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                vis_count[i] += 1
        if (j + 1) % BATCH == 0 or j == n_candidates - 1:
            yield (j + 1) / n_candidates

    return vis_count


def _occ_fom_generator(normals, areas, centers, bvh_trees,
                       max_cameras, coverage_target, n_candidates=200,
                       existing_dirs=None):
    """Generator: Full Occlusion Matrix approach.

    Yields progress floats in [0, 1].  Final result is available via
    ``StopIteration.value`` as ``(directions, coverage)``.
    """
    total_area = areas.sum()
    if total_area <= 0:
        yield 1.0
        return [], 0.0

    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26
    n_faces = len(normals)
    vis = np.zeros((n_faces, n_candidates), dtype=bool)
    epsilon = 0.001
    BATCH = 5

    for j in range(n_candidates):
        cam_dir = candidates[j]
        for i in range(n_faces):
            if not backface_vis[i, j]:
                continue
            origin = centers[i] + normals[i] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                vis[i, j] = True
        if (j + 1) % BATCH == 0 or j == n_candidates - 1:
            yield (j + 1) / n_candidates

    return _greedy_select_from_visibility(
        vis, areas, max_cameras, coverage_target, candidates,
        existing_dirs=existing_dirs, normals=normals)


def _occ_tpr_generator(normals, areas, centers, bvh_trees,
                       max_cameras, coverage_target, n_candidates=200,
                       existing_dirs=None):
    """Generator: Two-Pass Refinement approach.

    Yields progress floats in [0, 1].  Final result is available via
    ``StopIteration.value`` as ``(directions, coverage)``.
    """
    total_area = areas.sum()
    if total_area <= 0:
        yield 1.0
        return [], 0.0

    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26

    # ── Pass 1: fast greedy (back-face only, instant) ─────────────────
    covered_bf = np.zeros(len(areas), dtype=bool)
    # Pre-seed coverage from existing cameras
    if existing_dirs:
        for edir in existing_dirs:
            covered_bf |= normals @ np.asarray(edir, dtype=float) > 0.26
    selected_indices = []

    for _ in range(max_cameras):
        uncov = ~covered_bf
        new_vis = backface_vis & uncov[:, np.newaxis]
        new_areas_arr = (new_vis * areas[:, np.newaxis]).sum(axis=0)
        best = int(np.argmax(new_areas_arr))
        if new_areas_arr[best] < total_area * 0.005:
            break
        selected_indices.append(best)
        covered_bf |= backface_vis[:, best]
        if float(areas[covered_bf].sum() / total_area) >= coverage_target:
            break

    if not selected_indices:
        yield 1.0
        return [], 0.0

    yield 0.0  # pass-1 done

    # ── Phase 2a: BVH validate selected set (0 % → 30 %) ─────────────
    epsilon = 0.001
    true_covered = np.zeros(len(areas), dtype=bool)
    n_sel = len(selected_indices)
    for idx, ci in enumerate(selected_indices):
        cam_dir = candidates[ci]
        for fi in range(len(normals)):
            if not backface_vis[fi, ci]:
                continue
            origin = centers[fi] + normals[fi] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                true_covered[fi] = True
        yield 0.3 * (idx + 1) / n_sel

    # ── Phase 2b: patch phantom-uncovered faces (30 % → 100 %) ───────
    phantom_uncovered = covered_bf & ~true_covered

    if phantom_uncovered.any():
        remaining_budget = max_cameras - len(selected_indices)
        if remaining_budget > 0:
            used_set = set(selected_indices)
            unused_mask = np.array([i not in used_set
                                    for i in range(n_candidates)])
            unused_indices = np.where(unused_mask)[0]

            if len(unused_indices) > 0:
                phantom_indices = np.where(phantom_uncovered)[0]
                ph_normals = normals[phantom_indices]
                ph_centers = centers[phantom_indices]
                ph_areas = areas[phantom_indices]
                ph_bf = ph_normals @ candidates[unused_indices].T > 0.26

                ph_vis = np.zeros_like(ph_bf)
                n_unused = len(unused_indices)
                BATCH = 5
                for j_local, j_global in enumerate(unused_indices):
                    cam_dir = candidates[j_global]
                    for i_local in range(len(phantom_indices)):
                        if not ph_bf[i_local, j_local]:
                            continue
                        origin = (ph_centers[i_local]
                                  + ph_normals[i_local] * epsilon)
                        if not _ray_occluded(bvh_trees, origin,
                                             cam_dir, 1e6):
                            ph_vis[i_local, j_local] = True
                    if (j_local + 1) % BATCH == 0 or j_local == n_unused - 1:
                        yield 0.3 + 0.7 * (j_local + 1) / n_unused

                # Greedy on sub-matrix (instant)
                ph_covered = np.zeros(len(ph_areas), dtype=bool)
                for _ in range(remaining_budget):
                    uncov = ~ph_covered
                    nv = ph_vis & uncov[:, np.newaxis]
                    na = (nv * ph_areas[:, np.newaxis]).sum(axis=0)
                    best_local = int(np.argmax(na))
                    if na[best_local] < total_area * 0.005:
                        break
                    best_global = int(unused_indices[best_local])
                    selected_indices.append(best_global)
                    ph_covered |= ph_vis[:, best_local]
                    for i_local in range(len(phantom_indices)):
                        if ph_vis[i_local, best_local]:
                            true_covered[phantom_indices[i_local]] = True

    yield 1.0  # ensure caller sees 100 %
    selected = [candidates[ci].copy() for ci in selected_indices]
    final_cov = (float(areas[true_covered].sum() / total_area)
                 if true_covered.any() else 0.0)
    return selected, final_cov


def _sort_directions_spatially(directions, ref_direction=None):
    """Sort direction vectors by azimuth angle so cameras progress smoothly
    around the subject.  The camera whose azimuth is closest to
    *ref_direction* (typically the viewport look direction) becomes the first
    entry.  Important for sequential generation mode where each camera needs
    spatial context from the previous one."""
    if len(directions) <= 1:
        return directions
    # Azimuth = atan2(y, x) gives a smooth circular ordering
    angles = [math.atan2(float(d[1]), float(d[0])) for d in directions]
    paired = sorted(zip(angles, directions), key=lambda p: p[0])

    if ref_direction is not None:
        # Use full 3D angular distance (dot product) so elevation is
        # considered when choosing the start camera, not just azimuth.
        ref_v = mathutils.Vector(ref_direction).normalized()
        best_idx = 0
        best_dot = -2.0
        for idx, (_, d) in enumerate(paired):
            dot = mathutils.Vector(d).normalized().dot(ref_v)
            if dot > best_dot:
                best_dot = dot
                best_idx = idx
        # Rotate so closest-to-ref is first
        paired = paired[best_idx:] + paired[:best_idx]

    return [d for _, d in paired]


def _classify_camera_direction(cam_dir_np, ref_front_np):
    """Classify a camera direction into a human-readable view label.

    Uses Option-B scheme: elevation tiers first (top/bottom if >60°),
    then 4 azimuth quadrants (front/right/left/rear, 90° each),
    with 'from above'/'from below' modifiers for 30–60° elevation.

    Parameters
    ----------
    cam_dir_np : array-like, shape (3,)
        Centre-to-camera unit vector (where the camera is placed relative to
        the mesh centre).
    ref_front_np : array-like, shape (3,)
        The reference "front" direction (centre-to-camera direction that
        corresponds to the user's viewport, i.e. Camera_0's neighbourhood).

    Returns
    -------
    str
        A prompt-friendly label such as ``"front view"``,
        ``"right side view, from above"``, ``"top view"``, etc.
    """
    d = np.array(cam_dir_np, dtype=float)
    d /= max(np.linalg.norm(d), 1e-12)

    # --- Elevation (angle above / below the horizontal XY plane) ---
    elevation_rad = math.asin(np.clip(d[2], -1.0, 1.0))
    elevation_deg = math.degrees(elevation_rad)

    if elevation_deg > 60.0:
        return "top view"
    if elevation_deg < -60.0:
        return "bottom view"

    # --- Azimuth relative to ref_front (projected onto XY plane) ---
    ref = np.array(ref_front_np, dtype=float)
    # Project both onto XY
    d_h = np.array([d[0], d[1]], dtype=float)
    r_h = np.array([ref[0], ref[1]], dtype=float)
    d_h_len = np.linalg.norm(d_h)
    r_h_len = np.linalg.norm(r_h)
    if d_h_len < 1e-8 or r_h_len < 1e-8:
        # Nearly vertical – should have been caught by the elevation check,
        # but fall back just in case.
        return "top view" if d[2] >= 0 else "bottom view"
    d_h /= d_h_len
    r_h /= r_h_len

    # Signed angle: positive = counterclockwise from above = to the user's right
    sin_a = r_h[0] * d_h[1] - r_h[1] * d_h[0]
    cos_a = r_h[0] * d_h[0] + r_h[1] * d_h[1]
    azimuth_deg = math.degrees(math.atan2(sin_a, cos_a))

    # Quadrant classification (each 90°)
    abs_az = abs(azimuth_deg)
    if abs_az <= 45.0:
        base = "front view"
    elif abs_az >= 135.0:
        base = "rear view"
    elif azimuth_deg > 0:
        base = "right side view"
    else:
        base = "left side view"

    # Elevation modifier
    if elevation_deg > 30.0:
        return f"{base}, from above"
    if elevation_deg < -30.0:
        return f"{base}, from below"
    return base


def _compute_per_camera_aspect(direction_np, verts_world, center):
    """Compute the silhouette aspect ratio (width / height) from a single
    camera direction.  Returns the aspect ratio as a float."""
    right, up, _d = _camera_basis(direction_np)
    rel = verts_world - center
    proj_r = rel @ right
    proj_u = rel @ up
    w = max(float(proj_r.max() - proj_r.min()), 0.001)
    h = max(float(proj_u.max() - proj_u.min()), 0.001)
    return w / h


def _perspective_aspect(verts_world, cam_pos_np, cam_dir_np):
    """Compute the visible angular aspect ratio (width / height) as seen
    from a perspective camera at *cam_pos_np* looking along *cam_dir_np*."""
    right, up, d = _camera_basis(cam_dir_np)
    forward = -d

    rel = verts_world - cam_pos_np
    depth = rel @ forward  # positive = in front of camera
    depth = np.maximum(depth, 0.001)
    pr = rel @ right
    pu = rel @ up

    angle_r = np.arctan2(pr, depth)
    angle_u = np.arctan2(pu, depth)

    angular_w = float(angle_r.max() - angle_r.min())
    angular_h = float(angle_u.max() - angle_u.min())
    if angular_h < 0.001:
        return 1.0
    return angular_w / angular_h


def _resolution_from_aspect(aspect, total_px, align=8):
    """Compute (res_x, res_y) for a given *aspect* ratio (w/h) keeping
    approximately *total_px* pixels, snapped to *align*."""
    new_x = math.sqrt(total_px * aspect)
    new_y = total_px / new_x
    new_x = max(align, int(round(new_x / align)) * align)
    new_y = max(align, int(round(new_y / align)) * align)
    return new_x, new_y


def _get_resolution_align(context):
    """Return the resolution alignment step: 112 for Qwen with alignment
    enabled, 8 otherwise."""
    if (context.scene.model_architecture.startswith('qwen')
            and getattr(context.scene, 'qwen_rescale_alignment', False)):
        return 112
    return 8


def _apply_auto_aspect(directions_np, context, verts_world):
    """Adjust scene render resolution to match the mesh's average apparent
    aspect ratio across the given camera *directions_np*.  This is the
    'shared' mode — all cameras use the same resolution.
    Keeps total pixel count approximately constant and snaps to alignment step.
    Returns (new_res_x, new_res_y)."""
    center = verts_world.mean(axis=0)
    aspects = [_compute_per_camera_aspect(d, verts_world, center)
               for d in directions_np]
    avg_aspect = float(np.mean(aspects))

    render = context.scene.render
    total_px = render.resolution_x * render.resolution_y
    align = _get_resolution_align(context)
    new_x, new_y = _resolution_from_aspect(avg_aspect, total_px, align=align)
    render.resolution_x = new_x
    render.resolution_y = new_y
    return new_x, new_y


def _store_per_camera_resolution(cam_obj, res_x, res_y):
    """Store per-camera resolution as custom properties on the camera object."""
    cam_obj["sg_res_x"] = res_x
    cam_obj["sg_res_y"] = res_y


def _get_camera_resolution(cam_obj, scene):
    """Return (res_x, res_y) for a camera.  Falls back to scene render
    resolution if no per-camera resolution is stored."""
    if "sg_res_x" in cam_obj and "sg_res_y" in cam_obj:
        return int(cam_obj["sg_res_x"]), int(cam_obj["sg_res_y"])
    return scene.render.resolution_x, scene.render.resolution_y


class _SGCameraResolution:
    """Context manager: temporarily set scene render resolution to a camera's
    per-camera values (if stored), and restore the original on exit.

    Usage::

        with _SGCameraResolution(context, camera_obj):
            bpy.ops.render.render(write_still=True)
    """
    def __init__(self, context, cam_obj):
        self._render = context.scene.render
        self._cam = cam_obj
        self._scene = context.scene
        self._orig_x = self._render.resolution_x
        self._orig_y = self._render.resolution_y

    def __enter__(self):
        rx, ry = _get_camera_resolution(self._cam, self._scene)
        self._render.resolution_x = rx
        self._render.resolution_y = ry
        return self

    def __exit__(self, *exc):
        self._render.resolution_x = self._orig_x
        self._render.resolution_y = self._orig_y
        return False


# ---- Per-camera crop overlay (GPU draw handler) -------------------------

_sg_crop_draw_handle = None


def _sg_draw_crop_overlays():
    """SpaceView3D draw callback: renders a coloured rectangle inside each
    camera's pyramid to visualise the actual (non-square) crop region."""
    context = bpy.context
    scene = context.scene

    # Respect Blender's overlay toggle
    space = context.space_data
    if space and hasattr(space, 'overlay') and not space.overlay.show_overlays:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    for obj in scene.objects:
        if obj.type != 'CAMERA' or 'sg_display_crop' not in obj:
            continue
        res_x = obj.get('sg_res_x', 0)
        res_y = obj.get('sg_res_y', 0)
        if res_x <= 0 or res_y <= 0:
            continue

        cam = obj.data
        # Use whichever side of THIS camera is longer as the reference.
        # The scene is set to a square of max_side, so the frustum is square
        # and each side = half_w = half_h.  The crop rectangle occupies
        # res_x/cam_max × res_y/cam_max of that square.
        cam_max = max(res_x, res_y)

        frame = cam.view_frame(scene=scene)
        corner = frame[0]
        half_w = abs(corner[0])
        half_h = abs(corner[1])
        z_depth = corner[2]
        if half_w < 1e-8 or half_h < 1e-8:
            continue

        sx = res_x / cam_max
        sy = res_y / cam_max

        new_hw = half_w * sx
        new_hh = half_h * sy

        crop_local = [
            mathutils.Vector((+new_hw, +new_hh, z_depth)),
            mathutils.Vector((-new_hw, +new_hh, z_depth)),
            mathutils.Vector((-new_hw, -new_hh, z_depth)),
            mathutils.Vector((+new_hw, -new_hh, z_depth)),
        ]

        wm = obj.matrix_world
        world_pts = [wm @ v for v in crop_local]
        coords = [(v.x, v.y, v.z) for v in world_pts]
        indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        batch = batch_for_shader(shader, 'LINES', {"pos": coords}, indices=indices)
        shader.bind()
        _prefs = bpy.context.preferences.addons.get(__package__)
        _oc = _prefs.preferences.overlay_color if _prefs else (0.3, 0.5, 1.0)
        shader.uniform_float("color", (_oc[0], _oc[1], _oc[2], 0.9))
        gpu.state.line_width_set(2.0)
        gpu.state.blend_set('ALPHA')
        batch.draw(shader)

    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


# ---- Per-camera view-label overlay (GPU text handler) --------------------

_sg_label_draw_handle = None
_sg_labels_user_visible = True


def _sg_draw_view_labels():
    """SpaceView3D POST_PIXEL callback: draws full camera prompt text
    (from ``scene.camera_prompts``) near each camera in the viewport."""
    context = bpy.context
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    # Respect Blender's overlay toggle
    space = context.space_data
    if space and hasattr(space, 'overlay') and not space.overlay.show_overlays:
        return

    from bpy_extras.view3d_utils import location_3d_to_region_2d

    font_id = 0
    blf.size(font_id, 13)
    _prefs = bpy.context.preferences.addons.get(__package__)
    _oc = _prefs.preferences.overlay_color if _prefs else (0.3, 0.5, 1.0)
    blf.color(font_id, _oc[0], _oc[1], _oc[2], 0.95)

    # Build a lookup of camera name -> prompt text
    prompt_lookup = {item.name: item.prompt for item in scene.camera_prompts
                     if item.prompt}

    for obj in scene.objects:
        if obj.type != 'CAMERA':
            continue
        label = prompt_lookup.get(obj.name, '')
        if not label:
            continue
        co_2d = location_3d_to_region_2d(region, rv3d, obj.location)
        if co_2d is None:
            continue
        # Offset slightly below the camera marker
        blf.position(font_id, co_2d.x - blf.dimensions(font_id, label)[0] * 0.5,
                     co_2d.y - 20, 0)
        gpu.state.blend_set('ALPHA')
        blf.draw(font_id, label)

    gpu.state.blend_set('NONE')


def _sg_ensure_crop_overlay():
    """Register the crop overlay draw handler if not already active."""
    global _sg_crop_draw_handle
    if _sg_crop_draw_handle is None:
        _sg_crop_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _sg_draw_crop_overlays, (), 'WINDOW', 'POST_VIEW')


def _sg_remove_crop_overlay():
    """Remove the crop overlay draw handler if active."""
    global _sg_crop_draw_handle
    if _sg_crop_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_sg_crop_draw_handle, 'WINDOW')
        _sg_crop_draw_handle = None


def _sg_ensure_label_overlay():
    """Register the floating prompt-text label overlay if not already active."""
    global _sg_label_draw_handle, _sg_labels_user_visible
    _sg_labels_user_visible = True
    if _sg_label_draw_handle is None:
        _sg_label_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _sg_draw_view_labels, (), 'WINDOW', 'POST_PIXEL')
        # Redraw all viewports so labels appear immediately
        for area in (bpy.context.screen.areas if bpy.context.screen else []):
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sg_remove_label_overlay():
    """Remove the floating prompt-text label overlay if active."""
    global _sg_label_draw_handle, _sg_labels_user_visible
    _sg_labels_user_visible = False
    if _sg_label_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_sg_label_draw_handle, 'WINDOW')
        _sg_label_draw_handle = None
        for area in (bpy.context.screen.areas if bpy.context.screen else []):
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sg_hide_label_overlay():
    """Temporarily hide the label overlay without changing the user's preference."""
    global _sg_label_draw_handle
    if _sg_label_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_sg_label_draw_handle, 'WINDOW')
        _sg_label_draw_handle = None
        for area in (bpy.context.screen.areas if bpy.context.screen else []):
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sg_restore_label_overlay():
    """Restore the label overlay only if the user had it enabled."""
    if _sg_labels_user_visible:
        global _sg_label_draw_handle
        if _sg_label_draw_handle is None:
            _sg_label_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                _sg_draw_view_labels, (), 'WINDOW', 'POST_PIXEL')
            for area in (bpy.context.screen.areas if bpy.context.screen else []):
                if area.type == 'VIEW_3D':
                    area.tag_redraw()


def _sg_restore_square_display(scene):
    """Restore scene resolution to the max-side square and tag viewports
    for redraw so crop overlays reappear correctly."""
    cameras = [o for o in scene.objects if o.type == 'CAMERA' and 'sg_res_x' in o]
    if not cameras:
        return
    max_side = max(
        max(int(c.get('sg_res_x', 0)), int(c.get('sg_res_y', 0)))
        for c in cameras
    )
    if max_side > 0:
        scene.render.resolution_x = max_side
        scene.render.resolution_y = max_side
    # Refresh viewports
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _setup_square_camera_display(cam_obj, res_x, res_y):
    """Mark a camera for the crop overlay and enable passepartout."""
    cam_data = cam_obj.data
    cam_data.show_passepartout = True
    cam_data.passepartout_alpha = 0.5
    cam_obj["sg_display_crop"] = True
    # Ensure the draw handler is alive
    _sg_ensure_crop_overlay()


def _get_fov(cam_settings, context, res_x=None, res_y=None):
    """Return (fov_x, fov_y) in radians for the given camera data block.
    If *res_x* / *res_y* are not provided, reads from scene render settings."""
    fov_x = cam_settings.angle_x
    if res_x is None:
        res_x = context.scene.render.resolution_x
    if res_y is None:
        res_y = context.scene.render.resolution_y
    if res_y > res_x:
        fov_x = 2 * math.atan(math.tan(fov_x / 2) * res_x / res_y)
    fov_y = 2 * math.atan(math.tan(fov_x / 2) * res_y / res_x)
    return fov_x, fov_y


class AddCameras(bpy.types.Operator):
    """Add cameras using various placement strategies and adjust their positions
    
    Uses the active camera as a reference for settings. If there is no active camera, a new one is created based on the viewport.
    
    Placement modes:
    - Orbit Ring: cameras in a circle (inherits elevation from reference camera)
    - Sphere Coverage: Fibonacci spiral for even sphere distribution
    - Auto (Normal-Weighted): K-means clustering of mesh face normals
    - Auto (PCA Axes): cameras along the mesh's principal component axes
    - Auto (Greedy Coverage): iteratively adds cameras that maximise new visible surface area and auto-determines the count
    - Fan from Camera: arc of cameras near the active camera
    
    Tips: 
    - Try to frame the object / scene with minimal margin around it.
    - Aim to achieve a uniform coverage of the object / scene.
    - Areas not visible from any camera won't get textured. (Can still be UV-inpainted)
    - Aspect ratio is set by Blender's output settings (or auto-computed)."""
    bl_category = "ControlNet"
    bl_idname = "object.add_cameras"
    bl_label = "Add Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    placement_mode: bpy.props.EnumProperty(
        name="Placement Mode",
        description="Strategy for placing cameras around the subject",
        items=[
            ('orbit_ring', "Orbit Ring", "Place cameras in a circle around the center (original behaviour). Inherits elevation from the reference camera or viewport"),
            ('hemisphere', "Sphere Coverage", "Distribute cameras evenly across a sphere using a Fibonacci spiral"),
            ('normal_weighted', "Auto (Normal-Weighted)", "Automatically place cameras to cover the most surface area, using K-means on face normals weighted by area"),
            ('pca_axes', "Auto (PCA Axes)", "Place cameras along the mesh's principal axes of variation"),
            ('greedy_coverage', "Auto (Greedy Coverage)", "Iteratively add cameras that maximise new visible surface. Automatically determines the number of cameras needed"),
            ('fan_from_camera', "Fan from Camera", "Spread cameras in an arc around the active camera's orbit position"),
        ],
        default='normal_weighted'
    ) # type: ignore

    num_cameras: bpy.props.IntProperty(
        name="Number of Cameras",
        description="Number of cameras to add (not used by Greedy Coverage which auto-determines count)",
        default=4,
        min=1,
        max=100
    ) # type: ignore

    center_type: bpy.props.EnumProperty(
        name="Center Type",
        description="Type of center for the cameras",
        items=[
            ('object', "Object", "Use the active object as the center"),
            ('view center', "View Center", "Use the view center as the center"),
        ],
        default='object'
    ) # type: ignore

    purge_others: bpy.props.BoolProperty(
        name="Remove Existing Cameras",
        description="Delete ALL existing cameras (including active) before adding new ones",
        default=True
    ) # type: ignore

    consider_existing: bpy.props.BoolProperty(
        name="Consider Existing Cameras",
        description="Treat existing cameras as already-placed directions so auto modes avoid duplicating their coverage",
        default=True
    ) # type: ignore

    fan_angle: bpy.props.FloatProperty(
        name="Fan Angle",
        description="Total angular spread of the fan in degrees",
        default=90.0,
        min=10.0,
        max=350.0
    ) # type: ignore

    coverage_target: bpy.props.FloatProperty(
        name="Coverage Target",
        description="Stop adding cameras when this fraction of surface area is visible (Greedy mode)",
        default=0.95,
        min=0.5,
        max=1.0,
        subtype='FACTOR'
    ) # type: ignore

    max_auto_cameras: bpy.props.IntProperty(
        name="Max Cameras",
        description="Upper limit on cameras for Greedy Coverage mode",
        default=12,
        min=2,
        max=50
    ) # type: ignore

    auto_aspect: bpy.props.EnumProperty(
        name="Auto Aspect Ratio",
        description="Automatically adjust render aspect ratio to match the mesh silhouette",
        items=[
            ('off', "Off", "Use current scene resolution for all cameras"),
            ('shared', "Shared", "Average silhouette aspect across all cameras and set a single scene resolution"),
            ('per_camera', "Per Camera", "Each camera gets its own optimal aspect ratio (resolution is swapped during generation)"),
        ],
        default='per_camera'
    ) # type: ignore

    exclude_bottom: bpy.props.BoolProperty(
        name="Exclude Bottom Faces",
        description="Ignore downward-facing geometry (e.g. flat building undersides) when placing cameras",
        default=True
    ) # type: ignore

    exclude_bottom_angle: bpy.props.FloatProperty(
        name="Bottom Angle Threshold",
        description="Faces whose normal points more than this many degrees below the horizon are excluded",
        default=1.5533,  # 89 degrees in radians
        min=0.1745,       # 10 degrees
        max=1.5708,       # 90 degrees
        subtype='ANGLE',
        unit='ROTATION'
    ) # type: ignore

    auto_prompts: bpy.props.BoolProperty(
        name="Auto View Prompts",
        description="Automatically generate view-direction prompts (e.g. 'front view', 'rear view, from above') for each camera based on the viewport reference orientation",
        default=False
    ) # type: ignore

    review_placement: bpy.props.BoolProperty(
        name="Review Camera Placement",
        description="After placing cameras, fly through each one for review. "
                    "When disabled the cameras are created immediately without the interactive fly-through",
        default=True
    ) # type: ignore

    occlusion_mode: bpy.props.EnumProperty(
        name="Occlusion Handling",
        description="How to account for self-occlusion when choosing camera directions",
        items=[
            ('none', "None (Fast)",
             "Back-face culling only – ignores self-occlusion. Fastest option"),
            ('full_matrix', "Full Occlusion Matrix",
             "Build a complete BVH-validated visibility matrix before greedy selection. Most accurate but slower"),
            ('two_pass', "Two-Pass Refinement",
             "Fast back-face pass, then targeted BVH refinement only for faces with zero true coverage"),
            ('vis_weighted', "Visibility-Weighted",
             "Weight faces by their visibility fraction from 200 directions. "
             "Mostly-occluded faces have reduced influence on camera placement (linear). "
             "Only affects Normal-Weighted mode; other modes fall back to Full Occlusion Matrix"),
            ('vis_interactive', "Interactive Visibility",
             "Like Visibility-Weighted but with a real-time preview: scroll to adjust "
             "the occlusion balance and see cameras reposition instantly. "
             "Only affects Normal-Weighted mode; other modes fall back to Full Occlusion Matrix"),
        ],
        default='none'
    ) # type: ignore

    clamp_elevation: bpy.props.BoolProperty(
        name="Clamp Elevation",
        description="Restrict camera elevation to avoid extreme top-down or bottom-up views "
                    "that diffusion models often struggle with",
        default=False
    ) # type: ignore

    max_elevation_angle: bpy.props.FloatProperty(
        name="Max Elevation",
        description="Maximum upward elevation angle (degrees above horizon). "
                    "Cameras looking further up will be clamped to this angle",
        default=1.2217,  # 70 degrees in radians
        min=0.0,
        max=1.5708,      # 90 degrees
        subtype='ANGLE',
        unit='ROTATION'
    ) # type: ignore

    min_elevation_angle: bpy.props.FloatProperty(
        name="Min Elevation",
        description="Minimum downward elevation angle (degrees below horizon). "
                    "Cameras looking further down will be clamped to this angle",
        default=-0.1745,  # -10 degrees in radians
        min=-1.5708,      # -90 degrees
        max=0.0,
        subtype='ANGLE',
        unit='ROTATION'
    ) # type: ignore

    _timer = None
    _camera_index = 0
    _cameras = []
    _initial_camera = None
    _draw_handle = None
    # Occlusion modal state
    _occ_phase = False
    _occ_gen = None
    _occ_progress = 0.0
    _occ_state = None
    # Interactive visibility preview state
    _vis_preview_phase = False
    _vis_count = None
    _vis_n_candidates = 200
    _vis_balance = 0.2
    _vis_state = None
    _vis_directions = None

    def draw(self, context):
        """Custom dialog layout for the placement mode selector."""
        layout = self.layout
        layout.prop(self, "placement_mode")
        layout.separator()
        is_auto = self.placement_mode in ('hemisphere', 'normal_weighted', 'pca_axes', 'greedy_coverage')
        if self.placement_mode == 'greedy_coverage':
            layout.prop(self, "coverage_target")
            layout.prop(self, "max_auto_cameras")
        else:
            layout.prop(self, "num_cameras")
        if not is_auto:
            layout.prop(self, "center_type")
        if self.placement_mode == 'fan_from_camera':
            layout.prop(self, "fan_angle")
        if is_auto:
            if self.placement_mode != 'hemisphere':
                layout.prop(self, "occlusion_mode")
            layout.prop(self, "auto_aspect")
            layout.prop(self, "exclude_bottom")
            if self.exclude_bottom:
                layout.prop(self, "exclude_bottom_angle")
            layout.prop(self, "auto_prompts")
            layout.prop(self, "clamp_elevation")
            if self.clamp_elevation:
                row = layout.row(align=True)
                row.prop(self, "min_elevation_angle", text="Min")
                row.prop(self, "max_elevation_angle", text="Max")
        layout.prop(self, "review_placement")
        layout.prop(self, "purge_others")
        if not self.purge_others and is_auto:
            layout.prop(self, "consider_existing")

    def draw_callback(self, context):
        # ── Occlusion progress display ───────────────────────────────────
        if self._occ_phase:
            font_id = 0
            region = context.region
            rw, rh = region.width, region.height
            pct = self._occ_progress

            # Progress bar background (dark)
            bar_w, bar_h = 300, 18
            bar_x = (rw - bar_w) / 2
            bar_y = rh * 0.10

            # Draw progress text above bar
            msg = f"Computing occlusion… {pct * 100:.0f}%"
            blf.size(font_id, 18)
            tw, _th = blf.dimensions(font_id, msg)
            blf.position(font_id, (rw - tw) / 2, bar_y + bar_h + 6, 0)
            blf.color(font_id, 1.0, 0.85, 0.35, 0.95)
            blf.draw(font_id, msg)

            hint = "Press ESC to cancel"
            blf.size(font_id, 13)
            hw, _hh = blf.dimensions(font_id, hint)
            blf.position(font_id, (rw - hw) / 2, bar_y - 18, 0)
            blf.color(font_id, 0.7, 0.7, 0.7, 0.8)
            blf.draw(font_id, hint)
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            return

        # ── Interactive visibility preview HUD ───────────────────────────
        if self._vis_preview_phase:
            font_id = 0
            region = context.region
            rw, rh = region.width, region.height
            balance = self._vis_balance

            msg = f"Occlusion Balance: {balance:.0%}"
            blf.size(font_id, 22)
            tw, _th = blf.dimensions(font_id, msg)
            blf.position(font_id, (rw - tw) / 2, rh * 0.13, 0)
            blf.color(font_id, 1.0, 0.85, 0.35, 0.95)
            blf.draw(font_id, msg)

            n_cams = len(self._cameras) if self._cameras else 0
            info = f"{n_cams} cameras"
            blf.size(font_id, 16)
            iw, _ih = blf.dimensions(font_id, info)
            blf.position(font_id, (rw - iw) / 2, rh * 0.13 - 26, 0)
            blf.color(font_id, 0.9, 0.9, 0.9, 0.9)
            blf.draw(font_id, info)

            hint = "Scroll \u2191\u2193 to adjust  |  ENTER to confirm  |  ESC to cancel"
            blf.size(font_id, 13)
            hw, _hh = blf.dimensions(font_id, hint)
            blf.position(font_id, (rw - hw) / 2, rh * 0.13 - 48, 0)
            blf.color(font_id, 0.7, 0.7, 0.7, 0.8)
            blf.draw(font_id, hint)
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            return

        # ── Fly-through HUD ──────────────────────────────────────────────
        try:
            count = len(self._cameras)
            idx = self._camera_index
        except Exception:
            return
        if count == 0:
            return
        font_id = 0
        region = context.region
        rw, rh = region.width, region.height

        msg = f"Camera: {idx+1}/{count}  |  Press SPACE to confirm"
        blf.size(font_id, 20)
        text_width, text_height = blf.dimensions(font_id, msg)
        x = (rw - text_width) / 2
        y = rh * 0.10
        blf.position(font_id, x, y, 0)
        blf.draw(font_id, msg)

        # Show auto-generated view label below the main HUD line
        if idx < count:
            cam_obj = self._cameras[idx]
            view_label = cam_obj.get('sg_view_label', '')
            if view_label:
                blf.size(font_id, 16)
                blf.color(font_id, 1.0, 0.85, 0.35, 0.95)
                lw, _lh = blf.dimensions(font_id, view_label)
                blf.position(font_id, (rw - lw) / 2, y - 28, 0)
                blf.draw(font_id, view_label)
                blf.color(font_id, 1.0, 1.0, 1.0, 1.0)

    def execute(self, context):
        # --- Delete existing cameras if requested ---
        if self.purge_others:
            scene = context.scene
            to_remove = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
            for cam in to_remove:
                for col in list(cam.users_collection):
                    col.objects.unlink(cam)
                bpy.data.objects.remove(cam, do_unlink=True)
            for cam_data in list(bpy.data.cameras):
                if not cam_data.users:
                    bpy.data.cameras.remove(cam_data)
            scene.camera = None

        # --- Validate mesh requirement for mesh-based modes ---
        if self.placement_mode in ('hemisphere', 'normal_weighted', 'pca_axes', 'greedy_coverage'):
            target_meshes = _gather_target_meshes(context)
            if not target_meshes:
                self.report({'ERROR'}, "No mesh objects found for this placement mode. Select meshes or ensure the scene has meshes.")
                return {'CANCELLED'}
            total_faces = sum(len(o.data.polygons) for o in target_meshes)
            if total_faces == 0:
                self.report({'ERROR'}, "Target meshes have no faces.")
                return {'CANCELLED'}

        # --- Fallback for center_type ---
        if self.center_type == 'object':
            obj = context.object
            if not obj:
                self.report({'WARNING'}, "No active object found. Using view center instead.")
                self.center_type = 'view center'

        # --- Add draw handler ---
        if AddCameras._draw_handle is None:
            AddCameras._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback, (context,), 'WINDOW', 'POST_PIXEL')

        # --- Determine if this is a "manual" mode (needs ref camera for orbit)
        #     or an "auto" mode (computes its own positions) ---
        is_auto_mode = self.placement_mode in ('hemisphere', 'normal_weighted', 'pca_axes', 'greedy_coverage')

        # --- Determine center location ---
        # Auto modes compute their own center from mesh vertices later, so
        # only the manual modes (orbit_ring, fan_from_camera) need this.
        if is_auto_mode:
            # Placeholder – auto branch overrides center_location from verts
            center_location = context.region_data.view_location.copy()
        elif self.center_type == 'object':
            obj = context.object
            cursor_loc = context.scene.cursor.location.copy()
            context.scene.cursor.location = obj.location.copy()
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            center_location = obj.location.copy()
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            context.scene.cursor.location = cursor_loc
            center_location = obj.matrix_world.translation + obj.matrix_world.to_3x3() @ (center_location - obj.matrix_world.translation)
        else:
            center_location = context.region_data.view_location.copy()

        # --- Set or create reference camera ---
        self._initial_camera = context.scene.camera
        using_viewport_ref = False
        if not self._initial_camera and not is_auto_mode:
            # Manual modes: create a camera from the viewport as Camera_0
            rv3d = context.region_data
            cam_data = bpy.data.cameras.new(name='Camera_0')
            cam_obj = bpy.data.objects.new('Camera_0', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.matrix_world = rv3d.view_matrix.inverted()
            context.scene.camera = cam_obj
            self._initial_camera = cam_obj
            using_viewport_ref = True

        # --- Capture reference camera settings (for lens/sensor/clip copy) ---
        cam_settings = self._initial_camera.data if self._initial_camera else None

        self._cameras.clear()
        self._camera_index = 0

        # --- Branch by placement mode ---
        if self.placement_mode == 'orbit_ring':
            ref_mat = self._initial_camera.matrix_world.copy()
            initial_pos = self._initial_camera.location.copy()
            radius = (initial_pos - center_location).length
            if radius < 0.001:
                radius = 5.0
            self._cameras.append(self._initial_camera)
            self._place_orbit_ring(context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref)
        elif self.placement_mode == 'fan_from_camera':
            ref_mat = self._initial_camera.matrix_world.copy()
            initial_pos = self._initial_camera.location.copy()
            radius = (initial_pos - center_location).length
            if radius < 0.001:
                radius = 5.0
            self._cameras.append(self._initial_camera)
            self._place_fan(context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref)
        else:
            # ============================================================
            # Auto modes: per-camera optimal distance from mesh silhouette
            # ============================================================
            # target_meshes was validated above

            # Borrow settings from existing camera or create temp defaults
            temp_cam_data = None
            if not cam_settings:
                temp_cam_data = bpy.data.cameras.new(name='_sg_temp_cam')
                cam_settings = temp_cam_data

            # Pre-compute combined mesh data across all target meshes
            verts_world = _get_mesh_verts_world(target_meshes)
            mesh_center = verts_world.mean(axis=0)
            center_location = mathutils.Vector(mesh_center.tolist())
            mesh_names = ', '.join(o.name for o in target_meshes)
            self.report({'INFO'}, f"Target meshes ({len(target_meshes)}): {mesh_names}")

            # --- Collect existing camera directions (for "consider existing") ---
            existing_dirs = None
            if not self.purge_others and self.consider_existing:
                existing_dirs = _existing_camera_directions(mesh_center)
                if existing_dirs:
                    self.report({'INFO'},
                        f"Considering {len(existing_dirs)} existing camera(s)")

            # --- Determine directions ---
            if self.placement_mode == 'greedy_coverage':
                normals, areas, centers = _get_mesh_face_data(target_meshes)
                if self.exclude_bottom:
                    normals, areas, centers = _filter_bottom_faces(
                        normals, areas, centers, self.exclude_bottom_angle)

                occ = self.occlusion_mode
                if occ == 'none':
                    directions, coverage = _greedy_coverage_directions(
                        normals, areas,
                        max_cameras=self.max_auto_cameras,
                        coverage_target=self.coverage_target,
                        existing_dirs=existing_dirs)
                    directions = [np.array(d) for d in directions]
                    self.report({'INFO'},
                        f"Greedy coverage (no occlusion): "
                        f"{len(directions)} cameras, {coverage*100:.1f}% coverage")
                else:
                    # Greedy doesn't use K-means; vis modes fall back to FOM
                    eff_occ = occ if occ in ('full_matrix', 'two_pass') else 'full_matrix'
                    # ── Async occlusion via modal ────────────────────────
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    gen_func = (_occ_fom_generator if eff_occ == 'full_matrix'
                                else _occ_tpr_generator)
                    self._occ_gen = gen_func(
                        normals, areas, centers, bvh_trees,
                        max_cameras=self.max_auto_cameras,
                        coverage_target=self.coverage_target,
                        existing_dirs=existing_dirs)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    self._occ_state = {
                        'result_type': 'greedy',
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'occ_label': ('full occlusion' if occ == 'full_matrix'
                                      else 'two-pass'),
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
            elif self.placement_mode == 'hemisphere':
                points = _fibonacci_sphere_points(self.num_cameras)
                directions = [np.array(p) for p in points]
                if self.exclude_bottom:
                    # Remove directions that point more than the threshold below the horizon
                    threshold_z = -math.cos(self.exclude_bottom_angle)
                    directions = [d for d in directions if d[2] >= threshold_z]
                if existing_dirs:
                    directions = _filter_near_existing(directions, existing_dirs)
            elif self.placement_mode == 'normal_weighted':
                normals, areas, centers = _get_mesh_face_data(target_meshes)
                if self.exclude_bottom:
                    normals, areas, centers = _filter_bottom_faces(
                        normals, areas, centers, self.exclude_bottom_angle)

                occ = self.occlusion_mode
                if occ in ('vis_weighted', 'vis_interactive'):
                    # Visibility-count generator (shared by both modes)
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    self._occ_gen = _occ_vis_count_generator(
                        normals, areas, centers, bvh_trees)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    rt = 'vis_kmeans' if occ == 'vis_weighted' else 'vis_interactive'
                    self._occ_state = {
                        'result_type': rt,
                        'normals': normals,
                        'areas': areas,
                        'centers': centers,
                        'num_cameras': self.num_cameras,
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'existing_dirs': existing_dirs,
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
                elif occ != 'none':
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    self._occ_gen = _occ_filter_faces_generator(
                        normals, areas, centers, bvh_trees)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    self._occ_state = {
                        'result_type': 'filter_kmeans',
                        'normals': normals,
                        'areas': areas,
                        'centers': centers,
                        'num_cameras': self.num_cameras,
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'existing_dirs': existing_dirs,
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
                else:
                    k = min(self.num_cameras, len(normals))
                    cluster_dirs = _kmeans_on_sphere(normals, areas, k)
                    directions = [cluster_dirs[i] for i in range(len(cluster_dirs))]
                    if existing_dirs:
                        directions = _filter_near_existing(directions, existing_dirs)
            elif self.placement_mode == 'pca_axes':
                normals, areas, centers = _get_mesh_face_data(target_meshes)
                if self.exclude_bottom:
                    normals, areas, centers = _filter_bottom_faces(
                        normals, areas, centers, self.exclude_bottom_angle)

                occ = self.occlusion_mode
                # PCA doesn't use K-means; vis modes fall back to filter
                if occ not in ('none',):
                    depsgraph = context.evaluated_depsgraph_get()
                    bvh_trees = _build_bvh_trees(target_meshes, depsgraph)
                    self._occ_gen = _occ_filter_faces_generator(
                        normals, areas, centers, bvh_trees)
                    self._occ_phase = True
                    self._occ_progress = 0.0
                    self._occ_state = {
                        'result_type': 'filter_pca',
                        'normals': normals,
                        'areas': areas,
                        'centers': centers,
                        'num_cameras': self.num_cameras,
                        'exclude_bottom': self.exclude_bottom,
                        'exclude_bottom_angle': self.exclude_bottom_angle,
                        'verts_world': verts_world,
                        'mesh_center': mesh_center,
                        'cam_settings': cam_settings,
                        'temp_cam_data': temp_cam_data,
                        'existing_dirs': existing_dirs,
                    }
                    context.window_manager.modal_handler_add(self)
                    self._timer = context.window_manager.event_timer_add(
                        0.01, window=context.window)
                    return {'RUNNING_MODAL'}
                else:
                    axes = _compute_pca_axes(verts_world)
                    directions = []
                    for axis in axes:
                        directions.append(axis)
                        directions.append(-axis)
                    if self.exclude_bottom:
                        threshold_z = -math.cos(self.exclude_bottom_angle)
                        directions = [d for d in directions if d[2] >= threshold_z]
                    directions = directions[:min(self.num_cameras, len(directions))]
                    if existing_dirs:
                        directions = _filter_near_existing(directions, existing_dirs)

            # Finalize: sort, aspect ratio, camera creation, auto prompts
            self._finalize_auto_cameras(
                context, directions, verts_world, mesh_center,
                cam_settings, temp_cam_data)

        # --- Start fly-through review ---
        if self.review_placement:
            if not self._start_fly_review(context):
                return {'CANCELLED'}
            return {'RUNNING_MODAL'}
        else:
            # Skip review — just finish immediately
            self._finish_without_review(context)
            return {'FINISHED'}

    # -------------------------------------------------------
    # Auto-mode finalization helpers
    # -------------------------------------------------------

    def _finalize_auto_cameras(self, context, directions, verts_world,
                               mesh_center, cam_settings, temp_cam_data):
        """Sort directions, compute aspect ratios, create cameras, and
        generate auto prompts.  Shared by both the synchronous (no-occlusion)
        and asynchronous (modal occlusion) code paths."""
        ref_dir = None
        if directions:
            rv3d = context.region_data
            if rv3d:
                view_dir = rv3d.view_rotation @ mathutils.Vector((0, 0, -1))
                ref_dir = np.array([-view_dir.x, -view_dir.y, -view_dir.z])
            directions = _sort_directions_spatially(directions, ref_dir)

        # --- Clamp elevation angles to avoid extreme top/bottom views ---
        if self.clamp_elevation and directions:
            min_rad = self.min_elevation_angle
            max_rad = self.max_elevation_angle
            clamped = []
            for d in directions:
                d_np = np.array(d, dtype=float)
                norm = np.linalg.norm(d_np)
                if norm < 1e-8:
                    continue
                d_np /= norm
                elev = math.asin(float(np.clip(d_np[2], -1.0, 1.0)))
                if min_rad <= elev <= max_rad:
                    clamped.append(d_np)
                    continue
                elev = max(min_rad, min(max_rad, elev))
                horiz = math.sqrt(float(d_np[0]**2 + d_np[1]**2))
                if horiz < 1e-8:
                    # Pure vertical direction – pick an arbitrary azimuth
                    azimuth = 0.0
                else:
                    azimuth = math.atan2(float(d_np[1]), float(d_np[0]))
                cos_e = math.cos(elev)
                d_np = np.array([cos_e * math.cos(azimuth),
                                 cos_e * math.sin(azimuth),
                                 math.sin(elev)])
                clamped.append(d_np)

            # Deduplicate near-identical directions that may have been created
            # by clamping several cameras to the same elevation limit.
            _MERGE_COS = math.cos(math.radians(8.0))  # merge within 8°
            deduped = []
            for d in clamped:
                if all(float(np.dot(d, existing)) < _MERGE_COS
                       for existing in deduped):
                    deduped.append(d)
            n_before = len(directions)
            directions = deduped
            if len(directions) < n_before:
                self.report({'INFO'},
                    f"Elevation clamp: {n_before} → {len(directions)} cameras "
                    f"(merged {n_before - len(directions)} near-duplicates)")

        render = context.scene.render
        total_px = render.resolution_x * render.resolution_y
        center_np_for_aspect = verts_world.mean(axis=0)

        if self.auto_aspect == 'shared' and directions:
            dirs_np = [d / np.linalg.norm(d) for d in directions]
            old_x, old_y = render.resolution_x, render.resolution_y
            new_x, new_y = _apply_auto_aspect(dirs_np, context, verts_world)
            if (new_x, new_y) != (old_x, old_y):
                self.report({'INFO'},
                    f"Aspect ratio adjusted: {old_x}x{old_y} → {new_x}x{new_y}")

        if self.auto_aspect == 'per_camera' and directions:
            self._create_cameras_per_aspect(
                context, directions, mesh_center, verts_world,
                cam_settings, total_px, center_np_for_aspect)
        else:
            fov_x, fov_y = _get_fov(cam_settings, context)
            self._create_cameras_from_directions(
                context, directions, mesh_center, verts_world,
                cam_settings, fov_x, fov_y)

        if temp_cam_data:
            bpy.data.cameras.remove(temp_cam_data)

        if self.auto_prompts and self._cameras:
            ref_front = (ref_dir if ref_dir is not None
                         else np.array([0.0, 1.0, 0.0]))
            mesh_center_np = np.array(mesh_center, dtype=float)
            for cam_obj in self._cameras:
                cam_pos = np.array(cam_obj.location, dtype=float)
                cam_dir = cam_pos - mesh_center_np
                label = _classify_camera_direction(cam_dir, ref_front)
                cam_obj["sg_view_label"] = label
                prompt_item = next(
                    (item for item in context.scene.camera_prompts
                     if item.name == cam_obj.name), None)
                if not prompt_item:
                    prompt_item = context.scene.camera_prompts.add()
                    prompt_item.name = cam_obj.name
                prompt_item.prompt = label
            context.scene.use_camera_prompts = True
            self.report({'INFO'},
                        f"Auto-prompts: assigned view labels to "
                        f"{len(self._cameras)} cameras")
        elif not self.auto_prompts:
            # Clear any pre-existing auto-prompts so stale labels don't persist
            context.scene.camera_prompts.clear()
            context.scene.use_camera_prompts = False
            for cam_obj in self._cameras:
                if 'sg_view_label' in cam_obj:
                    del cam_obj['sg_view_label']
            _sg_remove_label_overlay()

    def _update_vis_cameras(self, context):
        """Regenerate cameras based on current visibility balance setting."""
        # Remove existing preview cameras
        for cam in list(self._cameras):
            cam_data = cam.data
            bpy.data.objects.remove(cam, do_unlink=True)
            if cam_data and not cam_data.users:
                bpy.data.cameras.remove(cam_data)
        self._cameras.clear()

        state = self._vis_state
        vis_count = self._vis_count
        n_cand = self._vis_n_candidates
        vis_fraction = vis_count.astype(float) / n_cand
        balance = self._vis_balance

        exterior = vis_count > 0
        normals = state['normals'][exterior]
        areas_base = state['areas'][exterior]
        vf = vis_fraction[exterior]
        # weight = area × (vis_fraction + balance × (1 - vis_fraction))
        # balance=0: weight = area × vis_fraction (full vis-weighting)
        # balance=1: weight = area (no vis-weighting, all exterior equal)
        weighted_areas = areas_base * (vf + balance * (1.0 - vf))

        existing_dirs = state.get('existing_dirs')
        k = min(state['num_cameras'], len(normals))
        if k > 0 and len(normals) > 0:
            cluster_dirs = _kmeans_on_sphere(normals, weighted_areas, k)
            directions = [cluster_dirs[i] for i in range(len(cluster_dirs))]
            if existing_dirs:
                directions = _filter_near_existing(directions, existing_dirs)
        else:
            directions = []

        # Sort directions
        ref_dir = None
        rv3d = context.region_data
        if rv3d:
            view_dir = rv3d.view_rotation @ mathutils.Vector((0, 0, -1))
            ref_dir = np.array([-view_dir.x, -view_dir.y, -view_dir.z])
        directions = _sort_directions_spatially(directions, ref_dir)
        self._vis_directions = directions

        # Quick camera creation (without full finalize overhead)
        if directions:
            cam_settings = state['cam_settings']
            fov_x, fov_y = _get_fov(cam_settings, context)
            self._create_cameras_from_directions(
                context, directions, state['mesh_center'],
                state['verts_world'], cam_settings, fov_x, fov_y)
            if self._cameras:
                context.scene.camera = self._cameras[0]

        n_ext = int(exterior.sum())
        context.area.header_text_set(
            f"Occlusion Balance: {balance:.0%}  |  "
            f"{len(directions)} cameras, {n_ext} visible faces  |  "
            f"Scroll to adjust  |  ENTER to confirm  |  ESC to cancel")

    def _cleanup_vis_preview(self, context):
        """Clean up interactive visibility preview state."""
        # Delete preview cameras
        for cam in list(self._cameras):
            cam_data = cam.data
            bpy.data.objects.remove(cam, do_unlink=True)
            if cam_data and not cam_data.users:
                bpy.data.cameras.remove(cam_data)
        self._cameras.clear()

        # Clean up temp camera data
        state = self._vis_state
        if state and state.get('temp_cam_data'):
            bpy.data.cameras.remove(state['temp_cam_data'])

        self._vis_preview_phase = False
        self._vis_state = None
        self._vis_count = None
        context.area.header_text_set(None)
        if AddCameras._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(
                AddCameras._draw_handle, 'WINDOW')
            AddCameras._draw_handle = None

    def _start_fly_review(self, context, add_modal_handler=True):
        """Frame the first camera and start fly-through review.

        *add_modal_handler* should be False when called from an already-running
        modal (e.g. after occlusion computation finishes).
        Returns True if fly-through started, False if no cameras were created.
        """
        if not self._cameras:
            self.report({'WARNING'}, "No cameras were created.")
            if AddCameras._draw_handle:
                bpy.types.SpaceView3D.draw_handler_remove(
                    AddCameras._draw_handle, 'WINDOW')
                AddCameras._draw_handle = None
            return False

        rv3d = context.region_data
        context.scene.camera = self._cameras[0]
        if rv3d.view_perspective != 'CAMERA':
            bpy.ops.view3d.view_camera()
        bpy.ops.view3d.view_center_camera()
        try:
            rv3d.view_camera_zoom = 1.0
        except Exception:
            pass

        _sg_hide_label_overlay()
        if add_modal_handler:
            context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(
            0.5, window=context.window)
        self._last_time = time.time()
        bpy.ops.view3d.fly('INVOKE_DEFAULT')
        return True

    def _finish_without_review(self, context):
        """Clean up draw handler and restore state without fly-through."""
        if AddCameras._draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(
                AddCameras._draw_handle, 'WINDOW')
            AddCameras._draw_handle = None
        if self._cameras:
            context.scene.camera = self._cameras[0]
        _sg_restore_label_overlay()
        self.report({'INFO'},
                    f"Cameras added successfully ({len(self._cameras)} cameras).")

    # -------------------------------------------------------
    # Placement methods
    # -------------------------------------------------------

    def _place_orbit_ring(self, context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref):
        """Place cameras in a circle (original AddCameras behaviour)."""
        total = self.num_cameras
        count = total - 1 if using_viewport_ref else total
        angle_initial = math.atan2(initial_pos.y - center_location.y, initial_pos.x - center_location.x)
        angle_step = 2 * math.pi / (count + 1)

        for i in range(count):
            angle = (i + 1) * angle_step + angle_initial
            cam_data_new = bpy.data.cameras.new(name=f'Camera_{i+1}')
            cam_obj_new = bpy.data.objects.new(f'Camera_{i+1}', cam_data_new)
            context.collection.objects.link(cam_obj_new)
            if using_viewport_ref:
                delta = (i + 1) * angle_step
                T1 = Matrix.Translation(-center_location)
                Rz = Matrix.Rotation(delta, 4, 'Z')
                T2 = Matrix.Translation(center_location)
                cam_obj_new.matrix_world = T2 @ Rz @ T1 @ ref_mat
            else:
                x = center_location.x + radius * math.cos(angle)
                y = center_location.y + radius * math.sin(angle)
                z = initial_pos.z
                cam_obj_new.location = (x, y, z)
                direction = center_location - cam_obj_new.location
                rot = direction.to_track_quat('-Z', 'Y')
                cam_obj_new.rotation_euler = rot.to_euler()
            self._copy_camera_settings(cam_obj_new, cam_settings)
            self._cameras.append(cam_obj_new)

    def _place_fan(self, context, center_location, ref_mat, initial_pos, cam_settings, radius, using_viewport_ref):
        """Spread cameras in an arc around the active camera's position."""
        fan_rad = math.radians(self.fan_angle)
        angle_initial = math.atan2(initial_pos.y - center_location.y, initial_pos.x - center_location.x)
        total = self.num_cameras
        count = total - 1 if using_viewport_ref else total
        if count <= 0:
            return
        for i in range(count):
            if count == 1:
                t = 0.0
            else:
                t = (i / (count - 1)) - 0.5  # range -0.5 .. 0.5
            angle = angle_initial + t * fan_rad
            cam_data_new = bpy.data.cameras.new(name=f'Camera_fan_{i}')
            cam_obj_new = bpy.data.objects.new(f'Camera_fan_{i}', cam_data_new)
            context.collection.objects.link(cam_obj_new)
            if using_viewport_ref:
                delta = t * fan_rad
                T1 = Matrix.Translation(-center_location)
                Rz = Matrix.Rotation(delta, 4, 'Z')
                T2 = Matrix.Translation(center_location)
                cam_obj_new.matrix_world = T2 @ Rz @ T1 @ ref_mat
            else:
                x = center_location.x + radius * math.cos(angle)
                y = center_location.y + radius * math.sin(angle)
                z = initial_pos.z
                cam_obj_new.location = (x, y, z)
                direction = center_location - cam_obj_new.location
                rot = direction.to_track_quat('-Z', 'Y')
                cam_obj_new.rotation_euler = rot.to_euler()
            self._copy_camera_settings(cam_obj_new, cam_settings)
            self._cameras.append(cam_obj_new)

    def _create_cameras_from_directions(self, context, directions, center_np,
                                         verts_world, cam_settings, fov_x, fov_y):
        """Shared camera creation for all auto-placement modes.
        Each camera gets its own optimal distance computed from the mesh's
        silhouette extent in that viewing direction."""
        center_vec = mathutils.Vector(center_np.tolist())
        prefix_map = {
            'hemisphere': 'Camera_sphere',
            'normal_weighted': 'Camera_auto',
            'pca_axes': 'Camera_pca',
            'greedy_coverage': 'Camera',
        }
        prefix = prefix_map.get(self.placement_mode, 'Camera')

        for i, d in enumerate(directions):
            d_np = np.array(d, dtype=float)
            dist, aim_off = _compute_silhouette_distance(verts_world, center_np, d_np, fov_x, fov_y)
            dir_vec = mathutils.Vector(d_np.tolist()).normalized()
            aim_point = center_vec + mathutils.Vector(aim_off.tolist())
            pos = aim_point + dir_vec * dist

            cam_data = bpy.data.cameras.new(name=f'{prefix}_{i}')
            cam_obj = bpy.data.objects.new(f'{prefix}_{i}', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.location = pos
            right, up_v, d_unit = _camera_basis(d_np)
            cam_obj.rotation_euler = _rotation_from_basis(right, up_v, d_unit)
            self._copy_camera_settings(cam_obj, cam_settings)
            self._cameras.append(cam_obj)

    def _create_cameras_per_aspect(self, context, directions, center_np,
                                    verts_world, cam_settings, total_px,
                                    center_for_aspect):
        """Create cameras with per-camera optimal aspect ratio and distance.
        Uses an iterative refinement: first computes aspect from the
        orthographic silhouette, then refines it using the actual perspective
        projection from the computed camera position."""
        center_vec = mathutils.Vector(center_np.tolist())
        prefix_map = {
            'hemisphere': 'Camera_sphere',
            'normal_weighted': 'Camera_auto',
            'pca_axes': 'Camera_pca',
            'greedy_coverage': 'Camera',
        }
        prefix = prefix_map.get(self.placement_mode, 'Camera')

        for i, d in enumerate(directions):
            d_np = np.array(d, dtype=float)
            d_unit = d_np / np.linalg.norm(d_np)

            # --- Pass 1: orthographic aspect as initial estimate ---
            align = _get_resolution_align(context)
            aspect = _compute_per_camera_aspect(d_unit, verts_world, center_for_aspect)
            res_x, res_y = _resolution_from_aspect(aspect, total_px, align=align)
            fov_x, fov_y = _get_fov(cam_settings, context, res_x, res_y)
            dist, aim_off = _compute_silhouette_distance(
                verts_world, center_np, d_np, fov_x, fov_y)

            # --- Pass 2: refine aspect from actual perspective camera pos ---
            aim_point_np = center_np + aim_off
            cam_pos_np = aim_point_np + d_unit * dist
            aspect = _perspective_aspect(verts_world, cam_pos_np, d_np)
            res_x, res_y = _resolution_from_aspect(aspect, total_px, align=align)
            fov_x, fov_y = _get_fov(cam_settings, context, res_x, res_y)
            dist, aim_off = _compute_silhouette_distance(
                verts_world, center_np, d_np, fov_x, fov_y)

            dir_vec = mathutils.Vector(d_np.tolist()).normalized()
            aim_point = center_vec + mathutils.Vector(aim_off.tolist())
            pos = aim_point + dir_vec * dist

            cam_data = bpy.data.cameras.new(name=f'{prefix}_{i}')
            cam_obj = bpy.data.objects.new(f'{prefix}_{i}', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.location = pos
            right, up_v, d_unit_cam = _camera_basis(d_np)
            cam_obj.rotation_euler = _rotation_from_basis(right, up_v, d_unit_cam)
            self._copy_camera_settings(cam_obj, cam_settings)

            # Store per-camera resolution
            _store_per_camera_resolution(cam_obj, res_x, res_y)
            _setup_square_camera_display(cam_obj, res_x, res_y)
            self._cameras.append(cam_obj)

        # Set scene to square resolution (max side length) for viewport display
        max_side = max(
            max(int(c.get('sg_res_x', 0)), int(c.get('sg_res_y', 0)))
            for c in self._cameras if 'sg_res_x' in c
        ) if self._cameras else max(context.scene.render.resolution_x,
                                     context.scene.render.resolution_y)
        if max_side > 0:
            context.scene.render.resolution_x = max_side
            context.scene.render.resolution_y = max_side

    @staticmethod
    def _copy_camera_settings(cam_obj, cam_settings):
        """Copy lens / sensor / clip settings from a reference camera data block."""
        cam_obj.data.type = cam_settings.type
        cam_obj.data.lens = cam_settings.lens
        cam_obj.data.sensor_width = cam_settings.sensor_width
        cam_obj.data.sensor_height = cam_settings.sensor_height
        cam_obj.data.clip_start = cam_settings.clip_start
        cam_obj.data.clip_end = cam_settings.clip_end

    def modal(self, context, event):
        # ── Occlusion computation phase ──────────────────────────────────
        if self._occ_phase:
            if event.type in {'ESC', 'RIGHTMOUSE'}:
                # Cancel occlusion computation
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
                self._occ_gen = None
                self._occ_phase = False
                context.area.header_text_set(None)
                if AddCameras._draw_handle:
                    bpy.types.SpaceView3D.draw_handler_remove(
                        AddCameras._draw_handle, 'WINDOW')
                    AddCameras._draw_handle = None
                # Clean up temp camera data stored in state
                state = self._occ_state
                if state and state.get('temp_cam_data'):
                    bpy.data.cameras.remove(state['temp_cam_data'])
                self._occ_state = None
                self.report({'WARNING'}, "Occlusion computation cancelled.")
                return {'CANCELLED'}

            if event.type == 'TIMER':
                try:
                    progress = next(self._occ_gen)
                    self._occ_progress = progress
                    context.area.header_text_set(
                        f"Computing occlusion visibility… "
                        f"{progress * 100:.0f}%   (ESC to cancel)")
                    # Force viewport redraw so the GPU overlay updates too
                    for area in context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                except StopIteration as done:
                    # Generator finished – retrieve result
                    context.area.header_text_set(None)

                    # Clean up occlusion timer
                    context.window_manager.event_timer_remove(self._timer)
                    self._timer = None
                    self._occ_gen = None
                    self._occ_phase = False

                    state = self._occ_state
                    result_type = state.get('result_type', 'greedy')

                    if result_type == 'greedy':
                        # Greedy occlusion: result is (directions, coverage)
                        directions, coverage = done.value
                        directions = [np.array(d) for d in directions]
                        occ_label = state['occ_label']
                        self.report({'INFO'},
                            f"Greedy coverage ({occ_label}): "
                            f"{len(directions)} cameras, "
                            f"{coverage * 100:.1f}% coverage")
                    elif result_type == 'filter_kmeans':
                        # Face filter for K-means: result is exterior mask
                        exterior = done.value
                        normals = state['normals'][exterior]
                        areas = state['areas'][exterior]
                        n_removed = int((~exterior).sum())
                        self.report({'INFO'},
                            f"Occlusion filter: removed {n_removed} "
                            f"interior faces, {len(normals)} remain")
                        k = min(state['num_cameras'], len(normals))
                        if k > 0 and len(normals) > 0:
                            cluster_dirs = _kmeans_on_sphere(normals, areas, k)
                            directions = [cluster_dirs[i]
                                          for i in range(len(cluster_dirs))]
                        else:
                            directions = []
                    elif result_type == 'filter_pca':
                        # Face filter for PCA: result is exterior mask
                        exterior = done.value
                        # PCA uses filtered verts (only exterior faces)
                        f_centers = state['centers'][exterior]
                        n_removed = int((~exterior).sum())
                        self.report({'INFO'},
                            f"Occlusion filter: removed {n_removed} "
                            f"interior faces, {len(f_centers)} remain")
                        if len(f_centers) >= 3:
                            axes = _compute_pca_axes(f_centers)
                        else:
                            axes = _compute_pca_axes(
                                state['verts_world'])
                        directions = []
                        for axis in axes:
                            directions.append(axis)
                            directions.append(-axis)
                        if state.get('exclude_bottom'):
                            threshold_z = -math.cos(
                                state['exclude_bottom_angle'])
                            directions = [
                                d for d in directions
                                if d[2] >= threshold_z]
                        directions = directions[
                            :min(state['num_cameras'],
                                 len(directions))]

                    elif result_type == 'vis_kmeans':
                        # Visibility-weighted K-means: weight by fraction
                        vis_count = done.value
                        n_cand = 200
                        vis_fraction = vis_count.astype(float) / n_cand
                        exterior = vis_count > 0
                        normals = state['normals'][exterior]
                        areas_base = state['areas'][exterior]
                        weighted_areas = areas_base * vis_fraction[exterior]
                        n_ext = int(exterior.sum())
                        n_removed = len(vis_count) - n_ext
                        self.report({'INFO'},
                            f"Visibility-weighted: {n_ext} visible faces, "
                            f"{n_removed} fully occluded removed")
                        k = min(state['num_cameras'], len(normals))
                        if k > 0 and len(normals) > 0:
                            cluster_dirs = _kmeans_on_sphere(
                                normals, weighted_areas, k)
                            directions = [cluster_dirs[i]
                                          for i in range(len(cluster_dirs))]
                        else:
                            directions = []

                    elif result_type == 'vis_interactive':
                        # Enter interactive preview phase
                        vis_count = done.value
                        self._vis_count = vis_count
                        self._vis_n_candidates = 200
                        self._vis_balance = 0.2
                        self._vis_preview_phase = True
                        self._vis_state = state
                        self._occ_state = None
                        self._update_vis_cameras(context)
                        return {'RUNNING_MODAL'}

                    # Angular dedup against existing cameras (filter paths)
                    ex_dirs = state.get('existing_dirs')
                    if ex_dirs and result_type in (
                            'filter_kmeans', 'filter_pca', 'vis_kmeans'):
                        directions = _filter_near_existing(
                            directions, ex_dirs)

                    # Finalize cameras (sort, aspect, create, prompts)
                    self._finalize_auto_cameras(
                        context, directions,
                        state['verts_world'], state['mesh_center'],
                        state['cam_settings'], state['temp_cam_data'])
                    self._occ_state = None

                    # Transition to fly-through review
                    if self.review_placement:
                        if not self._start_fly_review(
                                context, add_modal_handler=False):
                            return {'CANCELLED'}
                        # Stay in modal for the fly-through phase
                    else:
                        self._finish_without_review(context)
                        return {'FINISHED'}
                return {'RUNNING_MODAL'}

            # Let Blender process UI events during occlusion
            return {'RUNNING_MODAL'}

        # ── Interactive visibility preview phase ─────────────────────────
        if self._vis_preview_phase:
            if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
                self._cleanup_vis_preview(context)
                self.report({'WARNING'}, "Interactive visibility cancelled.")
                return {'CANCELLED'}

            changed = False
            if event.type == 'WHEELUPMOUSE':
                self._vis_balance = min(1.0, self._vis_balance + 0.05)
                changed = True
            elif event.type == 'WHEELDOWNMOUSE':
                self._vis_balance = max(0.0, self._vis_balance - 0.05)
                changed = True
            elif event.type == 'NUMPAD_PLUS' and event.value == 'PRESS':
                self._vis_balance = min(1.0, self._vis_balance + 0.05)
                changed = True
            elif event.type == 'NUMPAD_MINUS' and event.value == 'PRESS':
                self._vis_balance = max(0.0, self._vis_balance - 0.05)
                changed = True

            if changed:
                self._update_vis_cameras(context)
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type in {'RET', 'SPACE'} and event.value == 'PRESS':
                # Confirm: finalize with current directions
                self._vis_preview_phase = False
                context.area.header_text_set(None)

                # Delete preview cameras
                for cam in list(self._cameras):
                    cam_data = cam.data
                    bpy.data.objects.remove(cam, do_unlink=True)
                    if cam_data and not cam_data.users:
                        bpy.data.cameras.remove(cam_data)
                self._cameras.clear()

                state = self._vis_state
                self._finalize_auto_cameras(
                    context, self._vis_directions,
                    state['verts_world'], state['mesh_center'],
                    state['cam_settings'], state['temp_cam_data'])
                self._vis_state = None
                self._vis_count = None

                # Transition to fly-through or finish
                if self.review_placement:
                    if not self._start_fly_review(
                            context, add_modal_handler=False):
                        return {'CANCELLED'}
                else:
                    self._finish_without_review(context)
                    return {'FINISHED'}
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        # ── Fly-through review phase ─────────────────────────────────────
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if time.time() - self._last_time < 0.2:
                return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            fly_running = any(op.bl_idname == 'VIEW3D_OT_fly'
                              for w in context.window_manager.windows
                              for op in w.modal_operators)
            if not fly_running:
                self._camera_index += 1
                if self._camera_index >= len(self._cameras):
                    context.window_manager.event_timer_remove(self._timer)
                    if AddCameras._draw_handle:
                        bpy.types.SpaceView3D.draw_handler_remove(AddCameras._draw_handle, 'WINDOW')
                        AddCameras._draw_handle = None
                    # Restore to initial camera, or first placed camera for auto modes
                    if self._initial_camera and self._initial_camera.name in [o.name for o in context.scene.objects]:
                        context.scene.camera = self._initial_camera
                    elif self._cameras:
                        context.scene.camera = self._cameras[0]
                    # Enable the floating prompt labels now that review is over
                    _sg_restore_label_overlay()
                    self.report({'INFO'}, "Cameras added successfully.")
                    return {'FINISHED'}
                context.scene.camera = self._cameras[self._camera_index]
                self._last_time = time.time()
                bpy.ops.view3d.fly('INVOKE_DEFAULT')
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    @classmethod
    def poll(cls, context):
        # Check for any running modal operator
        operator = None
        for window in context.window_manager.windows:
                for op in window.modal_operators:
                    if op.bl_idname == 'OBJECT_OT_test_stable' or op.bl_idname == 'OBJECT_OT_collect_camera_prompts' or op.bl_idname == 'OBJECT_OT_bake_textures' or op.bl_idname == 'OBJECT_OT_add_cameras':
                        operator = op
                        break
                if operator:
                    break
        if operator:
            return False
        return not any(op.bl_idname == cls.bl_idname
                       for w in context.window_manager.windows
                       for op in w.modal_operators)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


    
def switch_viewport_to_camera(context, camera):
    """Switches the first found 3D viewport to the specified camera's view."""
    if not camera:
        return
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            # Ensure we are in the right space and region
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
                    # Make sure the scene's active camera is set
                    context.scene.camera = camera
                    area.tag_redraw()
                    break
            break 

class CloneCamera(bpy.types.Operator):
    """Create a new camera at the active camera's position and enter fly mode to reposition it.
    
    If no camera exists, one is created from the current viewport.
    Useful for incrementally adding cameras one at a time."""
    bl_idname = "object.clone_camera"
    bl_label = "Clone Camera"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _camera = None
    _original_camera = None

    @classmethod
    def poll(cls, context):
        for window in context.window_manager.windows:
            for op in window.modal_operators:
                if op.bl_idname in ('OBJECT_OT_test_stable', 'OBJECT_OT_collect_camera_prompts',
                                    'OBJECT_OT_bake_textures', 'OBJECT_OT_add_cameras',
                                    'OBJECT_OT_clone_camera'):
                    return False
        return True

    def execute(self, context):
        ref_cam = context.scene.camera
        self._original_camera = ref_cam

        if not ref_cam:
            # Create from viewport
            rv3d = context.region_data
            cam_data = bpy.data.cameras.new(name='Camera_clone')
            cam_obj = bpy.data.objects.new('Camera_clone', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.matrix_world = rv3d.view_matrix.inverted()
        else:
            # Clone from active camera
            cam_data = bpy.data.cameras.new(name=f'{ref_cam.name}_clone')
            cam_obj = bpy.data.objects.new(cam_data.name, cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.matrix_world = ref_cam.matrix_world.copy()
            cam_obj.data.type = ref_cam.data.type
            cam_obj.data.lens = ref_cam.data.lens
            cam_obj.data.sensor_width = ref_cam.data.sensor_width
            cam_obj.data.sensor_height = ref_cam.data.sensor_height
            cam_obj.data.clip_start = ref_cam.data.clip_start
            cam_obj.data.clip_end = ref_cam.data.clip_end

        self._camera = cam_obj
        context.scene.camera = cam_obj

        rv3d = context.region_data
        if rv3d.view_perspective != 'CAMERA':
            bpy.ops.view3d.view_camera()
        bpy.ops.view3d.view_center_camera()
        try:
            rv3d.view_camera_zoom = 1.0
        except Exception:
            pass

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        bpy.ops.view3d.fly('INVOKE_DEFAULT')
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            fly_running = any(op.bl_idname == 'VIEW3D_OT_fly'
                              for w in context.window_manager.windows
                              for op in w.modal_operators)
            if not fly_running:
                context.window_manager.event_timer_remove(self._timer)
                if self._original_camera:
                    context.scene.camera = self._original_camera
                self.report({'INFO'}, f"Camera cloned: {self._camera.name}")
                return {'FINISHED'}
            return {'PASS_THROUGH'}
        return {'PASS_THROUGH'}


class MirrorCamera(bpy.types.Operator):
    """Create a mirror of the active camera across a chosen axis through the object/scene center.
    
    The new camera is placed symmetrically on the opposite side and oriented to look at the center."""
    bl_idname = "object.mirror_camera"
    bl_label = "Mirror Camera"
    bl_options = {'REGISTER', 'UNDO'}

    mirror_axis: bpy.props.EnumProperty(
        name="Mirror Axis",
        description="Axis to mirror across",
        items=[
            ('X', "X Axis", "Mirror left / right"),
            ('Y', "Y Axis", "Mirror front / back"),
            ('Z', "Z Axis", "Mirror top / bottom"),
        ],
        default='X'
    ) # type: ignore

    @classmethod
    def poll(cls, context):
        return context.scene.camera is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        cam = context.scene.camera
        obj = context.object

        # Determine center point
        if obj and obj.type == 'MESH':
            center = obj.matrix_world.translation.copy()
        else:
            center = mathutils.Vector((0, 0, 0))

        # Mirror position
        pos = cam.location.copy()
        axis_idx = 'XYZ'.index(self.mirror_axis)
        delta = pos - center
        delta[axis_idx] = -delta[axis_idx]
        new_pos = center + delta

        # Create new camera
        cam_data = bpy.data.cameras.new(name=f'{cam.name}_mirror_{self.mirror_axis}')
        cam_obj = bpy.data.objects.new(cam_data.name, cam_data)
        context.collection.objects.link(cam_obj)

        # Copy settings
        cam_obj.data.type = cam.data.type
        cam_obj.data.lens = cam.data.lens
        cam_obj.data.sensor_width = cam.data.sensor_width
        cam_obj.data.sensor_height = cam.data.sensor_height
        cam_obj.data.clip_start = cam.data.clip_start
        cam_obj.data.clip_end = cam.data.clip_end

        cam_obj.location = new_pos
        direction = center - new_pos
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        self.report({'INFO'}, f"Mirrored camera created: {cam_obj.name}")
        return {'FINISHED'}


class ToggleCameraLabels(bpy.types.Operator):
    """Toggle floating camera prompt labels in the 3D viewport.

    Shows or hides the per-camera prompt text (from Collect Camera Prompts
    or auto-generated view labels) next to each camera in the viewport."""
    bl_idname = "object.toggle_camera_labels"
    bl_label = "Toggle Camera Labels"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return any(o.type == 'CAMERA' for o in context.scene.objects)

    def execute(self, context):
        global _sg_label_draw_handle
        if _sg_label_draw_handle is not None:
            _sg_remove_label_overlay()
            self.report({'INFO'}, "Camera labels hidden")
        else:
            _sg_ensure_label_overlay()
            self.report({'INFO'}, "Camera labels visible")
        return {'FINISHED'}

        
# Define the PropertyGroup to store camera prompts
class CameraPromptItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        name="Camera Name",
        description="Name of the camera object"
    ) # type: ignore
    prompt: bpy.props.StringProperty(
        name="View Description",
        description="Description of the view from this camera"
    ) # type: ignore

# ── Camera Generation Order ──────────────────────────────────────────────────

class CameraOrderItem(bpy.types.PropertyGroup):
    """One entry in the generation-order list. Stores the Blender camera name."""
    name: bpy.props.StringProperty(
        name="Camera Name",
        description="Name of the camera object"
    ) # type: ignore


class SG_UL_CameraOrderList(bpy.types.UIList):
    """UIList that shows cameras in their current generation order."""
    bl_idname = "SG_UL_CameraOrderList"

    def draw_item(self, _context, layout, _data, item, _icon,
                  _active_data, _active_propname, index):
        cam_obj = bpy.data.objects.get(item.name)
        if cam_obj is None:
            layout.label(text=f"{item.name} (missing)", icon='ERROR')
            return
        # Show index, camera icon, name, and prompt if any
        row = layout.row(align=True)
        row.label(text=f"{index + 1}.")
        row.label(text=item.name, icon='CAMERA_DATA')
        # Show prompt preview if one exists
        prompt_item = next(
            (p for p in _context.scene.camera_prompts if p.name == item.name), None)
        if prompt_item and prompt_item.prompt:
            sub = row.row()
            sub.scale_x = 1.5
            sub.label(text=prompt_item.prompt, icon='SHORTDISPLAY')


class SyncCameraOrder(bpy.types.Operator):
    """Rebuild the generation order list from all cameras in the scene (sorted alphabetically)"""
    bl_idname = "stablegen.sync_camera_order"
    bl_label = "Sync Camera Order"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        cameras = sorted(
            [obj for obj in scene.objects if obj.type == 'CAMERA'],
            key=lambda x: x.name)
        scene.sg_camera_order.clear()
        for cam in cameras:
            item = scene.sg_camera_order.add()
            item.name = cam.name
        scene.sg_camera_order_index = 0
        self.report({'INFO'}, f"Synced {len(cameras)} cameras to generation order list.")
        return {'FINISHED'}


class MoveCameraOrder(bpy.types.Operator):
    """Move the selected camera up or down in the generation order"""
    bl_idname = "stablegen.move_camera_order"
    bl_label = "Move Camera"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[('UP', "Up", ""), ('DOWN', "Down", "")],
        name="Direction"
    ) # type: ignore

    def execute(self, context):
        scene = context.scene
        idx = scene.sg_camera_order_index
        total = len(scene.sg_camera_order)
        if total < 2:
            return {'CANCELLED'}
        new_idx = idx - 1 if self.direction == 'UP' else idx + 1
        if new_idx < 0 or new_idx >= total:
            return {'CANCELLED'}
        scene.sg_camera_order.move(idx, new_idx)
        scene.sg_camera_order_index = new_idx
        return {'FINISHED'}


class ApplyCameraOrderPreset(bpy.types.Operator):
    """Sort the generation order list using a preset strategy.
    Cameras are first synced from the scene, then reordered."""
    bl_idname = "stablegen.apply_camera_order_preset"
    bl_label = "Apply Camera Order Preset"
    bl_options = {'REGISTER', 'UNDO'}

    strategy: bpy.props.EnumProperty(
        name="Strategy",
        items=[
            ('ALPHABETICAL', "Alphabetical (Original)",
             "Sort cameras alphabetically by name (original order for auto-generated cameras)"),
            ('FRONT_FIRST', "Front → Back → Sides",
             "Front-facing cameras first, then back, then sides. "
             "Prioritises the main view for sequential inpainting"),
            ('BACK_FIRST', "Back → Front → Sides",
             "Back-facing cameras first, then front, then sides"),
            ('ALTERNATING', "Alternating (Opposites)",
             "Alternate between opposing cameras (front, back, left, right…) "
             "for maximum context spread between consecutive views"),
            ('TOP_DOWN', "Top → Bottom",
             "Cameras sorted from highest elevation to lowest"),
            ('REVERSE', "Reverse Current",
             "Reverse the current generation order"),
        ],
        default='FRONT_FIRST'
    ) # type: ignore

    def execute(self, context):
        scene = context.scene

        # Always sync from scene first (picks up new/removed cameras)
        cameras = sorted(
            [obj for obj in scene.objects if obj.type == 'CAMERA'],
            key=lambda x: x.name)

        if not cameras:
            self.report({'ERROR'}, "No cameras in the scene.")
            return {'CANCELLED'}

        if self.strategy == 'REVERSE':
            # Reverse whatever is currently in the list
            names = [item.name for item in scene.sg_camera_order]
            if not names:
                names = [c.name for c in cameras]
            names.reverse()
        elif self.strategy == 'ALPHABETICAL':
            names = [c.name for c in cameras]
        else:
            # Compute mesh centre for direction classification
            mesh_objects = [obj for obj in scene.objects
                           if obj.type == 'MESH' and not obj.hide_get()]
            if mesh_objects:
                centres = [obj.matrix_world @ (
                    sum((mathutils.Vector(v.co) for v in obj.data.vertices),
                        mathutils.Vector()) / max(len(obj.data.vertices), 1))
                    for obj in mesh_objects]
                mesh_center = sum(centres, mathutils.Vector()) / len(centres)
            else:
                mesh_center = mathutils.Vector((0, 0, 0))

            mc = np.array(mesh_center, dtype=float)

            # Use the first camera (alphabetically) as the reference
            # "front" direction — avoids assuming any world axis is front.
            ref_pos = np.array(cameras[0].location, dtype=float)
            ref_d = ref_pos - mc
            ref_d_xy = ref_d[:2]
            ref_xy_len = np.linalg.norm(ref_d_xy)
            if ref_xy_len < 1e-8:
                ref_d_xy = np.array([0.0, 1.0])
            else:
                ref_d_xy = ref_d_xy / ref_xy_len

            # Compute direction angles for each camera
            cam_data = []
            for cam in cameras:
                pos = np.array(cam.location, dtype=float)
                d = pos - mc
                norm = np.linalg.norm(d)
                if norm < 1e-8:
                    d = np.array([0, 1, 0], dtype=float)
                else:
                    d /= norm
                # Elevation (angle above XY plane)
                elev = math.degrees(math.asin(np.clip(d[2], -1, 1)))
                # Azimuth relative to the first camera's XY direction
                d_xy = d[:2]
                d_xy_len = np.linalg.norm(d_xy)
                if d_xy_len < 1e-8:
                    azimuth = 0.0
                else:
                    d_xy = d_xy / d_xy_len
                    cos_a = float(np.clip(np.dot(d_xy, ref_d_xy), -1, 1))
                    cross = float(ref_d_xy[0] * d_xy[1]
                                  - ref_d_xy[1] * d_xy[0])
                    azimuth = math.degrees(math.atan2(cross, cos_a))
                cam_data.append((cam.name, elev, azimuth))

            if self.strategy == 'FRONT_FIRST':
                # Priority: front (small |azimuth|), then back (large |azimuth|), then sides
                def _front_key(item):
                    _name, _elev, az = item
                    abs_az = abs(az)
                    if abs_az <= 45:
                        bucket = 0  # front
                    elif abs_az >= 135:
                        bucket = 1  # back
                    else:
                        bucket = 2  # sides
                    return (bucket, abs_az, -_elev)
                cam_data.sort(key=_front_key)

            elif self.strategy == 'BACK_FIRST':
                def _back_key(item):
                    _name, _elev, az = item
                    abs_az = abs(az)
                    if abs_az >= 135:
                        bucket = 0  # back
                    elif abs_az <= 45:
                        bucket = 1  # front
                    else:
                        bucket = 2  # sides
                    return (bucket, -abs_az, -_elev)
                cam_data.sort(key=_back_key)

            elif self.strategy == 'ALTERNATING':
                # Sort by azimuth, then pick from alternating ends
                cam_data.sort(key=lambda x: x[2])
                alternated = []
                left, right = 0, len(cam_data) - 1
                pick_left = True
                while left <= right:
                    if pick_left:
                        alternated.append(cam_data[left])
                        left += 1
                    else:
                        alternated.append(cam_data[right])
                        right -= 1
                    pick_left = not pick_left
                cam_data = alternated

            elif self.strategy == 'TOP_DOWN':
                cam_data.sort(key=lambda x: -x[1])  # highest elevation first

            names = [item[0] for item in cam_data]

        # Apply to collection
        scene.sg_camera_order.clear()
        for n in names:
            item = scene.sg_camera_order.add()
            item.name = n
        scene.sg_camera_order_index = 0
        self.report({'INFO'},
                    f"Applied '{self.strategy}' order to {len(names)} cameras.")
        return {'FINISHED'}


class CollectCameraPrompts(bpy.types.Operator):
    """Cycles through cameras and prompts for a viewpoint description for each. This can help when specific perspectives fail to generate correctly.
    
    - These prompts will be appended before the main prompt for each camera generation.
    - Applicable for separate, sequential and refine (also within grid mode) modes.
    - Examples: 'front view', 'close-up on face', 'from above'."""
    bl_idname = "object.collect_camera_prompts"
    bl_label = "Collect Camera View Prompts"
    bl_options = {'REGISTER', 'UNDO'}

    camera_prompt: bpy.props.StringProperty(
        name="View Description",
        description="Describe the view from this camera (e.g., 'front view', 'close-up on face', 'from above')",
        default=""
    ) # type: ignore

    # Internal state
    _cameras: list = []
    _camera_index: int = 0

    @classmethod
    def poll(cls, context):
        # Check for any running modal operator
        operator = None
        for window in context.window_manager.windows:
                for op in window.modal_operators:
                    if op.bl_idname == 'OBJECT_OT_test_stable' or op.bl_idname == 'OBJECT_OT_add_cameras' or op.bl_idname == 'OBJECT_OT_bake_textures' or op.bl_idname == 'OBJECT_OT_collect_camera_prompts':
                        operator = op
                        break
                if operator:
                    break
        if operator:
            return False
        return any(obj.type == 'CAMERA' for obj in context.scene.objects)

    def invoke(self, context, event):
        # Get all camera objects and sort them by name
        self._cameras = sorted([obj for obj in context.scene.objects if obj.type == 'CAMERA'], key=lambda x: x.name)

        if not self._cameras:
            self.report({'ERROR'}, "No cameras found in the scene.")
            return {'CANCELLED'}

        # Initialize state
        self._camera_index = 0

        # Set the first camera and pre-fill prompt if exists
        current_cam = self._cameras[self._camera_index]

        # Find existing prompt or set default
        existing_item = next((item for item in context.scene.camera_prompts if item.name == current_cam.name), None)
        self.camera_prompt = existing_item.prompt if existing_item else ""

        context.scene.camera = current_cam
        switch_viewport_to_camera(context, current_cam) # Switch viewport
        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        if self._camera_index < len(self._cameras):
            layout.label(text=f"Camera: {self._cameras[self._camera_index].name} ({self._camera_index + 1}/{len(self._cameras)})")
            layout.prop(self, "camera_prompt")

    def execute(self, context):
        cam_name = self._cameras[self._camera_index].name

        # Find existing item or add a new one
        prompt_item = next((item for item in context.scene.camera_prompts if item.name == cam_name), None)
        if not prompt_item:
            prompt_item = context.scene.camera_prompts.add()
            prompt_item.name = cam_name

        prompt_item.prompt = self.camera_prompt.strip() # Store trimmed prompt

        self._camera_index += 1
        if self._camera_index < len(self._cameras):
            next_cam = self._cameras[self._camera_index]
            # Pre-fill next prompt
            existing_item = next((item for item in context.scene.camera_prompts if item.name == next_cam.name), None)
            self.camera_prompt = existing_item.prompt if existing_item else ""
            # Ensure scene camera is set for next dialog and switch view
            context.scene.camera = next_cam
            switch_viewport_to_camera(context, next_cam) # Switch viewport
            return context.window_manager.invoke_props_dialog(self, width=400) # Show next dialog
        else:
            # Refresh the floating labels so edited prompts appear
            _sg_ensure_label_overlay()
            self.report({'INFO'}, f"Collected prompts for {len(self._cameras)} cameras.")
            return {'FINISHED'}
    

class SwitchMaterial(bpy.types.Operator):
    """Switches the material of all objects to a desired index."""

    bl_idname = "object.switch_material"
    bl_label = "Switch Material"
    bl_options = {'REGISTER', 'UNDO'}

    material_index: bpy.props.IntProperty(
        name="Material Index",
        description="Index of the material to switch to",
        default=0,
        min=0
    ) # type: ignore

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        """     
        Executes the operator.         
        :param context: Blender context.         
        :return: {'FINISHED'}     
        """
        # Select all objects
        bpy.ops.object.select_all(action='SELECT')
        

        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            if self.material_index < len(obj.data.materials):
                obj.active_material_index = self.material_index
                # Store original materials
                original_materials = list(obj.data.materials)
                # Store the material to be set as active
                to_be_active_material = obj.active_material
                # Clear materials and assign the material at the specified index
                obj.data.materials.clear()
                obj.data.materials.append(to_be_active_material)
                # Restore original materials
                for mat in original_materials:
                    if mat != to_be_active_material:
                        obj.data.materials.append(mat)
                
        # Deselct all objects
        bpy.ops.object.select_all(action='DESELECT')
        return {'FINISHED'}
    
def prepare_baking(context):
    bpy.context.scene.render.engine = 'CYCLES'
    # Force CPU + OSL only for Blender < 5.1 (native Raycast nodes don't need it)
    if bpy.app.version < (5, 1, 0):
        if hasattr(bpy.context.scene.cycles, 'shading_system'):
            bpy.context.scene.cycles.shading_system = True
        else:
            bpy.context.scene.cycles.use_osl = True
        bpy.context.scene.cycles.device = 'CPU'
    else:
        # Blender 5.1+: prefer GPU baking for massive speedup on high-poly.
        # Setting cycles.device = 'GPU' alone is not enough — we must also
        # set compute_device_type on Cycles preferences and ensure the GPU
        # device is enabled, otherwise Blender silently falls back to CPU.
        try:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            # Probe all available backends in order of preference
            for backend in ('CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'):
                try:
                    cycles_prefs.compute_device_type = backend
                    cycles_prefs.get_devices()
                    gpu_devs = [d for d in cycles_prefs.devices
                                if d.type == backend]
                    if gpu_devs:
                        for d in gpu_devs:
                            d.use = True
                        bpy.context.scene.cycles.device = 'GPU'
                        print(f"[StableGen] Baking with {backend} GPU: "
                              f"{', '.join(d.name for d in gpu_devs)}")
                        break
                except TypeError:
                    # This backend is not available on this platform
                    continue
            else:
                print("[StableGen] No GPU backend found, baking on CPU")
        except Exception:
            pass  # Fall back to whatever the user had

    # Set bake type to diffuse and contributions to color only
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True
    bpy.context.scene.render.bake.view_from = 'ABOVE_SURFACE'

    # Set steps to 1 for faster baking
    bpy.context.scene.cycles.samples = 1

    # Minimize light bounces — not needed for EMIT or color-only DIFFUSE bakes
    bpy.context.scene.cycles.max_bounces = 0

    # Keep the BVH in memory between consecutive bakes (huge win for PBR
    # where we bake 5-7 channels of the same mesh back-to-back)
    bpy.context.scene.render.use_persistent_data = True

def unwrap(obj, method, overlap_only):
        """     
        Unwraps the UVs of the given object using the selected method.         
        :param obj: Blender object.         
        :return: None     
        """
        if method == 'none':
            return
        
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # Set object as active and select it
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        is_new = False

        # If all UV maps are ProjectionUV or buffer, add new one
        if all(["ProjectionUV" in uv.name or uv.name == "_SG_ProjectionBuffer" for uv in obj.data.uv_layers]):
            # Add a new UV map
            obj.data.uv_layers.new(name=f"BakeUV")
            is_new = True
            obj.data.uv_layers.active_index = len(obj.data.uv_layers) - 1
            # Set it for rendering
            obj.data.uv_layers.active = obj.data.uv_layers[-1]
        else:
            # Ensure the active UV is a non-ProjectionUV/buffer map
            for uv_layer in obj.data.uv_layers:
                if "ProjectionUV" not in uv_layer.name and uv_layer.name != "_SG_ProjectionBuffer":
                    obj.data.uv_layers.active = uv_layer
                    break
        
        bpy.ops.object.mode_set(mode='EDIT')

        # Ensure UV selection sync is OFF
        bpy.context.scene.tool_settings.use_uv_select_sync = False

        if overlap_only:
            # Deselect
            bpy.ops.uv.select_all(action='DESELECT')
            # Check if the object has overlapping UVs, if not, skip unwrapping
            bpy.ops.uv.select_overlap()
            # Get a BMesh representation of the mesh
            bm = bmesh.from_edit_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            # Use the active UV layer (this is a BMUVLayer)
            uv_layer = bm.loops.layers.uv.active
            uv_layer = bm.loops.layers.uv.active
            if not uv_layer:
                bpy.ops.object.mode_set(mode='OBJECT')
                return
            
            # Check for ANY selected UV elements
            if bpy.app.version >= (5, 0, 0):
                # Blender 5.0+ uses loop.uv_select_vert
                has_overlap = any(
                    loop.uv_select_vert
                    for face in bm.faces 
                    for loop in face.loops
                )
            else:
                has_overlap = any(
                    loop[uv_layer].select 
                    for face in bm.faces 
                    for loop in face.loops
                )

            if not has_overlap:
                bpy.ops.object.mode_set(mode='OBJECT')
                return

        bpy.context.scene.tool_settings.use_uv_select_sync = True
        # Select all faces
        bpy.ops.mesh.select_all(action='SELECT')
        
        if method == 'basic':
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
        elif method == 'smart':
            bpy.ops.uv.smart_project()
        elif method == 'cube':
            bpy.ops.uv.cube_project(cube_size=1.0)
        elif method == 'lightmap':
            bpy.ops.uv.lightmap_pack()
        elif method == 'pack':
            bpy.ops.uv.pack_islands()

        bpy.ops.uv.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

def bake_texture(context, obj, texture_resolution, suffix = "", view_transform = 'Standard', output_dir = None):
        """     
        Bakes the texture of the given object.         
        :param context: Blender context.         
        :param obj: Blender object.         
        :return: True if successful, False otherwise. 
        """
        bpy.context.scene.display_settings.display_device = 'sRGB'
        # Backup original view transform
        original_view_transform = bpy.context.scene.view_settings.view_transform
        # Backup whether the object was enabled for rendering
        original_render = obj.hide_render
    
        bpy.context.scene.view_settings.view_transform = view_transform
        # Set the object to be rendered
        obj.hide_render = False

        # Create a new image for baking
        image_name = f"{obj.name}_baked{suffix}" 
        image = bpy.data.images.new(name=image_name, width=texture_resolution, height=texture_resolution)

        # Create a new texture node in the object's material
        if not obj.data.materials:
            # Cancel
            print("No materials found")
            return False
        else:
            # Temporarily remove all original materials
            mat = obj.active_material.copy()
            original_materials = list(obj.data.materials)
            original_active_material = obj.active_material
            obj.data.materials.clear()
            obj.data.materials.append(mat)

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.image = image
        tex_image.location = (0, 0)

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # Set object as active and select it
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        # Set the image node as the active bake target
        nodes.active = tex_image    

        # Ensure the active UV is a non-ProjectionUV/buffer map for correct baking
        for uv_layer in obj.data.uv_layers:
            if "ProjectionUV" not in uv_layer.name and uv_layer.name != "_SG_ProjectionBuffer":
                obj.data.uv_layers.active = uv_layer
                break
        
        # Check if there is a BSDF node before the output node
        output = None
        had_bsdf = False
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output = node
                break
            
        before_output = output.inputs[0].links[0].from_node if output and output.inputs[0].is_linked else None
        if before_output and before_output.type != 'BSDF_PRINCIPLED':
            bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf_node.location = (before_output.location[0] + 200, before_output.location[1])
            # Connect the before output node to the BSDF node
            links.new(before_output.outputs[0], bsdf_node.inputs[0])
            # Connect the BSDF node to the output node
            links.new(bsdf_node.outputs[0], output.inputs[0])

        # Disconnect Metallic so the DIFFUSE color-only pass returns the
        # true Base Color.  Principled BSDF multiplies diffuse by
        # (1 - Metallic), which would dim metallic areas incorrectly.
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                met = node.inputs.get("Metallic")
                if met:
                    for link in list(met.links):
                        links.remove(link)
                    met.default_value = 0.0
                break
            
        # Bake the texture
        # Ensure OBJECT mode — bake fails silently in EDIT mode
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.bake(type='DIFFUSE', width=texture_resolution, height=texture_resolution)
       
        # Save the image if required
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if suffix == "":
            image.filepath_raw = os.path.join(output_dir, f"{obj.name}.png")
        else:
            image.filepath_raw = os.path.join(output_dir, f"{obj.name}_baked{suffix}.png")
        image.file_format = 'PNG'
        image.save()

        # Remove tex_image node
        nodes.remove(tex_image)

        # Restore original view transform
        bpy.context.scene.view_settings.view_transform = original_view_transform
        # Restore original render settings
        obj.hide_render = original_render

        # Restore original materials
        obj.data.materials.clear()
        # First append the original active material
        if original_active_material:
            obj.data.materials.append(original_active_material)
        for mat_item in original_materials:
            if mat_item != original_active_material:
                obj.data.materials.append(mat_item)

        # Remove the temporary material copy (prevent leak)
        bpy.data.materials.remove(mat)
                
        return True

def _has_non_projection_uv(obj):
    """Return True if the object has at least one UV map that can be used
    for baking (not a StableGen projection UV or buffer)."""
    if not obj or obj.type != 'MESH':
        return False
    if not obj.data.uv_layers:
        return False
    for uv in obj.data.uv_layers:
        if (not uv.name.startswith("ProjectionUV")
                and uv.name != "_SG_ProjectionBuffer"):
            return True
    return False

def _uvs_likely_overlap(obj, uv_name=None):
    me = obj.data
    if not me.uv_layers:
        return False
    uv_layer = me.uv_layers.get(uv_name) if uv_name else me.uv_layers.active
    if not uv_layer:
        return False

    seen = set()
    for poly in me.polygons:
        for li in poly.loop_indices:
            uv = uv_layer.data[li].uv
            key = (round(uv.x, 5), round(uv.y, 5))
            if key in seen:
                return True
            seen.add(key)
    return False


# ── PBR Channel Baking Helpers ─────────────────────────────────────────

# Standard export suffixes per PBR channel
_PBR_CHANNEL_SUFFIXES = {
    'base_color': 'BaseColor',
    'roughness':  'Roughness',
    'metallic':   'Metallic',
    'normal':     'Normal',
    'emission':   'Emission',
    'height':     'Height',
    'ao':         'AO',
}


def _find_pbr_bake_sources(mat):
    """Inspect a Principled BSDF material and return bakeable PBR channel sources.

    Returns:
        dict: ``{channel_name: output_socket}`` for each connected PBR
        channel on the material's Principled BSDF.
    """
    if not mat or not mat.use_nodes:
        return {}

    nodes = mat.node_tree.nodes
    sources = {}

    # Find Principled BSDF
    principled = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled = node
            break
    if not principled:
        return {}

    # ── Base Color + AO ──────────────────────────────────────────────
    base_input = principled.inputs.get("Base Color")
    if base_input and base_input.links:
        src_node = base_input.links[0].from_node
        is_ao_mult = (
            getattr(src_node, 'label', '') == 'AO Multiply'
            or (src_node.type == 'MIX'
                and getattr(src_node, 'data_type', '') == 'RGBA'
                and getattr(src_node, 'blend_type', '') == 'MULTIPLY'))
        if is_ao_mult:
            # A input (index 6) = base colour, B input (index 7) = AO
            a_in = src_node.inputs[6]
            if a_in.links:
                sources['base_color'] = a_in.links[0].from_socket
            b_in = src_node.inputs[7]
            if b_in.links:
                sources['ao'] = b_in.links[0].from_socket
        else:
            sources['base_color'] = base_input.links[0].from_socket

    # ── Roughness ────────────────────────────────────────────────────
    ri = principled.inputs.get("Roughness")
    if ri and ri.links:
        sources['roughness'] = ri.links[0].from_socket

    # ── Metallic ─────────────────────────────────────────────────────
    mi = principled.inputs.get("Metallic")
    if mi and mi.links:
        sources['metallic'] = mi.links[0].from_socket

    # ── Normal (raw colours before Normal Map node) ──────────────────
    ni = principled.inputs.get("Normal")
    if ni and ni.links:
        n_node = ni.links[0].from_node
        if n_node.type == 'NORMAL_MAP':
            ci = n_node.inputs.get("Color")
            if ci and ci.links:
                sources['normal'] = ci.links[0].from_socket

    # ── Emission ─────────────────────────────────────────────────────
    ei = principled.inputs.get("Emission Color") or principled.inputs.get("Emission")
    if ei and ei.links:
        sources['emission'] = ei.links[0].from_socket

    # ── Height (from Displacement node → Material Output) ────────────
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            di = node.inputs.get("Displacement")
            if di and di.links:
                d_node = di.links[0].from_node
                if d_node.type == 'DISPLACEMENT':
                    hi = d_node.inputs.get("Height")
                    if hi and hi.links:
                        sources['height'] = hi.links[0].from_socket
            break

    return sources


def _restore_materials(obj, original_materials, original_active):
    """Restore an object's material slots back to their original state."""
    obj.data.materials.clear()
    if original_active:
        obj.data.materials.append(original_active)
    for m in original_materials:
        if m != original_active:
            obj.data.materials.append(m)


def bake_pbr_channel(context, obj, channel_name, texture_resolution, output_dir):
    """Bake one PBR channel via the Emit technique.

    Copies the material, finds the source socket for *channel_name*,
    rewires it through an Emission shader to the Material Output, bakes
    ``EMIT``, and saves the result to ``output_dir/{obj.name}_{Suffix}.png``.

    Returns:
        File path of the baked image, or ``None`` on failure.
    """
    if not obj.data.materials or not obj.active_material:
        return None

    # ── Save state ─────────────────────────────────────────────────
    orig_vt = context.scene.view_settings.view_transform
    orig_render = obj.hide_render
    context.scene.view_settings.view_transform = 'Standard'
    obj.hide_render = False

    # ── Copy material onto the object ──────────────────────────────
    original_materials = list(obj.data.materials)
    original_active = obj.active_material
    mat_copy = original_active.copy()
    obj.data.materials.clear()
    obj.data.materials.append(mat_copy)
    mat_copy.use_nodes = True
    nodes = mat_copy.node_tree.nodes
    links = mat_copy.node_tree.links

    # ── Find source socket in the copy ─────────────────────────────
    sources = _find_pbr_bake_sources(mat_copy)
    source_socket = sources.get(channel_name)
    if source_socket is None:
        _restore_materials(obj, original_materials, original_active)
        bpy.data.materials.remove(mat_copy)
        context.scene.view_settings.view_transform = orig_vt
        obj.hide_render = orig_render
        return None

    # ── Find Output node ───────────────────────────────────────────
    output_node = None
    for n in nodes:
        if n.type == 'OUTPUT_MATERIAL':
            output_node = n
            break
    if not output_node:
        _restore_materials(obj, original_materials, original_active)
        bpy.data.materials.remove(mat_copy)
        context.scene.view_settings.view_transform = orig_vt
        obj.hide_render = orig_render
        return None

    # ── Rewire material for baking ───────────────────────────────
    # Normal channel: use Cycles' built-in NORMAL bake which automatically
    # converts to tangent space — game-engine / glTF compatible regardless
    # of the original normal mode (world, bump, tangent).
    # All other channels: rewire source → Emission → Output for EMIT bake.
    bake_type = 'EMIT'
    if channel_name == 'normal':
        # Keep the Principled BSDF wired to the Output with the Normal
        # input connected.  Cycles NORMAL bake evaluates the shader's
        # final shading normal and converts it to tangent space for us.
        bake_type = 'NORMAL'
        # Ensure the normal influence is at full strength for baking
        principled = None
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                principled = n
                break
        if principled and principled.inputs.get("Normal") and principled.inputs["Normal"].links:
            nmap_node = principled.inputs["Normal"].links[0].from_node
            if nmap_node.type == 'NORMAL_MAP':
                nmap_node.inputs["Strength"].default_value = 1.0
    else:
        for link in list(output_node.inputs["Surface"].links):
            links.remove(link)

        emit_node = nodes.new("ShaderNodeEmission")
        emit_node.location = (output_node.location[0] - 200,
                              output_node.location[1])
        links.new(source_socket, emit_node.inputs["Color"])
        links.new(emit_node.outputs["Emission"],
                  output_node.inputs["Surface"])

    # ── Bake target image ──────────────────────────────────────────
    suffix = _PBR_CHANNEL_SUFFIXES.get(channel_name, channel_name)
    img_name = f"{obj.name}_{suffix}"
    bake_img = bpy.data.images.new(name=img_name,
                                   width=texture_resolution,
                                   height=texture_resolution)

    # Non-color channels (roughness, metallic, normal, height, AO) must be
    # stored in linear / Non-Color so the round-trip is lossless:
    #   Cycles linear → raw 8-bit → PNG → load as Non-Color → linear.
    # Without this, the default sRGB image applies gamma encoding on write,
    # but add_baked_material() loads them as Non-Color (no decode) →
    # values are gamma-shifted (e.g. roughness 0.22 → 0.50).
    if channel_name not in ('base_color', 'emission'):
        bake_img.colorspace_settings.name = 'Non-Color'

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bake_img
    nodes.active = tex_node

    # ── Select object + non-projection UV ──────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    context.view_layer.objects.active = obj
    obj.select_set(True)
    for uv in obj.data.uv_layers:
        if "ProjectionUV" not in uv.name and uv.name != "_SG_ProjectionBuffer":
            obj.data.uv_layers.active = uv
            break

    # ── Bake ─────────────────────────────────────────────────────
    # Ensure OBJECT mode — bake fails silently in EDIT mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if bake_type == 'NORMAL':
        bpy.ops.object.bake(type='NORMAL',
                            normal_space='TANGENT',
                            width=texture_resolution,
                            height=texture_resolution)
    else:
        bpy.ops.object.bake(type='EMIT',
                            width=texture_resolution,
                            height=texture_resolution)

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{obj.name}_{suffix}.png")
    bake_img.filepath_raw = out_path
    bake_img.file_format = 'PNG'
    bake_img.save()
    print(f"    Saved PBR channel: {out_path}")

    # ── Cleanup ────────────────────────────────────────────────────
    _restore_materials(obj, original_materials, original_active)
    bpy.data.materials.remove(mat_copy)
    context.scene.view_settings.view_transform = orig_vt
    obj.hide_render = orig_render
    return out_path


def _flip_normal_green(filepath):
    """Flip the green channel of a normal map (OpenGL ↔ DirectX)."""
    if not os.path.exists(filepath):
        return
    try:
        import numpy as np
        img = bpy.data.images.load(filepath, check_existing=False)
        w, h = img.size
        px = np.array(img.pixels[:]).reshape(h, w, 4)
        px[:, :, 1] = 1.0 - px[:, :, 1]
        img.pixels = px.flatten().tolist()
        img.save()
        bpy.data.images.remove(img)
        print(f"    Flipped normal green channel (DirectX): {filepath}")
    except Exception as e:
        print(f"[StableGen] Failed to flip normal green channel: {e}")


def _pack_orm_texture(output_dir, obj_name):
    """Pack AO, Roughness, Metallic into an ORM texture.

    Channel layout: R = AO, G = Roughness, B = Metallic  (Unreal / glTF).
    Requires at least Roughness and Metallic to exist.
    """
    import numpy as np

    rough_path = os.path.join(output_dir, f"{obj_name}_Roughness.png")
    metal_path = os.path.join(output_dir, f"{obj_name}_Metallic.png")
    if not os.path.exists(rough_path) or not os.path.exists(metal_path):
        return None

    rough_img = bpy.data.images.load(rough_path, check_existing=True)
    metal_img = bpy.data.images.load(metal_path, check_existing=True)
    w, h = rough_img.size

    rough_px = np.array(rough_img.pixels[:]).reshape(h, w, 4)
    metal_px = np.array(metal_img.pixels[:]).reshape(h, w, 4)

    ao_path = os.path.join(output_dir, f"{obj_name}_AO.png")
    if os.path.exists(ao_path):
        ao_img = bpy.data.images.load(ao_path, check_existing=True)
        ao_px = np.array(ao_img.pixels[:]).reshape(h, w, 4)
        ao_chan = ao_px[:, :, 0]
    else:
        ao_chan = np.ones((h, w), dtype=np.float32)

    orm = np.ones((h, w, 4), dtype=np.float32)
    orm[:, :, 0] = ao_chan
    orm[:, :, 1] = rough_px[:, :, 0]
    orm[:, :, 2] = metal_px[:, :, 0]

    orm_img = bpy.data.images.new(name=f"{obj_name}_ORM", width=w, height=h)
    orm_img.pixels = orm.flatten().tolist()

    orm_path = os.path.join(output_dir, f"{obj_name}_ORM.png")
    orm_img.filepath_raw = orm_path
    orm_img.file_format = 'PNG'
    orm_img.save()
    print(f"    Packed ORM texture: {orm_path}")
    return orm_path


class BakeTextures(bpy.types.Operator):
    """Bakes textures using the cycles render engine.
    
    - This will convert the textures to use a UV map. The first non-projection UV map will be used. If there are none, a new one will be created.
    - Textures will be output to the "baked" directory in the output path.
    - This will make the generated textures available in the Material Preview viewport shading mode."""
    bl_idname = "object.bake_textures"
    bl_label = "Bake Textures"
    bl_options = {'REGISTER', 'UNDO'}

    texture_resolution: bpy.props.IntProperty(
        name="Texture Resolution",
        description="Resolution of the baked textures",
        default=2048,
        min=128,
        max=8192
    ) # type: ignore

    try_unwrap: bpy.props.EnumProperty(
        name="Unwrap Method",
        description="Method to unwrap UVs before baking",
        items=[
            ('none', 'None', 'Skip UV unwrapping'),
            ('cube', 'Cube Project', 'Fast cube projection — great for high-poly / TRELLIS meshes (handles 1M+ faces instantly)'),
            ('smart', 'Smart UV Project', 'Smart UV Project — good balance of quality and speed'),
            ('basic', 'Basic Unwrap', 'Angle-based unwrap — best quality but very slow on high-poly meshes'),
            ('lightmap', 'Lightmap Pack', 'Use Lightmap Pack with default parameters'),
            ('pack', 'Pack Islands', 'Use Pack Islands with default parameters')
        ],
        default='none'
    ) # type: ignore

    add_material: bpy.props.BoolProperty(
        name="Add Material",
        description="Add the baked texture as a material to the objects",
        default=True
    ) # type: ignore

    flatten_for_refine: bpy.props.BoolProperty(
        name="Bake & Continue Refining",
        description="After baking, apply the baked texture to the StableGen projection material and clean up previous projection images",
        default=False
    ) # type: ignore

    overlap_only: bpy.props.BoolProperty(
        name="Overlap Only",
        description="Only unwrap objects with overlapping UVs",
        default=False
    ) # type: ignore

    bake_pbr: bpy.props.BoolProperty(
        name="Bake PBR Maps",
        description="Bake individual PBR channel maps (BaseColor, Roughness, Metallic, Normal, Emission, Height, AO) "
                    "with standard naming for game-engine import",
        default=True
    ) # type: ignore

    export_orm: bpy.props.BoolProperty(
        name="Pack ORM Texture",
        description="Create a packed ORM texture (R=AO, G=Roughness, B=Metallic) for Unreal Engine / glTF workflows",
        default=False
    ) # type: ignore

    normal_convention: bpy.props.EnumProperty(
        name="Normal Convention",
        description="Normal map Y-axis convention",
        items=[
            ('opengl', 'OpenGL (Y+)', 'Standard OpenGL / glTF / Unity / Blender convention'),
            ('directx', 'DirectX (Y-)', 'DirectX / Unreal Engine convention (flips green channel)'),
        ],
        default='opengl'
    ) # type: ignore

    _timer = None
    _objects = []
    _current_index = 0
    _phase = 'unwrap'

    # Add properties to track progress
    _progress = 0.0
    _stage = ""
    _current_object = 0
    _total_objects = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress = 0
        self._stage = ""
        self._current_object = 0
        self._total_objects = 0

    @classmethod
    def poll(self, context):
        """     
        Checks if the operator can be executed
        :param context: Blender context.
        :return: True if the operator can be executed, False otherwise.
        """
        # Check for any running modal operator
        operator = None
        for window in context.window_manager.windows:
                for op in window.modal_operators:
                    if op.bl_idname == 'OBJECT_OT_test_stable' or op.bl_idname == 'OBJECT_OT_add_cameras' or op.bl_idname == 'OBJECT_OT_collect_camera_prompts' or op.bl_idname == 'OBJECT_OT_bake_textures':
                        operator = op
                        break
                if operator:
                    break
        if operator:
            return False
        
        addon_prefs = context.preferences.addons[__package__].preferences
        if not os.path.exists(addon_prefs.output_dir):
            return False
        
        # Check if there are any textures to bake
        return True

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "texture_resolution")
        layout.prop(self, "try_unwrap")
        layout.prop(self, "overlap_only")
        layout.prop(self, "add_material")
        layout.prop(self, "flatten_for_refine")
        layout.separator()
        layout.label(text="PBR Export:")
        layout.prop(self, "bake_pbr")
        pbr_col = layout.column()
        pbr_col.enabled = self.bake_pbr
        pbr_col.prop(self, "export_orm")
        pbr_col.prop(self, "normal_convention")

    def invoke(self, context, event):
        """     
        Invokes the operator.         
        :param context: Blender context.         
        :param event: Blender event.         
        :return: {'RUNNING_MODAL'}     
        """
        if context.scene.texture_objects == 'all':
            self._objects = [obj for obj in context.view_layer.objects if obj.type == 'MESH' and not obj.hide_get()]
        else: # 'selected'
            self._objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        self._current_index = 0
        self._phase = 'unwrap'
        self._total_objects = len(self._objects)
        self._stage = "Preparing"
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        """     
        Executes the operator.         
        :param context: Blender context.         
        :return: {'RUNNING_MODAL'}     
        """

        self.original_engine = bpy.context.scene.render.engine
        self.original_shading = bpy.context.space_data.shading.type
        self.original_device = bpy.context.scene.cycles.device
        self.original_samples = bpy.context.scene.cycles.samples
        self.original_max_bounces = bpy.context.scene.cycles.max_bounces
        self.original_persistent = bpy.context.scene.render.use_persistent_data
        # Save Cycles compute_device_type so we can restore after GPU bake
        try:
            self._original_compute_device_type = (
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type
            )
        except Exception:
            self._original_compute_device_type = None
        # Set render engine to CYCLES (required for baking)
        bpy.context.scene.render.engine = 'CYCLES'
        # Switch to Solid shading to avoid conflicts with Rendered mode
        # (Rendered viewport + bake = crash).  Solid is engine-independent
        # and won't cause pink textures like Material Preview (EEVEE) does.
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'
                        break
        prepare_baking(context)

        # Start modal operation
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """     
        Handles modal events.         
        :param context: Blender context.         
        :param event: Blender event.         
        :return: {'PASS_THROUGH'}     
        """
        if event.type == 'TIMER':
            def redraw():
                for area in context.screen.areas:
                    area.tag_redraw()
            redraw()

            # ── PBR channel bake (flat queue, one item per tick) ──────
            if self._phase == 'bake_pbr':
                if self._pbr_queue_idx < len(self._pbr_queue):
                    obj, channel = self._pbr_queue[self._pbr_queue_idx]
                    nice = _PBR_CHANNEL_SUFFIXES.get(channel, channel)
                    self._stage = f"PBR {nice} — {obj.name}"
                    self._progress = (self._pbr_queue_idx / len(self._pbr_queue)) * 100
                    redraw()
                    bake_pbr_channel(
                        context, obj, channel,
                        self.texture_resolution,
                        get_dir_path(context, "baked"))
                    self._pbr_queue_idx += 1
                else:
                    # DirectX normal flip
                    if self.normal_convention == 'directx':
                        out_dir = get_dir_path(context, "baked")
                        for obj in self._objects:
                            _flip_normal_green(
                                os.path.join(out_dir,
                                             f"{obj.name}_Normal.png"))
                    # Pack ORM
                    if self.export_orm:
                        self.report({'INFO'}, "Packing ORM textures...")
                        self._phase = 'pack_orm'
                        self._current_index = 0
                    else:
                        self.report({'INFO'}, "Applying materials...")
                        self._phase = 'apply_material'
                        self._current_index = 0
                return {'PASS_THROUGH'}

            if self._current_index < len(self._objects):
                obj = self._objects[self._current_index]
                if self._phase == 'unwrap':
                    self._stage = f"Unwrapping {obj.name}"
                    self._progress = 0
                    redraw()

                    if self.try_unwrap != 'none':
                        if not _has_non_projection_uv(obj):
                            # No usable UV exists — must unwrap
                            unwrap(obj, self.try_unwrap, self.overlap_only)
                        elif self.overlap_only:
                            # Only re-unwrap if existing UVs overlap
                            if _uvs_likely_overlap(obj):
                                unwrap(obj, self.try_unwrap, self.overlap_only)
                        else:
                            # Always re-unwrap when overlap_only is disabled
                            unwrap(obj, self.try_unwrap, self.overlap_only)

                    self._progress = 100
                elif self._phase == 'bake':
                    if self.bake_pbr:
                        # PBR mode bakes BaseColor via EMIT — skip the
                        # redundant DIFFUSE bake entirely.
                        self._progress = 100
                    else:
                        self._stage = f"Baking {obj.name}"
                        self._progress = 0
                        redraw()
                        if not bake_texture(context, obj, self.texture_resolution, output_dir=get_dir_path(context, "baked")):
                            self.report({'ERROR'}, f"Failed to bake texture for {obj.name}. No materials found.")
                            context.window_manager.event_timer_remove(self._timer)
                            return {'CANCELLED'}

                        # NEW: optionally flatten into the projection material so you can keep refining
                        if self.flatten_for_refine:
                            try:
                                baked_path = get_file_path(context, "baked", object_name=obj.name)
                                flatten_projection_material_for_refine(context, obj, baked_path)
                                purge_orphans()
                            except Exception as e:
                                print(f"[StableGen] Failed to flatten projection material for {obj.name}: {e}")

                        self._progress = 100
                elif self._phase == 'pack_orm':
                    self._stage = f"Packing ORM — {obj.name}"
                    self._progress = 0
                    redraw()
                    _pack_orm_texture(get_dir_path(context, "baked"), obj.name)
                    self._progress = 100
                elif self._phase == 'apply_material' and self.add_material:
                    self._stage = f"Applying Material to {obj.name}"
                    self._progress = 0
                    redraw()
                    self.add_baked_material(context, obj)
                    self._progress = 100
                self._current_index += 1
                if self._current_index < len(self._objects):
                    self._current_object = self._current_index
            else:
                if self._phase == 'unwrap':
                    self.report({'INFO'}, "Baking textures...")
                    self._phase = 'bake'
                    self._current_index = 0
                elif self._phase == 'bake':
                    if self.bake_pbr:
                        self.report({'INFO'}, "Baking PBR channels...")
                        self._phase = 'bake_pbr'
                        self._pbr_queue = []
                        for obj in self._objects:
                            if obj.active_material:
                                for ch in _find_pbr_bake_sources(obj.active_material):
                                    self._pbr_queue.append((obj, ch))
                        self._pbr_queue_idx = 0
                        if not self._pbr_queue:
                            # No PBR sources — skip straight to apply
                            self.report({'INFO'}, "No PBR channels found, skipping PBR bake...")
                            self._phase = 'apply_material'
                            self._current_index = 0
                    else:
                        self.report({'INFO'}, "Applying materials...")
                        self._phase = 'apply_material'
                        self._current_index = 0
                elif self._phase == 'pack_orm':
                    self.report({'INFO'}, "Applying materials...")
                    self._phase = 'apply_material'
                    self._current_index = 0
                else:
                    context.window_manager.event_timer_remove(self._timer)
                    bpy.context.scene.render.engine = self.original_engine
                    bpy.context.scene.cycles.device = self.original_device
                    bpy.context.scene.cycles.samples = self.original_samples
                    bpy.context.scene.cycles.max_bounces = self.original_max_bounces
                    bpy.context.scene.render.use_persistent_data = self.original_persistent
                    # Restore Cycles compute_device_type
                    if self._original_compute_device_type is not None:
                        try:
                            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = (
                                self._original_compute_device_type
                            )
                        except Exception:
                            pass
                    # Restore original viewport shading
                    for area in context.screen.areas:
                        if area.type == 'VIEW_3D':
                            for space in area.spaces:
                                if space.type == 'VIEW_3D':
                                    space.shading.type = self.original_shading
                                    break
                    self.report({'INFO'}, "Textures baked successfully.")
                    remove_empty_dirs(context)
                    return {'FINISHED'}
        return {'PASS_THROUGH'}

    def add_baked_material(self, context, obj):
        """Create a baked material with standard PBR textures (game-engine ready).

        When ``bake_pbr`` is enabled, looks for ``{obj.name}_{Channel}.png``
        files in the baked output directory and wires them to a Principled
        BSDF.  Falls back to the single-texture path when PBR maps are absent.
        """
        output_dir = get_dir_path(context, "baked")

        mat = bpy.data.materials.new(name=f"{obj.name}_baked")
        obj.data.materials.append(mat)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        # Assign the new material to all faces
        obj.active_material_index = len(obj.material_slots) - 1
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Output node
        output_node = nodes.new("ShaderNodeOutputMaterial")
        output_node.location = (800, 0)

        # Principled BSDF
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (400, 0)
        # Fallback values when PBR textures are absent:
        # max roughness + zero metallic → matte, non-reflective surface.
        principled.inputs["Roughness"].default_value = 1.0
        principled.inputs["Metallic"].default_value = 0.0

        # UV Map
        uv_node = nodes.new("ShaderNodeUVMap")
        if "BakeUV" in [uv.name for uv in obj.data.uv_layers]:
            uv_node.uv_map = "BakeUV"
        else:
            uv_node.uv_map = obj.data.uv_layers[0].name
        uv_node.location = (-600, 0)

        # ── Base Color ────────────────────────────────────────────────
        base_path = os.path.join(output_dir, f"{obj.name}_BaseColor.png")
        if not os.path.exists(base_path):
            # Fallback to standard diffuse bake
            base_path = get_file_path(context, "baked", object_name=obj.name)

        has_base = os.path.exists(base_path)
        if has_base:
            tex_base = nodes.new("ShaderNodeTexImage")
            tex_base.image = bpy.data.images.load(base_path)
            tex_base.location = (-200, 200)
            tex_base.label = "BaseColor"
            links.new(uv_node.outputs["UV"], tex_base.inputs["Vector"])

            if context.scene.apply_bsdf or self.bake_pbr:
                links.new(tex_base.outputs["Color"],
                          principled.inputs["Base Color"])
                links.new(principled.outputs[0],
                          output_node.inputs["Surface"])
            else:
                links.new(tex_base.outputs["Color"],
                          output_node.inputs["Surface"])
                return
        else:
            links.new(principled.outputs[0],
                      output_node.inputs["Surface"])

        if not self.bake_pbr:
            return

        # ── PBR channel textures ──────────────────────────────────────
        y_pos = -100

        # Roughness
        rough_path = os.path.join(output_dir, f"{obj.name}_Roughness.png")
        if os.path.exists(rough_path):
            y_pos -= 250
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(rough_path)
            tex.image.colorspace_settings.name = 'Non-Color'
            tex.location = (-200, y_pos)
            tex.label = "Roughness"
            links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
            links.new(tex.outputs["Color"],
                      principled.inputs["Roughness"])

        # Metallic
        metal_path = os.path.join(output_dir, f"{obj.name}_Metallic.png")
        if os.path.exists(metal_path):
            y_pos -= 250
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(metal_path)
            tex.image.colorspace_settings.name = 'Non-Color'
            tex.location = (-200, y_pos)
            tex.label = "Metallic"
            links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
            links.new(tex.outputs["Color"],
                      principled.inputs["Metallic"])

        # Normal
        normal_path = os.path.join(output_dir, f"{obj.name}_Normal.png")
        if os.path.exists(normal_path):
            y_pos -= 250
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(normal_path)
            tex.image.colorspace_settings.name = 'Non-Color'
            tex.location = (-200, y_pos)
            tex.label = "Normal"
            links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
            nmap = nodes.new("ShaderNodeNormalMap")
            nmap.location = (100, y_pos)
            nmap.label = "Normal Map"
            # The normal bake uses Cycles NORMAL pass with tangent space,
            # so the texture is always tangent-space regardless of the
            # original projection material's normal mode (world/bump/tangent).
            nmap.space = 'TANGENT'

            links.new(tex.outputs["Color"], nmap.inputs["Color"])
            if "Normal" in principled.inputs:
                links.new(nmap.outputs["Normal"],
                          principled.inputs["Normal"])

        # Emission
        emission_path = os.path.join(output_dir,
                                     f"{obj.name}_Emission.png")
        if os.path.exists(emission_path):
            y_pos -= 250
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(emission_path)
            tex.location = (-200, y_pos)
            tex.label = "Emission"
            links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
            em_in = (principled.inputs.get("Emission Color")
                     or principled.inputs.get("Emission"))
            if em_in:
                links.new(tex.outputs["Color"], em_in)
            if "Emission Strength" in principled.inputs:
                principled.inputs[
                    "Emission Strength"].default_value = 1.0

        # Height → Displacement
        height_path = os.path.join(output_dir,
                                   f"{obj.name}_Height.png")
        if os.path.exists(height_path):
            y_pos -= 250
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(height_path)
            tex.image.colorspace_settings.name = 'Non-Color'
            tex.location = (-200, y_pos)
            tex.label = "Height"
            links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
            disp = nodes.new("ShaderNodeDisplacement")
            disp.location = (600, -300)
            disp.label = "Displacement"
            disp.inputs["Scale"].default_value = 0.1
            disp.inputs["Midlevel"].default_value = 0.5
            links.new(tex.outputs["Color"], disp.inputs["Height"])
            links.new(disp.outputs["Displacement"],
                      output_node.inputs["Displacement"])

        # AO → multiply with Base Color
        ao_path = os.path.join(output_dir, f"{obj.name}_AO.png")
        if (os.path.exists(ao_path)
                and principled.inputs["Base Color"].links):
            y_pos -= 250
            tex_ao = nodes.new("ShaderNodeTexImage")
            tex_ao.image = bpy.data.images.load(ao_path)
            tex_ao.image.colorspace_settings.name = 'Non-Color'
            tex_ao.location = (-200, y_pos)
            tex_ao.label = "AO"
            links.new(uv_node.outputs["UV"], tex_ao.inputs["Vector"])
            # Intercept Base Color → Principled with a Multiply
            old_src = principled.inputs[
                "Base Color"].links[0].from_socket
            mix = nodes.new("ShaderNodeMix")
            mix.data_type = 'RGBA'
            mix.blend_type = 'MULTIPLY'
            mix.inputs["Factor"].default_value = 1.0
            mix.location = (200, 150)
            mix.label = "AO Multiply"
            links.new(old_src, mix.inputs[6])           # A
            links.new(tex_ao.outputs["Color"], mix.inputs[7])  # B
            links.new(mix.outputs[2],
                      principled.inputs["Base Color"])


class ExportOrbitGIF(bpy.types.Operator):
    """Exports a GIF and MP4 animation orbiting the active object"""
    bl_idname = "object.export_orbit_gif"
    bl_label = "Export Orbit GIF/MP4"
    bl_options = {'REGISTER', 'UNDO'}

    duration: bpy.props.FloatProperty(
        name="Duration (seconds)",
        description="Duration of the 360-degree orbit",
        default=5.0,
        min=0.1,
        max=60.0
    ) # type: ignore

    frame_rate: bpy.props.IntProperty(
        name="Frame Rate (fps)",
        description="Frames per second for the animation",
        default=24,
        min=1,
        max=60
    ) # type: ignore

    resolution_percentage: bpy.props.IntProperty(
        name="Resolution %",
        description="Percentage of the scene's render resolution to use",
        default=50,
        min=10,
        max=100,
        subtype='PERCENTAGE'
    ) # type: ignore

    samples: bpy.props.IntProperty(
        name="Samples",
        description="Number of render samples per frame",
        default=32,
        min=1,
        max=4096
    ) # type: ignore
    
    engine: bpy.props.EnumProperty(
        name="Render Engine",
        description="Render engine to use",
        items=[
            ('BLENDER_WORKBENCH', "Workbench", "Use Workbench render engine"),
            ('EEVEE', "Eevee", "Use Eevee render engine"),
            ('CYCLES', "Cycles", "Use Cycles render engine")
        ],
        default='CYCLES'
    ) # type: ignore

    interpolation: bpy.props.EnumProperty(
        name="Rotation Curve",
        description="Keyframe interpolation for the orbit rotation",
        items=[
            ('LINEAR', "Linear", "Constant speed throughout the orbit"),
            ('BEZIER', "Ease In/Out", "Smooth acceleration and deceleration"),
        ],
        default='LINEAR'
    ) # type: ignore

    # ── PBR showcase settings ─────────────────────────────────────────
    use_hdri: bpy.props.BoolProperty(
        name="HDRI Environment",
        description="Use the scene's existing HDRI world (set up via the "
                    "Add HDRI operator) for realistic environment lighting",
        default=False
    ) # type: ignore

    hdri_rotation: bpy.props.FloatProperty(
        name="Environment Rotation",
        description="Initial Z-rotation offset for the HDRI (degrees)",
        default=0.0,
        min=0.0,
        max=360.0,
        subtype='ANGLE'
    ) # type: ignore

    env_mode: bpy.props.EnumProperty(
        name="Environment Mode",
        description="How the environment lighting interacts with the orbit",
        items=[
            ('FIXED', "Fixed",
             "Camera orbits, HDRI stays fixed — reflections shift naturally"),
            ('LOCKED', "Locked",
             "HDRI co-rotates with the camera — lighting stays identical "
             "from every angle (consistent showcase)"),
            ('COUNTER', "Counter-Rotate",
             "HDRI rotates opposite to camera — 2× apparent light change"),
            ('ENV_ONLY', "Environment Only",
             "Camera stays fixed, only the HDRI rotates around the model"),
        ],
        default='FIXED'
    ) # type: ignore

    use_shadow_plane: bpy.props.BoolProperty(
        name="Ground Shadow",
        description="Add a temporary shadow-catcher plane beneath the "
                    "model for a grounded appearance (Cycles only)",
        default=False
    ) # type: ignore

    use_denoiser: bpy.props.BoolProperty(
        name="Denoise",
        description="Enable Cycles denoising for cleaner specular "
                    "highlights with fewer samples",
        default=True
    ) # type: ignore

    _timer = None
    _rendering = False
    _cancelled = False # Added flag to track cancellation
    _handle_complete = None
    _handle_cancel = None
    _initial_settings = {}
    _temp_empty = None # Added for the pivot empty
    _temp_shadow_plane = None  # Added for shadow catcher
    _env_mapping_node = None   # For HDRI rotation animation
    _original_world_data = None  # Store entire world setup for restore
    _output_path = "" # Internal variable to store the final GIF path
    _output_path_mp4 = "" # Internal variable to store the final MP4 path
    _temp_dir = "" # Temporary directory for frames
    _frame_paths = [] # List to store paths of rendered frames
    _original_camera_parent = None # Store original camera parent
    _original_camera_matrix = None # Store original camera matrix

    @classmethod
    def poll(cls, context):
        # Check if imageio is available
        try:
            import imageio
            # Optionally check for ffmpeg plugin if strict MP4 support is needed
            # This basic check is usually sufficient as imageio tries to find ffmpeg
        except ImportError:
            return False
        # Check for active object (Mesh or Empty) and active camera
        return context.active_object is not None and context.active_object.type in {'MESH', 'EMPTY'} and context.scene.camera is not None

    def invoke(self, context, event):
        # Check dependency again in invoke to provide feedback
        try:
            import imageio
            # Check if ffmpeg is likely available for MP4 export
            if not imageio.plugins.ffmpeg.is_available():
                 self.report({'WARNING'}, "FFmpeg plugin for imageio not found. MP4 export might fail. Install 'imageio-ffmpeg'.")
        except ImportError:
            self.report({'ERROR'}, "Python module 'imageio' not found. Please install it (e.g., 'pip install imageio imageio-ffmpeg').")
            return {'CANCELLED'}
        except Exception as e:
             self.report({'WARNING'}, f"Could not check imageio ffmpeg availability: {e}. MP4 export might fail.")


        # Check for active camera
        if not context.scene.camera:
            self.report({'ERROR'}, "No active camera found in the scene.")
            return {'CANCELLED'}

        # Determine output paths using get_dir_path
        try:
            revision_dir = get_dir_path(context, "revision")
            os.makedirs(revision_dir, exist_ok=True)
            self._output_path = os.path.join(revision_dir, "orbit.gif")
            self._output_path_mp4 = os.path.join(revision_dir, "orbit.mp4") # MP4 path
        except Exception as e:
            self.report({'ERROR'}, f"Could not determine output directory: {e}")
            return {'CANCELLED'}

        # Create temporary directory for frames
        try:
            self._temp_dir = tempfile.mkdtemp(prefix="blender_gif_")
        except Exception as e:
            self.report({'ERROR'}, f"Could not create temporary directory: {e}")
            self.cleanup(context) # Clean up if temp dir fails
            return {'CANCELLED'}

        return context.window_manager.invoke_props_dialog(self)

    @staticmethod
    def _force_fcurve_interpolation(obj, interpolation_type):
        """Set interpolation on all keyframe points of *obj*'s fcurves.

        Works on Blender 4.x (action.fcurves) and 5.x+ (layered actions
        with slots / channelbags).
        """
        anim = obj.animation_data
        if not anim or not anim.action:
            return

        action = anim.action
        fcurves = []

        # Blender 5.x layered-action API (slots → channelbags → fcurves)
        action_slot = getattr(anim, "action_slot", None)
        if action_slot is not None:
            # Try the helper shipped in bpy_extras first
            try:
                from bpy_extras.anim_utils import action_get_channelbag_for_slot
                bag = action_get_channelbag_for_slot(action, action_slot)
                if bag is not None:
                    fcurves = list(bag.fcurves)
            except Exception:
                pass
            # Manual fallback: iterate channelbags on the *action*
            if not fcurves:
                for attr in ("channelbags", "channel_bags"):
                    bags = getattr(action, attr, None)
                    if bags is not None:
                        for bag in bags:
                            fcurves.extend(bag.fcurves)
                        break

        # Blender 4.x legacy path
        if not fcurves and hasattr(action, "fcurves"):
            fcurves = list(action.fcurves)

        for fc in fcurves:
            for kf in fc.keyframe_points:
                kf.interpolation = interpolation_type

    # ── PBR Showcase helpers ──────────────────────────────────────────

    def _setup_hdri_environment(self, context):
        """Prepare the scene's existing HDRI world for orbit rendering.

        Assumes the world was already set up by the AddHDRI operator
        (or any setup with an Environment Texture node).  This method:

        1. Stores the original world state for ``cleanup()``.
        2. Injects a Mapping node before the Environment Texture to
           allow rotation animation (counter-rotate / env-only modes).
        3. Applies the user's initial rotation offset.

        If no Environment Texture node is found, falls back to creating
        a procedural Nishita sky with a Mapping node.
        """
        world = context.scene.world
        if not world:
            world = bpy.data.worlds.new("SG_OrbitWorld")
            context.scene.world = world

        # Store original state for restore
        self._original_world_data = {
            'world_ref': world,
            'use_nodes': world.use_nodes,
            'color': world.color.copy(),
        }

        world.use_nodes = True
        tree = world.node_tree

        # Find the existing Environment Texture node
        env_tex = None
        for node in tree.nodes:
            if node.type == 'TEX_ENVIRONMENT':
                env_tex = node
                break

        if env_tex is None:
            # No HDRI world set up — create a procedural sky fallback
            print("[StableGen] Orbit GIF: No HDRI found, creating sky fallback")
            tree.nodes.clear()

            tex_coord = tree.nodes.new("ShaderNodeTexCoord")
            tex_coord.location = (-800, 300)

            mapping = tree.nodes.new("ShaderNodeMapping")
            mapping.location = (-600, 300)
            mapping.inputs["Rotation"].default_value[2] = self.hdri_rotation
            self._env_mapping_node = mapping

            sky_tex = tree.nodes.new("ShaderNodeTexSky")
            sky_tex.location = (-300, 300)
            sky_tex.sky_type = 'NISHITA'
            sky_tex.sun_elevation = math.radians(30)
            sky_tex.sun_rotation = math.radians(45)

            bg_node = tree.nodes.new("ShaderNodeBackground")
            bg_node.location = (0, 300)

            output_node = tree.nodes.new("ShaderNodeOutputWorld")
            output_node.location = (200, 300)

            tree.links.new(tex_coord.outputs["Generated"],
                           mapping.inputs["Vector"])
            tree.links.new(mapping.outputs["Vector"],
                           sky_tex.inputs["Vector"])
            tree.links.new(sky_tex.outputs["Color"],
                           bg_node.inputs["Color"])
            tree.links.new(bg_node.outputs["Background"],
                           output_node.inputs["Surface"])

            # Track that we created the fallback so cleanup can remove it
            self._original_world_data['created_fallback'] = True
            return

        # ── Inject Mapping node before the Environment Texture ────────
        # Store original Vector input link/socket for restore
        existing_vec_link = None
        if env_tex.inputs["Vector"].links:
            existing_vec_link = env_tex.inputs["Vector"].links[0]
            self._original_world_data['env_vec_from_socket'] = (
                existing_vec_link.from_socket)

        # Create TexCoord + Mapping chain
        tex_coord = tree.nodes.new("ShaderNodeTexCoord")
        tex_coord.name = "SG_OrbitTexCoord"
        tex_coord.location = (env_tex.location[0] - 500,
                              env_tex.location[1])

        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.name = "SG_OrbitMapping"
        mapping.location = (env_tex.location[0] - 250,
                            env_tex.location[1])
        mapping.inputs["Rotation"].default_value[2] = self.hdri_rotation
        self._env_mapping_node = mapping

        tree.links.new(tex_coord.outputs["Generated"],
                       mapping.inputs["Vector"])

        # Remove existing vector link and insert mapping before env tex
        if existing_vec_link:
            # Rewire: old_source → Mapping.Vector, Mapping.out → EnvTex.Vector
            old_source = existing_vec_link.from_socket
            tree.links.remove(existing_vec_link)
            tree.links.new(old_source, mapping.inputs["Vector"])
        # else: TexCoord → Mapping is already connected above

        tree.links.new(mapping.outputs["Vector"],
                       env_tex.inputs["Vector"])

        print("[StableGen] Orbit GIF: using existing HDRI world"
              f" (rotation offset={math.degrees(self.hdri_rotation):.0f}°)")

    def _setup_shadow_plane(self, context):
        """Create a temporary shadow-catcher plane beneath the model."""
        obj = context.active_object
        if not obj:
            return

        # Determine the bottom of the object's bounding box in world space
        bbox_corners = [obj.matrix_world @ mathutils.Vector(c)
                        for c in obj.bound_box]
        min_z = min(c.z for c in bbox_corners)
        center_x = sum(c.x for c in bbox_corners) / 8
        center_y = sum(c.y for c in bbox_corners) / 8

        # Determine a reasonable plane size (2× the bounding box extent)
        extent_x = max(c.x for c in bbox_corners) - min(c.x for c in bbox_corners)
        extent_y = max(c.y for c in bbox_corners) - min(c.y for c in bbox_corners)
        plane_size = max(extent_x, extent_y) * 3

        bpy.ops.mesh.primitive_plane_add(
            size=plane_size,
            location=(center_x, center_y, min_z))
        plane = context.active_object
        plane.name = "SG_ShadowPlane"

        # Set as shadow catcher (Cycles)
        plane.is_shadow_catcher = True

        # Create a holdout material so the plane is invisible except
        # for the shadow it catches.
        mat = bpy.data.materials.new("SG_ShadowCatcher")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        holdout = mat.node_tree.nodes.new("ShaderNodeHoldout")
        holdout.location = (-200, 0)
        mat.node_tree.links.new(holdout.outputs[0],
                                output.inputs["Surface"])
        plane.data.materials.append(mat)

        self._temp_shadow_plane = plane
        print("[StableGen] Orbit GIF: shadow catcher plane added")

    def _animate_hdri_rotation(self, context, total_frames):
        """Animate the HDRI Mapping node rotation for counter-rotate
        or environment-only modes."""
        if not self._env_mapping_node:
            return

        mapping = self._env_mapping_node
        prefs = bpy.context.preferences.edit
        prev_interp = prefs.keyframe_new_interpolation_type
        prefs.keyframe_new_interpolation_type = self.interpolation
        try:
            # Start rotation = current value (may include user offset)
            start_z = mapping.inputs["Rotation"].default_value[2]
            mapping.inputs["Rotation"].default_value[2] = start_z
            mapping.inputs["Rotation"].keyframe_insert(
                data_path="default_value", index=2, frame=1)

            # For counter-rotate: rotate in opposite direction to camera
            # For env-only: rotate in positive direction
            if self.env_mode == 'COUNTER':
                end_z = start_z - math.radians(360)
            elif self.env_mode == 'LOCKED':
                # Co-rotate: same direction as camera orbit (+Z)
                end_z = start_z + math.radians(360)
            else:
                end_z = start_z + math.radians(360)

            mapping.inputs["Rotation"].default_value[2] = end_z
            mapping.inputs["Rotation"].keyframe_insert(
                data_path="default_value", index=2, frame=total_frames + 1)
        finally:
            prefs.keyframe_new_interpolation_type = prev_interp

        # Force interpolation on the world node tree's fcurves
        world = context.scene.world
        if world and world.node_tree:
            self._force_fcurve_interpolation(
                world.node_tree, self.interpolation)

    def setup_animation(self, context):
        obj = context.active_object
        scene = context.scene
        active_camera = scene.camera # Use the scene's active camera

        if not active_camera: # Should be caught by poll/invoke, but double-check
             raise RuntimeError("No active camera found during setup.")

        # Store original camera state
        self._original_camera_parent = active_camera.parent
        self._original_camera_matrix = active_camera.matrix_world.copy()

        # Calculate Center (Center of Mass or Bounds)
        cursor_location = scene.cursor.location.copy()
        # Use object bounds center for better visual centering
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = obj
        obj.select_set(True)
        # Use geometry center for pivot, less prone to being skewed by outliers than bounds
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        center_location = obj.matrix_world.translation.copy()
        # Restore cursor location (origin setting might change cursor)
        scene.cursor.location = cursor_location

        # Create Temporary Empty at Center
        self._temp_empty = bpy.data.objects.new("OrbitPivot", None)
        self._temp_empty.location = center_location
        scene.collection.objects.link(self._temp_empty)
        # Update matrices after linking and setting location
        context.view_layer.update()

        # Calculate camera's local transform relative to the empty *before* parenting
        cam_original_world_matrix = self._original_camera_matrix.copy()
        empty_world_matrix_inv = self._temp_empty.matrix_world.inverted()
        cam_local_matrix = empty_world_matrix_inv @ cam_original_world_matrix

        # Parent Active Camera to Empty
        active_camera.parent = self._temp_empty
        # Set the camera's local transform (matrix_basis)
        active_camera.matrix_basis = cam_local_matrix

        # Set Up Animation Timing
        total_frames = int(self.duration * self.frame_rate)
        scene.frame_start = 1
        scene.frame_end = total_frames
        scene.render.fps = self.frame_rate

        # Animate Empty's Rotation (skip if ENV_ONLY — camera stays fixed)
        if self.env_mode != 'ENV_ONLY':
            # Set keyframe interpolation type before inserting so keyframes
            # are created with the correct curve.
            prefs = bpy.context.preferences.edit
            prev_interp = prefs.keyframe_new_interpolation_type
            prefs.keyframe_new_interpolation_type = self.interpolation
            try:
                self._temp_empty.rotation_euler = (0, 0, 0)
                self._temp_empty.keyframe_insert(data_path="rotation_euler", index=2, frame=1)

                self._temp_empty.rotation_euler = (0, 0, math.radians(360))
                self._temp_empty.keyframe_insert(data_path="rotation_euler", index=2, frame=total_frames + 1)
            finally:
                prefs.keyframe_new_interpolation_type = prev_interp

            # Post-insert: force the chosen interpolation on all keyframe points.
            # The preference alone doesn't always take effect on Blender 5.x.
            self._force_fcurve_interpolation(self._temp_empty, self.interpolation)

        # Animate HDRI environment rotation (locked, counter-rotate, or env-only)
        if self.use_hdri and self.env_mode in ('LOCKED', 'COUNTER', 'ENV_ONLY'):
            self._animate_hdri_rotation(context, total_frames)


    def execute(self, context):
        scene = context.scene
        render = scene.render
        cycles = scene.cycles # Get cycles settings

        # Store Initial Settings
        self._initial_settings = {
            'frame_start': scene.frame_start,
            'frame_end': scene.frame_end,
            'fps': render.fps,
            'camera': scene.camera, # Store the camera object itself
            'filepath': render.filepath,
            'file_format': render.image_settings.file_format,
            'color_mode': render.image_settings.color_mode,
            'resolution_percentage': render.resolution_percentage,
            'use_overwrite': render.use_overwrite,
            'use_placeholder': render.use_placeholder,
            'samples': cycles.samples, # Store original samples
            'engine': scene.render.engine, # Store original engine
            'film_transparent': render.film_transparent, # Store original film transparency
            'light': scene.display.shading.light, # Store original shading light
            'color_type': scene.display.shading.color_type, # Store original shading color type
            'use_compositing': render.use_compositing, # Store compositor pipeline state
        }

        scene.render.engine = get_eevee_engine_id() if self.engine == 'EEVEE' else self.engine
        
        if scene.render.engine == 'BLENDER_WORKBENCH':
            context.scene.display.shading.light = 'STUDIO'
            context.scene.display.shading.color_type = 'SINGLE'

        # Disable compositing in the render pipeline so Blender writes
        # the raw ViewLayer result to render.filepath. StableGen's
        # compositor uses File Output nodes without a Composite node,
        # which causes the Composite output (and thus saved frames) to be
        # blank/transparent.
        render.use_compositing = False

        # Apply Render Settings for PNG sequence
        render.filepath = os.path.join(self._temp_dir, "frame_") # Base path for frames
        render.image_settings.file_format = 'PNG' # Render as PNG sequence
        render.image_settings.color_mode = 'RGBA' # Use RGBA (needed for potential alpha, even if we make background opaque)
        render.resolution_percentage = self.resolution_percentage
        render.use_overwrite = True
        render.use_placeholder = False
        render.film_transparent = True 
        cycles.samples = self.samples # Set samples for rendering

        # ── PBR Showcase: HDRI Environment ────────────────────────────
        if self.use_hdri:
            self._setup_hdri_environment(context)

        # ── PBR Showcase: Cycles Denoiser ─────────────────────────────
        if self.use_denoiser and scene.render.engine == 'CYCLES':
            self._initial_settings['use_denoising'] = cycles.use_denoising
            self._initial_settings['denoiser'] = cycles.denoiser
            cycles.use_denoising = True
            # Prefer OpenImageDenoise (CPU-based, always available)
            try:
                cycles.denoiser = 'OPENIMAGEDENOISE'
            except TypeError:
                pass  # Fallback to whatever is default

        # ── PBR Showcase: Ground Shadow Plane ─────────────────────────
        if self.use_shadow_plane and scene.render.engine == 'CYCLES':
            self._setup_shadow_plane(context)

        # Setup Animation
        try:
            self.setup_animation(context)
        except Exception as e:
            # Clean up partially created objects if setup fails
            self.cleanup(context)
            self.report({'ERROR'}, f"Animation setup failed: {e}")
            # Print traceback for debugging
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        # Start Rendering
        self._rendering = True
        self._cancelled = False # Reset cancellation flag
        self._frame_paths.clear() # Clear list for new render
        self._handle_complete = bpy.app.handlers.render_complete.append(self.render_complete_handler)
        self._handle_cancel = bpy.app.handlers.render_cancel.append(self.render_cancel_handler)
        # Add handler for each rendered frame to collect paths
        bpy.app.handlers.render_post.append(self.render_post_handler)


        # Use timer to check render status without blocking UI completely
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)

        # Start the render process
        bpy.ops.render.render('INVOKE_DEFAULT', animation=True)

        self.report({'INFO'}, f"Rendering frames to temporary directory...")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            if not self._rendering:
                # Render finished or cancelled
                context.window_manager.event_timer_remove(self._timer)

                files_created = False
                # Check if frames exist AND it wasn't cancelled before trying to create files
                if self._frame_paths and not self._cancelled:
                    files_created = self.create_output_files() # Call create_output_files

                # Perform cleanup regardless
                self.cleanup(context)

                # Check for error AFTER cleanup, using the cancellation flag
                if not files_created and not self._cancelled: # Use self._cancelled flag
                     self.report({'ERROR'}, "No frames rendered or collected, or output file creation failed.")
                elif files_created: # Report success only if files were made
                     self.report({'INFO'}, f"GIF saved to {self._output_path}, MP4 saved to {self._output_path_mp4}")


                return {'FINISHED'}
        # Allow render window events to pass through
        return {'PASS_THROUGH'}


    def cleanup(self, context):
        # Remove handlers first to prevent them running during cleanup
        if self._handle_complete in bpy.app.handlers.render_complete:
            bpy.app.handlers.render_complete.remove(self._handle_complete)
        if self._handle_cancel in bpy.app.handlers.render_cancel:
            bpy.app.handlers.render_cancel.remove(self._handle_cancel)
        if self.render_post_handler in bpy.app.handlers.render_post:
             bpy.app.handlers.render_post.remove(self.render_post_handler)
        self._handle_complete = None
        self._handle_cancel = None

        # Restore initial settings
        scene = context.scene
        render = scene.render
        cycles = scene.cycles # Get cycles settings
        original_camera = self._initial_settings.get('camera')

        # Restore camera parent and transform BEFORE restoring scene.camera setting
        if original_camera and original_camera.name in bpy.data.objects:
            try:
                # Check if it's still parented to the temp empty before unparenting
                if original_camera.parent == self._temp_empty:
                    original_camera.parent = self._original_camera_parent
                    # Restore world matrix after unparenting
                    original_camera.matrix_world = self._original_camera_matrix
                else:
                    # If parent changed unexpectedly, just restore matrix
                    print("Warning: Camera parent changed during render, restoring world matrix only.")
                    original_camera.matrix_world = self._original_camera_matrix
                # Clear potentially stale parent inverse matrix
                original_camera.matrix_parent_inverse.identity()

            except Exception as e:
                print(f"Warning: Could not restore camera state for '{original_camera.name}': {e}")

        # Restore other settings
        for key, value in self._initial_settings.items():
            if key == 'camera':
                 # Restore the scene's active camera object if it exists
                 if value and value.name in bpy.data.objects:
                      scene.camera = value
                 continue # Already handled transform/parent above
            try:
                # Handle nested properties correctly
                if key == 'file_format':
                    setattr(render.image_settings, 'file_format', value)
                elif key == 'color_mode':
                    setattr(render.image_settings, 'color_mode', value)
                elif key in ('samples', 'use_denoising', 'denoiser'):
                    setattr(cycles, key, value)
                elif key == 'engine': # Restore engine
                    setattr(render, key, value)
                elif key == 'film_transparent': # Restore film transparency
                    setattr(render, key, value)
                elif key == 'use_compositing': # Restore compositor pipeline state
                    render.use_compositing = value
                elif hasattr(render, key):
                    setattr(render, key, value)
                elif hasattr(scene, key):
                    setattr(scene, key, value)
            except Exception as e:
                print(f"Warning: Could not restore setting '{key}': {e}")

        self._initial_settings = {} # Clear stored settings
        self._original_camera_parent = None
        self._original_camera_matrix = None

        # Remove temporary empty
        if self._temp_empty:
            # Remove animation data
            if self._temp_empty.animation_data and self._temp_empty.animation_data.action:
                action = self._temp_empty.animation_data.action
                # Check if action exists before removing
                if action and action.name in bpy.data.actions:
                     bpy.data.actions.remove(action)
            # Unlink and remove object
            if self._temp_empty.name in context.scene.collection.objects:
                context.scene.collection.objects.unlink(self._temp_empty)
            if self._temp_empty.name in bpy.data.objects:
                bpy.data.objects.remove(self._temp_empty, do_unlink=True)
            self._temp_empty = None

        # Remove temporary shadow-catcher plane
        if self._temp_shadow_plane:
            try:
                # Remove the material
                for mat_slot in self._temp_shadow_plane.data.materials:
                    if mat_slot and mat_slot.name in bpy.data.materials:
                        bpy.data.materials.remove(mat_slot)
                # Remove the mesh data
                mesh_data = self._temp_shadow_plane.data
                if self._temp_shadow_plane.name in context.scene.collection.objects:
                    context.scene.collection.objects.unlink(self._temp_shadow_plane)
                if self._temp_shadow_plane.name in bpy.data.objects:
                    bpy.data.objects.remove(self._temp_shadow_plane, do_unlink=True)
                if mesh_data and mesh_data.name in bpy.data.meshes:
                    bpy.data.meshes.remove(mesh_data)
            except Exception as e:
                print(f"Warning: Could not remove shadow plane: {e}")
            self._temp_shadow_plane = None

        # Restore the world environment (remove injected mapping nodes)
        if self._original_world_data:
            try:
                world = self._original_world_data['world_ref']
                if world and world.name in bpy.data.worlds:
                    # Remove HDRI animation keyframes
                    if world.node_tree and world.node_tree.animation_data:
                        action = world.node_tree.animation_data.action
                        if action and action.name in bpy.data.actions:
                            bpy.data.actions.remove(action)
                        world.node_tree.animation_data_clear()

                    if self._original_world_data.get('created_fallback'):
                        # We created a whole sky fallback — restore original
                        world.use_nodes = self._original_world_data['use_nodes']
                        world.color = self._original_world_data['color']
                        if not self._original_world_data['use_nodes']:
                            if world.node_tree:
                                world.node_tree.nodes.clear()
                    else:
                        # We injected SG_OrbitMapping + SG_OrbitTexCoord
                        # into the existing tree.  Remove them and restore
                        # the original vector link.
                        tree = world.node_tree
                        if tree:
                            # Find the env tex and restore its vector input
                            env_tex = None
                            for node in tree.nodes:
                                if node.type == 'TEX_ENVIRONMENT':
                                    env_tex = node
                                    break

                            orig_socket = self._original_world_data.get(
                                'env_vec_from_socket')
                            if env_tex and orig_socket:
                                tree.links.new(orig_socket,
                                               env_tex.inputs["Vector"])
                            elif env_tex:
                                # Had no vector link originally — just remove
                                for link in list(env_tex.inputs["Vector"].links):
                                    tree.links.remove(link)

                            # Remove injected nodes
                            for name in ("SG_OrbitMapping", "SG_OrbitTexCoord"):
                                if name in tree.nodes:
                                    tree.nodes.remove(tree.nodes[name])
            except Exception as e:
                print(f"Warning: Could not restore world environment: {e}")
            self._original_world_data = None
        self._env_mapping_node = None


        self._rendering = False # Ensure rendering flag is reset
        self._cancelled = False # Reset cancellation flag

        # Clean up temporary directory if it exists
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                print(f"Removed temporary directory: {self._temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory '{self._temp_dir}': {e}")
        self._temp_dir = ""
        self._frame_paths.clear() # Clear frame paths


    def render_post_handler(self, scene, _):
        """Called after each frame is rendered."""
        if self._rendering and self._temp_dir: # Check temp_dir exists
            frame_num = scene.frame_current
            # Construct the expected filename based on Blender's padding
            filename = f"frame_{frame_num:04d}.png"
            filepath = os.path.join(self._temp_dir, filename)
            if os.path.exists(filepath):
                self._frame_paths.append(filepath)
            else:
                # Check if render path uses frame number suffix differently
                # Use scene.render.frame_path() which respects output settings
                alt_filepath = scene.render.frame_path(frame=frame_num)
                if os.path.exists(alt_filepath):
                     self._frame_paths.append(alt_filepath)
                     # print(f"Used alternative frame path: {alt_filepath}") # Less verbose
                else:
                     print(f"Warning: Frame file not found after render: {filepath} or {alt_filepath}")


    def render_complete_handler(self, scene, _):
        try:
            # Only respond if we're still in the rendering phase
            if not getattr(self, "_rendering", False):
                return

            print("Frame rendering complete.")
            # Signal that rendering is done so modal can wrap up
            self._rendering = False

            # Unregister this handler so it won't fire after operator finishes
            # Check if handler exists before removing
            if hasattr(self, 'render_complete_handler') and self.render_complete_handler in bpy.app.handlers.render_complete:
                try:
                    bpy.app.handlers.render_complete.remove(self.render_complete_handler)
                except ValueError:
                    pass # Ignore if already removed elsewhere
        except ReferenceError:
            # RNA is gone ignore
            pass
        except Exception as e:
             print(f"Error in render_complete_handler: {e}") # Log other potential errors


    def render_cancel_handler(self, scene, _):
        if self._rendering: # Check if it was our render job that was cancelled
            self.report({'WARNING'}, "Render cancelled by user.")
            self._cancelled = True # Set the cancellation flag
            self._rendering = False # Signal modal loop to finish and cleanup
            # Unregister this handler
            if hasattr(self, 'render_cancel_handler') and self.render_cancel_handler in bpy.app.handlers.render_cancel:
                 try:
                      bpy.app.handlers.render_cancel.remove(self.render_cancel_handler)
                 except ValueError:
                      pass # Ignore if already removed


    def create_output_files(self):
        """Creates the GIF and MP4 from the rendered frames using imageio. Returns True on success, False otherwise."""
        if not self._frame_paths:
            # Error is reported in modal loop if needed
            return False

        print(f"Assembling output files from {len(self._frame_paths)} frames...")
        gif_success = False
        mp4_success = False

        try:
            # Sort frames numerically just in case paths weren't added perfectly in order
            self._frame_paths.sort()

            images = []
            for filename in self._frame_paths:
                try:
                    images.append(imageio.imread(filename))
                except FileNotFoundError:
                    print(f"Warning: Frame file disappeared before processing: {filename}")
                    continue # Skip missing frame

            if not images: # Check if any images were actually loaded
                 self.report({'ERROR'}, "No valid frame images found to create output files.")
                 return False

            # Calculate duration per frame for imageio (in seconds)
            frame_duration = 1.0 / self.frame_rate

            # --- Create GIF ---
            print(f"Creating GIF: {self._output_path}")
            try:
                 # imageio v3+ uses 'duration' in seconds per frame
                 imageio.mimsave(self._output_path, images, fps=self.frame_rate, loop=0, disposal=2) # loop=0 means infinite loop
                 print(f"GIF saved successfully.")
                 gif_success = True
            except TypeError:
                      # Fallback if 'duration' is not accepted 
                      imageio.mimsave(self._output_path, images, duration=frame_duration, loop=0)
                      print(f"GIF saved successfully (fallback duration).")
                      gif_success = True
            except Exception as e:
                 self.report({'ERROR'}, f"Failed to create GIF: {e}")
                 # Print traceback for debugging GIF errors
                 import traceback
                 traceback.print_exc()

            # --- Create MP4 ---
            print(f"Creating MP4: {self._output_path_mp4}")
            try:
                imageio.mimsave(self._output_path_mp4, images, format='mp4', fps=self.frame_rate, quality=8)
                print(f"MP4 saved successfully.")
                mp4_success = True
            except ImportError:
                 self.report({'ERROR'}, "Python module 'imageio' or its 'ffmpeg' plugin not found/configured correctly. Cannot create MP4.")
            except Exception as e:
                 self.report({'ERROR'}, f"Failed to create MP4: {e}")
                 # Print traceback for debugging MP4 errors
                 import traceback
                 traceback.print_exc()


            return gif_success and mp4_success # Return True only if both succeed

        except ImportError:
            # This top-level catch handles if imageio itself wasn't imported initially
            self.report({'ERROR'}, "Python module 'imageio' not found. Cannot create output files.")
            return False
        except Exception as e:
            # Catch other potential errors during image loading or sorting
            self.report({'ERROR'}, f"Failed during output file creation process: {e}")
            import traceback
            traceback.print_exc()
            return False


class ExportForGameEngine(bpy.types.Operator):
    """Export textured objects for game engines with PBR textures.

    Wraps Blender's glTF / FBX exporters with game-engine-optimal settings
    and auto-saves to the StableGen output directory."""
    bl_idname = "object.export_game_engine"
    bl_label = "Export for Game Engine"
    bl_options = {'REGISTER'}

    export_format: bpy.props.EnumProperty(
        name="Format",
        description="Export file format",
        items=[
            ('GLB', 'glTF Binary (.glb)',
             'Single-file binary glTF — best for web, Unity, Godot'),
            ('GLTF_SEPARATE', 'glTF + Textures (.gltf)',
             'Separate .gltf, .bin and texture files'),
            ('FBX', 'FBX (.fbx)',
             'Autodesk FBX with embedded textures — Unreal Engine'),
        ],
        default='GLB'
    ) # type: ignore

    export_scope: bpy.props.EnumProperty(
        name="Objects",
        description="Which objects to export",
        items=[
            ('SELECTED', 'Selected', 'Export selected mesh objects'),
            ('ALL_MESH', 'All Mesh', 'Export all visible mesh objects'),
        ],
        default='SELECTED'
    ) # type: ignore

    apply_transforms: bpy.props.BoolProperty(
        name="Apply Transforms",
        description="Apply location, rotation and scale before export",
        default=True
    ) # type: ignore

    export_animations: bpy.props.BoolProperty(
        name="Export Animations",
        description="Include animations in the export",
        default=False
    ) # type: ignore

    @classmethod
    def poll(cls, context):
        addon_prefs = context.preferences.addons[__package__].preferences
        return os.path.exists(addon_prefs.output_dir)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_format")
        layout.prop(self, "export_scope")
        layout.prop(self, "apply_transforms")
        layout.prop(self, "export_animations")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        # ── Prepare selection ─────────────────────────────────────────
        original_selection = list(context.selected_objects)
        original_active = context.view_layer.objects.active

        if self.export_scope == 'ALL_MESH':
            bpy.ops.object.select_all(action='DESELECT')
            meshes = [o for o in context.view_layer.objects
                      if o.type == 'MESH' and not o.hide_get()]
            for o in meshes:
                o.select_set(True)
            if meshes:
                context.view_layer.objects.active = meshes[0]
        else:
            meshes = [o for o in context.selected_objects
                      if o.type == 'MESH']
            if not meshes:
                self.report({'ERROR'}, "No mesh objects selected.")
                return {'CANCELLED'}

        # ── Output path ───────────────────────────────────────────────
        output_base = get_dir_path(context, "baked")
        os.makedirs(output_base, exist_ok=True)

        # Build filename from first object name
        base_name = meshes[0].name if meshes else "export"

        if self.export_format == 'FBX':
            ext = ".fbx"
        elif self.export_format == 'GLTF_SEPARATE':
            ext = ".gltf"
        else:
            ext = ".glb"
        filepath = os.path.join(output_base, f"{base_name}{ext}")

        # ── Apply transforms if requested ─────────────────────────────
        if self.apply_transforms:
            for o in meshes:
                o.select_set(True)
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True)

        # ── Export ────────────────────────────────────────────────────
        try:
            if self.export_format in ('GLB', 'GLTF_SEPARATE'):
                bpy.ops.export_scene.gltf(
                    filepath=filepath,
                    export_format=self.export_format,
                    use_selection=True,
                    export_materials='EXPORT',
                    export_image_format='AUTO',
                    export_tangents=True,
                    export_yup=True,
                    export_apply=False,
                    export_animations=self.export_animations,
                )
            else:  # FBX
                bpy.ops.export_scene.fbx(
                    filepath=filepath,
                    use_selection=True,
                    apply_scale_options='FBX_SCALE_ALL',
                    path_mode='COPY',
                    embed_textures=True,
                    bake_anim=self.export_animations,
                    mesh_smooth_type='FACE',
                )

            self.report({'INFO'},
                        f"Exported to {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            import traceback
            traceback.print_exc()

        # ── Restore selection ─────────────────────────────────────────
        bpy.ops.object.select_all(action='DESELECT')
        for o in original_selection:
            try:
                o.select_set(True)
            except ReferenceError:
                pass
        if original_active:
            try:
                context.view_layer.objects.active = original_active
            except ReferenceError:
                pass

        return {'FINISHED'}