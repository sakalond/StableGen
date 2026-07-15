"""Mask processing, render exports, visibility, texture baking, and material helpers."""

import os
import math
import bpy, bmesh  # pylint: disable=import-error
import numpy as np
import mathutils
from ..utils import get_file_path, get_dir_path, get_compositor_node_tree, configure_output_node_paths, get_eevee_engine_id, sg_modal_active, remove_empty_dirs
from PIL import Image
import cv2

_ADDON_PKG = __package__.rsplit('.', 1)[0]


def _snapshot_material_state(obj):
    """Capture exact slot order, active slot, and per-polygon assignments."""
    return {
        "materials": list(obj.data.materials),
        "active_index": int(obj.active_material_index),
        "polygon_indices": [polygon.material_index for polygon in obj.data.polygons],
    }


def _restore_material_state(obj, state):
    """Restore a material snapshot without active-first slot reordering."""
    obj.data.materials.clear()
    for material in state["materials"]:
        obj.data.materials.append(material)
    if obj.data.materials:
        obj.active_material_index = min(
            state["active_index"], len(obj.data.materials) - 1
        )
    polygon_indices = state["polygon_indices"]
    if len(polygon_indices) == len(obj.data.polygons):
        max_index = max(len(obj.data.materials) - 1, 0)
        for polygon, material_index in zip(obj.data.polygons, polygon_indices):
            polygon.material_index = min(material_index, max_index)
    else:
        print(
            f"[StableGen] Warning: polygon count changed while exporting '{obj.name}'; "
            "material slots were restored but polygon indices could not be."
        )


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
        print(f"[StableGen] {log_prefix} mask_file_path must be a string, got {type(mask_file_path)}")
        return

    if not os.path.exists(mask_file_path):
        print(f"[StableGen] {log_prefix} mask file not found: {mask_file_path}")
        return

    img = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[StableGen] {log_prefix} failed to read mask: {mask_file_path}")
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
    original_use_compositing = context.scene.render.use_compositing
    original_filepath = context.scene.render.filepath

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
    if context.scene.render.engine != 'CYCLES':
        context.scene.render.engine = 'CYCLES'
    context.scene.render.film_transparent = False
    context.scene.cycles.samples = 1
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.view_transform = 'Raw'

    view_layer = context.view_layer
    view_layer.use_pass_emit = True
    view_layer.use_pass_environment = True

    # ── Compositor ──────────────────────────────────────────────────────────
    context.scene.render.use_compositing = True
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
    if context.scene.render.engine != original_engine:
        context.scene.render.engine = original_engine
    context.scene.render.film_transparent = original_transparent
    context.scene.cycles.samples = original_samples
    bpy.context.scene.view_settings.view_transform = original_view_transform
    context.scene.render.use_compositing = original_use_compositing
    context.scene.render.filepath = original_filepath
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
        print("[StableGen] Exporting emit render")
        # Set animation frame to 1
        bpy.context.scene.frame_set(1)

        # Store original materials and create temporary ones
        original_materials = {}
        temporary_materials = {}

        # Check if there is BSDF applied

        # We need to temporarily disconnect BDSF nodes and connect their inputs directly to the output

        for obj in to_export:
            # Store original materials
            original_materials[obj] = _snapshot_material_state(obj)

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
        output_file = f"ctx_render{camera_id}" if camera_id is not None else "ctx_render"

        # Store and set world settings
        world = context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            context.scene.world = world

        # Store original settings
        original_engine = context.scene.render.engine
        original_film_transparent = context.scene.render.film_transparent
        original_use_compositing = context.scene.render.use_compositing
        original_filepath = context.scene.render.filepath
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
        if context.scene.render.engine != 'CYCLES':
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
        context.scene.render.use_compositing = True
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
        if context.scene.render.engine != original_engine:
            context.scene.render.engine = original_engine
        context.scene.render.film_transparent = original_film_transparent
        context.scene.render.use_compositing = original_use_compositing
        context.scene.render.filepath = original_filepath
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
        for obj, material_state in original_materials.items():
            _restore_material_state(obj, material_state)

        # Clean up temporary materials
        for _, temp_mat in temporary_materials.items():
            if temp_mat and temp_mat.name in bpy.data.materials:
                bpy.data.materials.remove(temp_mat)

        print(f"[StableGen] Emmision render saved to: {os.path.join(output_dir, output_file)}.png")


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
    print("[StableGen] Exporting render using Workbench")

    # Store original materials and create temporary ones
    original_materials = {}
    temporary_materials = []
    for obj in context.view_layer.objects:
        if obj.type == 'MESH':
            # Store original materials
            original_materials[obj] = _snapshot_material_state(obj)

            # Create temporary material
            temp_mat = bpy.data.materials.new(name="TempRenderMaterial")
            temporary_materials.append(temp_mat)
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
    original_use_compositing = context.scene.render.use_compositing

    # Switch to WORKBENCH render engine and configure settings
    if context.scene.render.engine != 'BLENDER_WORKBENCH':
        context.scene.render.engine = 'BLENDER_WORKBENCH'
    # Configure Workbench for a flat, consistent look if needed
    context.scene.display.shading.light = 'STUDIO'
    context.scene.display.shading.color_type = 'SINGLE'

    render_layer = context.view_layer
    original_combined = render_layer.use_pass_combined

    # Enable combined pass for Workbench
    render_layer.use_pass_combined = True

    # Set up output nodes (Compositor setup remains the same)
    context.scene.render.use_compositing = True
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
    for obj, material_state in original_materials.items():
        _restore_material_state(obj, material_state)

    # Restore original render settings
    render_layer.use_pass_combined = original_combined

    # Clean up temporary materials
    for mat in temporary_materials:
        if mat.name in bpy.data.materials:
            bpy.data.materials.remove(mat)


    # Restore original render settings
    if context.scene.render.engine != original_engine:
        context.scene.render.engine = original_engine
    context.scene.display.shading.light = original_workbench_settings['lighting']
    context.scene.display.shading.color_type = original_workbench_settings['color_type']
    context.scene.render.filepath = original_render_filepath
    context.scene.render.image_settings.file_format = original_image_settings
    context.scene.render.use_compositing = original_use_compositing


    print(f"[StableGen] Render saved to: {os.path.join(output_dir, output_file)}0001.png") # Blender adds frame number


def export_viewport(context, camera_id=None, output_dir=None, filename=None):
    """
    Renders the scene using viewport OpenGL render to include overlays.
    :param context: Blender context.
    :param camera_id: ID of the camera for the output filename.
    :param output_dir: Optional output directory.
    :param filename: Optional filename (without component/frame suffix).
    :return: None
    """
    print("[StableGen] Exporting render using Viewport (OpenGL)")

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
    if context.scene.render.engine != 'BLENDER_WORKBENCH':
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
        print("[StableGen] Viewport render failed: no VIEW_3D area found.")
        if context.scene.render.engine != original_engine:
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
    if context.scene.render.engine != original_engine:
        context.scene.render.engine = original_engine
    context.scene.render.filepath = original_render_filepath
    context.scene.render.image_settings.file_format = original_image_settings

    print(f"[StableGen] Viewport render saved to: {os.path.join(output_dir, output_file)}.png")

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

    print(f"[StableGen] Canny edge detection saved to: {os.path.join(output_dir_canny, output_file)}.png")


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
        print(f"[StableGen] Error: mask_file_path must be a string. Got: {type(mask_file_path)}")
        return None
    if not os.path.exists(mask_file_path):
        print(f"[StableGen] Error: Mask file not found at {mask_file_path}")
        return None
    if not os.path.isfile(mask_file_path):
         print(f"[StableGen] Error: Path provided is not a file: {mask_file_path}")
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
            print(f"[StableGen] Warning: Unsupported mask dtype {mask_array.dtype} after loading. Trying to convert.")
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
        print(f"[StableGen] Error: Mask file not found at {mask_file_path}")
        return None
    except Exception as e:
        print(f"[StableGen] Error loading or processing mask file {mask_file_path}: {e}")
        return None


def export_visibility(context, to_export, obj=None, camera_visibility=None,
                      prepare_only=False, camera_render=None, camera_id=None):
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
    temporary_materials = {}

    def restore_prepared_materials():
        for prepared_obj, material_state in original_materials.items():
            _restore_material_state(prepared_obj, material_state)
        for temp_mat in temporary_materials.values():
            if temp_mat and temp_mat.name in bpy.data.materials:
                bpy.data.materials.remove(temp_mat)

    def prepare_material(obj):
        mat = obj.active_material
        if not mat:
            return False

        # Store original materials
        original_materials[obj] = _snapshot_material_state(obj)

        # Store original active mat

        # Copy active material and switch to it
        mat = obj.active_material
        if not mat:
            return False
        mat_copy = mat.copy()

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
            bpy.data.materials.remove(mat_copy)
            return False
        
        if not output.inputs[0].links:
            bpy.data.materials.remove(mat_copy)
            return False
        
        # Determine which input to used based on existence of BSDF node before the output
        if output.inputs[0].links and output.inputs[0].links[0].from_node.type == 'BSDF_PRINCIPLED':
            principled = output.inputs[0].links[0].from_node
            if not principled.inputs[0].links:
                # Principled BSDF exists but nothing is connected to its
                # Base Color – this is not a projection material.
                bpy.data.materials.remove(mat_copy)
                return False
            color_mix = principled.inputs[0].links[0].from_node
            input = principled.inputs[0]
        else:
            color_mix = output.inputs[0].links[0].from_node
            input = output.inputs[0]

        # Validation succeeded; only now replace the object's slots. This
        # prevents an early return from leaving a complex mesh modified.
        obj.data.materials.clear()
        obj.data.materials.append(mat_copy)
        temporary_materials[obj] = mat_copy
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
            if not color_mix.inputs[0].links:
                break
            color_ramp = color_mix.inputs[0].links[0].from_node
            if not color_ramp.inputs[0].links:
                # Color ramp Fac is unlinked (constant value) — remove it and
                # fall through so the rest of the chain is wired correctly.
                nodes.remove(color_ramp)
                break
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
                    print(f"[StableGen]   - Warning: Failed to set Power for node '{node.name}'. Error: {e}")
            # Also handle native Raycast path (Blender 5.1+) where power is a MATH POWER node
            elif node.type == 'MATH' and node.operation == 'POWER' and node.label == 'power_weight':
                try:
                    node.inputs[1].default_value = context.scene.weight_exponent if context.scene.weight_exponent_mask else 1.0
                except Exception as e:
                    print(f"[StableGen]   - Warning: Failed to set Power for native node '{node.name}'. Error: {e}")

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
                        print(f"[StableGen]   - Warning: Failed to set power_weight '{node.name}'. Error: {e}")
                elif node.type == 'SCRIPT' and "Power" in node.inputs:
                    try:
                        node.inputs["Power"].default_value = target_exp
                    except Exception as e:
                        print(f"[StableGen]   - Warning: Failed to set OSL Power '{node.name}'. Error: {e}")

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
                        print(f"[StableGen]   - Warning: Failed to bypass SharpW '{node.name}'. Error: {e}")

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
                        print(f"[StableGen]   - Warning: Failed to bypass NormW '{node.name}'. Error: {e}")
               
        return True
    
    # Prepare the material

    if not obj:
        # Prepare for all objects
        for obj in to_export:
            if not prepare_material(obj):
                restore_prepared_materials()
                return False
    else:
        if not prepare_material(obj):
            restore_prepared_materials()
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
        if camera_render is None:
            from .view_plan import order_cameras
            cameras = order_cameras(
                obj for obj in context.scene.objects if obj.type == 'CAMERA'
            )
            camera_visibility_index = cameras.index(camera_visibility)
            camera_render_index = (camera_visibility_index + 1) % len(cameras)
            camera_render = cameras[camera_render_index]
        else:
            camera_render_index = camera_id
        context.scene.camera = camera_render
        export_emit_image(context, to_export, camera_id=f"{camera_render_index}_visibility", bg_color=(1, 1, 1), view_transform='Raw', fallback_color=(1,1,1))

    # Restore original materials
    restore_prepared_materials()

    return True


# =========================================================
# Camera Placement Helper Functions
# =========================================================



# ---- Baking ---------------------------------------------------------------


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

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

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
    if bpy.context.scene.render.engine != 'CYCLES':
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


_SG_BAKE_UV_NAME = "BakeUV"
_SG_PROJECTION_BUFFER_UV_NAME = "_SG_ProjectionBuffer"


def _is_projection_or_buffer_uv(name):
    return name.startswith("ProjectionUV") or name == _SG_PROJECTION_BUFFER_UV_NAME


def _first_source_uv_name(obj):
    if not obj or not obj.data.uv_layers:
        return None
    for uv in obj.data.uv_layers:
        if uv.name != _SG_BAKE_UV_NAME and not _is_projection_or_buffer_uv(uv.name):
            return uv.name
    return None


def _ensure_bake_uv(obj):
    uv_layers = obj.data.uv_layers
    bake_uv = uv_layers.get(_SG_BAKE_UV_NAME)
    if bake_uv is None:
        source_uv = _first_source_uv_name(obj)
        if source_uv and uv_layers.get(source_uv):
            uv_layers.active = uv_layers[source_uv]
        bake_uv = uv_layers.new(name=_SG_BAKE_UV_NAME)
        print(f"[StableGen] Created {_SG_BAKE_UV_NAME} on {obj.name}")
    uv_layers.active = bake_uv
    return bake_uv


def _set_bake_uv_active(obj):
    uv_layers = obj.data.uv_layers
    bake_uv = uv_layers.get(_SG_BAKE_UV_NAME)
    if bake_uv:
        uv_layers.active = bake_uv
        return bake_uv
    for uv in uv_layers:
        if not _is_projection_or_buffer_uv(uv.name):
            uv_layers.active = uv
            return uv
    return _ensure_bake_uv(obj)


def _pin_unlinked_image_textures_to_uv(nodes, links, uv_name, exclude_nodes=()):
    if not uv_name:
        return None

    exclude_nodes = set(exclude_nodes or ())
    image_nodes = [
        node for node in nodes
        if node.type == 'TEX_IMAGE'
        and node not in exclude_nodes
        and node.inputs.get("Vector")
        and not node.inputs["Vector"].links
    ]
    if not image_nodes:
        return None

    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_name
    uv_node.label = f"StableGen Source UV ({uv_name})"
    for image_node in image_nodes:
        links.new(uv_node.outputs["UV"], image_node.inputs["Vector"])
    print(
        f"[StableGen] Pinned {len(image_nodes)} source image texture node(s) "
        f"to UV map '{uv_name}' for baking"
    )
    return uv_node


def _copy_material_slots_for_bake(obj):
    original_materials = list(obj.data.materials)
    original_active_index = obj.active_material_index
    copied_materials = []

    obj.data.materials.clear()
    for mat in original_materials:
        if mat:
            mat_copy = mat.copy()
            obj.data.materials.append(mat_copy)
            copied_materials.append(mat_copy)

    return original_materials, original_active_index, copied_materials


def _restore_material_slots_after_bake(obj, original_materials, original_active_index, copied_materials):
    obj.data.materials.clear()
    for mat in original_materials:
        if mat:
            obj.data.materials.append(mat)
    if obj.data.materials:
        obj.active_material_index = min(max(0, original_active_index), len(obj.data.materials) - 1)
    for mat in copied_materials:
        if mat and mat.name in bpy.data.materials:
            bpy.data.materials.remove(mat)


def _first_principled_node(mat):
    if not mat or not mat.use_nodes:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def _socket_gray_rgba(socket, fallback=0.0):
    try:
        value = socket.default_value
    except Exception:
        value = fallback
    if isinstance(value, (float, int)):
        scalar = float(value)
    else:
        scalar = float(value[0]) if value else float(fallback)
    return (scalar, scalar, scalar, 1.0)


def _pbr_channel_default_rgba(mat, channel_name):
    principled = _first_principled_node(mat)
    if channel_name == 'base_color':
        if principled and principled.inputs.get("Base Color"):
            return tuple(principled.inputs["Base Color"].default_value)
        return (1.0, 1.0, 1.0, 1.0)
    if channel_name == 'roughness':
        if principled and principled.inputs.get("Roughness"):
            return _socket_gray_rgba(principled.inputs["Roughness"], 0.5)
        return (0.5, 0.5, 0.5, 1.0)
    if channel_name == 'metallic':
        if principled and principled.inputs.get("Metallic"):
            return _socket_gray_rgba(principled.inputs["Metallic"], 0.0)
        return (0.0, 0.0, 0.0, 1.0)
    if channel_name == 'ao':
        return (1.0, 1.0, 1.0, 1.0)
    return (0.0, 0.0, 0.0, 1.0)


def _is_bone_custom_shape_object(obj):
    for armature in bpy.context.view_layer.objects:
        if armature.type != 'ARMATURE' or not armature.pose:
            continue
        for bone in armature.pose.bones:
            if bone.custom_shape == obj:
                return True
    return False


def _is_gltf_not_exported_helper(obj):
    for collection in getattr(obj, "users_collection", []):
        if "not_exported" in collection.name.lower():
            return True
    return False


def _iter_collection_objects_recursive(collection):
    seen = set()

    def visit(col):
        for obj in col.objects:
            if obj.name in seen:
                continue
            seen.add(obj.name)
            yield obj
        for child in col.children:
            yield from visit(child)

    yield from visit(collection)


def _disable_not_exported_render_helpers(context=None):
    """Prevent glTF helper/control geometry from occluding texture bakes."""
    disabled_objects = 0
    disabled_collections = 0

    for collection in bpy.data.collections:
        if "not_exported" not in collection.name.lower():
            continue

        if not collection.hide_render:
            collection.hide_render = True
            disabled_collections += 1

        for obj in _iter_collection_objects_recursive(collection):
            if not obj.hide_render:
                obj.hide_render = True
                disabled_objects += 1

    if disabled_objects or disabled_collections:
        print(
            "[StableGen] Disabled render visibility for "
            f"{disabled_objects} glTF not_exported helper object(s) "
            f"across {disabled_collections} collection(s)."
        )

    if context:
        context.view_layer.update()


def _object_visible_in_view_layer(obj):
    if obj.hide_get():
        return False
    try:
        return obj.visible_get(view_layer=bpy.context.view_layer)
    except TypeError:
        try:
            return obj.visible_get()
        except Exception:
            return True
    except Exception:
        return True


def _is_texture_bake_target(obj):
    if obj is None or obj.type != 'MESH':
        return False
    if not _object_visible_in_view_layer(obj) or obj.hide_render:
        return False
    if _is_gltf_not_exported_helper(obj):
        return False
    if _is_bone_custom_shape_object(obj):
        return False
    if not obj.data or not obj.data.polygons:
        return False
    return any(mat for mat in obj.data.materials)


def _texture_bake_targets(objects):
    targets = []
    skipped = []
    for obj in objects:
        if _is_texture_bake_target(obj):
            targets.append(obj)
        elif obj and obj.type == 'MESH':
            skipped.append(obj)
    return targets, skipped


def _activate_mesh_for_object_ops(obj):
    """Select and activate a mesh object for bpy.ops calls."""
    if obj is None or obj.type != 'MESH':
        raise RuntimeError("Bake target is not a mesh object")

    if not any(view_obj == obj for view_obj in bpy.context.view_layer.objects):
        raise RuntimeError(
            f"Bake target '{getattr(obj, 'name', '<unknown>')}' is not in the active view layer"
        )

    if obj.hide_get():
        obj.hide_set(False)
    if obj.hide_select:
        obj.hide_select = False

    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.context.view_layer.update()

    if bpy.context.view_layer.objects.active != obj:
        raise RuntimeError(f"Could not activate bake target '{obj.name}'")

    return {
        'active_object': obj,
        'object': obj,
        'selected_objects': [obj],
        'selected_editable_objects': [obj],
    }


def unwrap(obj, method, overlap_only):
        """     
        Unwraps the UVs of the given object using the selected method.         
        :param obj: Blender object.         
        :return: None     
        """
        if method == 'none':
            return
        
        op_context = _activate_mesh_for_object_ops(obj)

        _ensure_bake_uv(obj)

        with bpy.context.temp_override(**op_context):
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
        _disable_not_exported_render_helpers(context)
        if _is_gltf_not_exported_helper(obj) or _is_bone_custom_shape_object(obj):
            print(f"[StableGen] Skipping helper mesh during bake: {obj.name}")
            return False
        bpy.context.scene.display_settings.display_device = 'sRGB'
        # Backup original view transform
        original_view_transform = bpy.context.scene.view_settings.view_transform
        # Backup whether the object was enabled for rendering
        original_render = obj.hide_render
    
        bpy.context.scene.view_settings.view_transform = view_transform
        # Set the object to be rendered
        obj.hide_render = False

        source_uv_name = _first_source_uv_name(obj)
        _set_bake_uv_active(obj)

        # Create a new image for baking
        image_name = f"{obj.name}_baked{suffix}" 
        image = bpy.data.images.new(name=image_name, width=texture_resolution, height=texture_resolution)

        if not obj.data.materials:
            print("[StableGen] No materials found")
            return False

        original_materials, original_active_index, copied_materials = _copy_material_slots_for_bake(obj)
        if not copied_materials:
            _restore_material_slots_after_bake(
                obj, original_materials, original_active_index, copied_materials)
            print("[StableGen] No material slots found")
            return False

        op_context = _activate_mesh_for_object_ops(obj)
        _set_bake_uv_active(obj)

        for mat in copied_materials:
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            tex_image = nodes.new("ShaderNodeTexImage")
            tex_image.image = image
            tex_image.location = (0, 0)
            nodes.active = tex_image

            _pin_unlinked_image_textures_to_uv(
                nodes, links, source_uv_name, exclude_nodes={tex_image})
        
            # Check if there is a BSDF node before the output node
            output = None
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
        with bpy.context.temp_override(**op_context):
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

        _restore_material_slots_after_bake(
            obj, original_materials, original_active_index, copied_materials)
                
        return True

def _has_non_projection_uv(obj):
    """Return True if the object has at least one UV map that can be used
    for baking (not a StableGen projection UV or buffer)."""
    if not obj or obj.type != 'MESH':
        return False
    return obj.data.uv_layers.get(_SG_BAKE_UV_NAME) is not None

def _uvs_likely_overlap(obj, uv_name=None):
    me = obj.data
    if not me.uv_layers:
        return False
    if uv_name is None:
        uv_name = _SG_BAKE_UV_NAME
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


def _has_pbr_channels(mat):
    """Check if the material contains any PBR channels other than base_color,
    and that a Principled BSDF is actually active/wired to the material output.
    """
    if not mat or not mat.use_nodes:
        return False

    nodes = mat.node_tree.nodes
    output_node = None
    for n in nodes:
        if n.type == 'OUTPUT_MATERIAL':
            output_node = n
            break

    if not output_node or not output_node.inputs["Surface"].links:
        return False

    # Check if a node is (or leads to) an active Principled BSDF
    def check_is_pbr_active(node, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return False
        visited.add(node)

        if node.type == 'BSDF_PRINCIPLED':
            return True
        if node.type == 'MIX_SHADER':
            for inp in node.inputs:
                if inp.links:
                    if check_is_pbr_active(inp.links[0].from_node, visited):
                        return True
        return False

    from_node = output_node.inputs["Surface"].links[0].from_node
    if not check_is_pbr_active(from_node):
        return False

    sources = _find_pbr_bake_sources(mat)
    return any(ch in sources for ch in ('roughness', 'metallic', 'normal', 'emission', 'height', 'ao'))


def _object_pbr_channels(obj):
    channels = []
    for mat in obj.data.materials:
        if mat and _has_pbr_channels(mat):
            for channel in _find_pbr_bake_sources(mat):
                if channel not in channels:
                    channels.append(channel)
    return channels


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

    _disable_not_exported_render_helpers(context)
    if _is_gltf_not_exported_helper(obj) or _is_bone_custom_shape_object(obj):
        print(f"[StableGen] Skipping helper mesh during PBR bake: {obj.name}")
        return None

    # ── Save state ─────────────────────────────────────────────────
    orig_vt = context.scene.view_settings.view_transform
    orig_render = obj.hide_render
    context.scene.view_settings.view_transform = 'Standard'
    obj.hide_render = False
    source_uv_name = _first_source_uv_name(obj)
    _set_bake_uv_active(obj)

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
    op_context = _activate_mesh_for_object_ops(obj)
    _pin_unlinked_image_textures_to_uv(
        nodes, links, source_uv_name, exclude_nodes={tex_node})
    _set_bake_uv_active(obj)

    # ── Bake ─────────────────────────────────────────────────────
    # Ensure OBJECT mode — bake fails silently in EDIT mode
    with bpy.context.temp_override(**op_context):
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
    print(f"[StableGen]     Saved PBR channel: {out_path}")

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
        print(f"[StableGen]     Flipped normal green channel (DirectX): {filepath}")
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
    print(f"[StableGen]     Packed ORM texture: {orm_path}")
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

    samples: bpy.props.IntProperty(
        name="Render Samples",
        description="Number of samples to use for baking. Higher values reduce noise but take longer",
        default=16,
        min=1,
        max=4096
    ) # type: ignore

    use_denoise: bpy.props.BoolProperty(
        name="Denoise Baked Textures",
        description="Use Cycles' built-in denoiser to clean up noise in baked textures",
        default=True
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
        default='smart'
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
        if sg_modal_active(context):
            self.poll_message_set("Another operation is in progress")
            return False
        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        if not os.path.exists(addon_prefs.output_dir):
            self.poll_message_set("Output directory not set or does not exist (check addon preferences)")
            return False
        return True

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "texture_resolution")
        layout.prop(self, "samples")
        layout.prop(self, "use_denoise")
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
            candidates = [obj for obj in context.view_layer.objects if obj.type == 'MESH']
        else: # 'selected'
            candidates = [obj for obj in context.selected_objects if obj.type == 'MESH']
        self._objects, skipped = _texture_bake_targets(candidates)
        if skipped:
            names = ", ".join(obj.name for obj in skipped[:5])
            more = "" if len(skipped) <= 5 else f", +{len(skipped) - 5} more"
            print(
                f"[StableGen] Skipping {len(skipped)} non-texture helper mesh(es) "
                f"during bake: {names}{more}"
            )
        if not self._objects:
            self.report({'ERROR'}, "No textureable mesh objects found to bake.")
            return {'CANCELLED'}
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

        _disable_not_exported_render_helpers(context)
        self._objects, skipped = _texture_bake_targets(self._objects)
        if skipped:
            names = ", ".join(obj.name for obj in skipped[:5])
            more = "" if len(skipped) <= 5 else f", +{len(skipped) - 5} more"
            print(
                f"[StableGen] Skipping {len(skipped)} non-texture helper mesh(es) "
                f"during bake: {names}{more}"
            )
        if not self._objects:
            self.report({'ERROR'}, "No textureable mesh objects found to bake.")
            return {'CANCELLED'}

        self.original_engine = bpy.context.scene.render.engine
        if bpy.context.space_data and hasattr(bpy.context.space_data, 'shading'):
            self.original_shading = bpy.context.space_data.shading.type
        else:
            self.original_shading = 'SOLID'
        self.original_device = bpy.context.scene.cycles.device
        self.original_samples = bpy.context.scene.cycles.samples
        self.original_max_bounces = bpy.context.scene.cycles.max_bounces
        self.original_persistent = bpy.context.scene.render.use_persistent_data
        if hasattr(bpy.context.scene.cycles, 'use_denoising'):
            self.original_use_denoising = bpy.context.scene.cycles.use_denoising
        else:
            self.original_use_denoising = False
        # Save Cycles compute_device_type so we can restore after GPU bake
        try:
            self._original_compute_device_type = (
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type
            )
        except Exception:
            self._original_compute_device_type = None
        # Set render engine to CYCLES (required for baking)
        if bpy.context.scene.render.engine != 'CYCLES':
            bpy.context.scene.render.engine = 'CYCLES'
        # Switch to Solid shading in ALL viewports to avoid conflicts with Rendered mode
        # (Rendered viewport + bake = crash). Solid is engine-independent
        # and won't cause pink textures like Material Preview (EEVEE) does.
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.type = 'SOLID'
        prepare_baking(context)
        # Apply user's custom bake samples and denoising configuration
        bpy.context.scene.cycles.samples = self.samples
        if hasattr(bpy.context.scene.cycles, 'use_denoising'):
            bpy.context.scene.cycles.use_denoising = self.use_denoise

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
                    # Fall back to standard color/diffuse bake if PBR channels are absent
                    has_pbr = False
                    if self.bake_pbr and obj.active_material:
                        has_pbr = _has_pbr_channels(obj.active_material)

                    if has_pbr:
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

                        # Copy fallback diffuse bake to PBR BaseColor if bake_pbr is True
                        if self.bake_pbr:
                            baked_dir = get_dir_path(context, "baked")
                            src_path = os.path.join(baked_dir, f"{obj.name}.png")
                            dst_path = os.path.join(baked_dir, f"{obj.name}_BaseColor.png")
                            if os.path.exists(src_path):
                                import shutil
                                try:
                                    shutil.copy(src_path, dst_path)
                                    print(f"[StableGen] Copied fallback diffuse bake to PBR BaseColor: {dst_path}")
                                except Exception as e:
                                    print(f"[StableGen] Failed to copy fallback diffuse bake to PBR BaseColor: {e}")

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
                            if obj.active_material and _has_pbr_channels(obj.active_material):
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
                    
                    # Capture original settings in local variables to avoid closures referencing self,
                    # as self (the operator instance) may be destroyed by Blender when modal exits.
                    orig_engine = str(self.original_engine)
                    orig_device = str(self.original_device)
                    orig_samples = int(self.original_samples)
                    orig_max_bounces = int(self.original_max_bounces)
                    orig_persistent = bool(self.original_persistent)
                    orig_use_denoising = bool(self.original_use_denoising)
                    orig_compute_device = self._original_compute_device_type
                    if orig_compute_device is not None:
                        orig_compute_device = str(orig_compute_device)
                    orig_shading = str(self.original_shading)
                    
                    # Define defer restore function to run outside modal loop
                    def defer_restore():
                        try:
                            # 1. Disable Cycles persistent data to release caches
                            bpy.context.scene.render.use_persistent_data = False
                            
                            # 2. Switch Cycles to CPU device to free GPU contexts and memory
                            bpy.context.scene.cycles.device = 'CPU'
                            
                            # 3. Update view layer to ensure Cycles releases everything
                            bpy.context.view_layer.update()
                            
                            # 4. Now restore the render engine
                            if bpy.context.scene.render.engine != orig_engine:
                                bpy.context.scene.render.engine = orig_engine
                            
                            # 5. Restore other scene/Cycles settings
                            bpy.context.scene.cycles.device = orig_device
                            bpy.context.scene.cycles.samples = orig_samples
                            bpy.context.scene.cycles.max_bounces = orig_max_bounces
                            bpy.context.scene.render.use_persistent_data = orig_persistent
                            if hasattr(bpy.context.scene.cycles, 'use_denoising'):
                                bpy.context.scene.cycles.use_denoising = orig_use_denoising
                            if orig_compute_device is not None:
                                try:
                                    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = orig_compute_device
                                except Exception:
                                    pass
                            
                            # 6. Restore original viewport shading in all screens
                            for window in bpy.context.window_manager.windows:
                                for area in window.screen.areas:
                                    if area.type == 'VIEW_3D':
                                        for space in area.spaces:
                                            if space.type == 'VIEW_3D':
                                                space.shading.type = orig_shading
                                                break
                            remove_empty_dirs(bpy.context)
                        except Exception as e:
                            print(f"[StableGen] Error in defer_restore: {e}")
                        return None
                    
                    import bpy
                    bpy.app.timers.register(defer_restore, first_interval=0.2)
                    
                    self.report({'INFO'}, "Textures baked successfully.")
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
        slot_idx = len(obj.material_slots) - 1
        obj.active_material_index = slot_idx
        obj.data.polygons.foreach_set("material_index", [slot_idx] * len(obj.data.polygons))
        obj.data.update()

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

            links.new(tex_base.outputs["Color"],
                      principled.inputs["Base Color"])
            links.new(principled.outputs[0],
                      output_node.inputs["Surface"])
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
