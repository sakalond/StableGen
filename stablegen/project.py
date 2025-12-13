import bpy
import os
from math import atan, tan

from .utils import get_last_material_index, get_file_path, get_dir_path
from .render_tools import prepare_baking, bake_texture, unwrap
from mathutils import Vector

import bpy
import os
from math import atan, tan

from .utils import get_last_material_index, get_file_path, get_dir_path
from .render_tools import prepare_baking, bake_texture, unwrap
from mathutils import Vector


def project_image(context, to_project, mat_id, stop_index=1000000):
    """
    Projects an image onto all mesh objects using UV Project Modifier.
    """

    scene = context.scene
    render = scene.render

    # Common flags / shortcuts
    gen_method = scene.generation_method
    gen_mode = scene.generation_mode
    is_sequential = gen_method == "sequential"
    is_refine = gen_method == "refine"
    refine_preserve = scene.refine_preserve
    bake_texture_flag = scene.bake_texture
    overwrite_material = scene.overwrite_material
    regenerate_selected = gen_mode == "regenerate_selected"

    # -------------------------------------------------------------------------
    # Nested helper: build mix tree for image shaders
    # -------------------------------------------------------------------------
    def build_mix_tree(
        shaders,
        weight_nodes,
        nodes_collection,
        links,
        last_node=None,
        level=0,
        stop_index=1000000,
    ):
        """
        Recursively builds a binary tree of mix shader nodes.

        The mix factor between two shader nodes is computed dynamically using
        the outputs of weight_nodes (e.g. SCRIPT/LESS_THAN nodes).

        Returns:
            (final_shader_node, final_weight_node)
        """

        # Compute offsets based on recursion level
        # (exact math preserved, just made more readable)
        step = 800 if scene.early_priority else 600
        if refine_preserve:
            base = (last_node.location[0] + 1200) if last_node else 1200
        else:
            base = 1000
        x_offset = base + level * step
        y_offset = 0

        # ---------------------------------------------------------------------
        # Base case: only one shader left -> mix against fallback color
        # ---------------------------------------------------------------------
        if len(shaders) == 1:
            final_mix = nodes_collection.new("ShaderNodeMixRGB")
            final_mix.location = (x_offset - 200, y_offset)

            # Convert fallback_color from (r,g,b) -> (r,g,b,a)
            final_mix.inputs["Color2"].default_value = (*scene.fallback_color, 1.0)

            # Connect the single color output to Color1
            links.new(shaders[0].outputs[0], final_mix.inputs["Color1"])

            if last_node:
                # Connect the previous node into Color2 if preserving
                links.new(last_node.outputs[0], final_mix.inputs["Color2"])

            # Compare/threshold node for driving the factor
            if is_refine and refine_preserve:
                compare_node = nodes_collection.new("ShaderNodeValToRGB")
                compare_node.location = (x_offset - 500, y_offset)
                cr = compare_node.color_ramp
                cr.elements[0].position = 0.0
                cr.elements[0].color = (1, 1, 1, 1)
                cr.elements[1].position = 0.6
                cr.elements[1].color = (0, 0, 0, 1)
                cr.interpolation = "LINEAR"
            else:
                compare_node = nodes_collection.new("ShaderNodeMath")
                compare_node.operation = "COMPARE"
                compare_node.inputs[1].default_value = 0.0  # value2
                compare_node.inputs[2].default_value = 0.0  # epsilon
                compare_node.location = (x_offset - 500, y_offset)

            # Connect final weight node to compare node
            links.new(weight_nodes[0].outputs[0], compare_node.inputs[0])
            links.new(compare_node.outputs[0], final_mix.inputs["Fac"])

            if not scene.apply_bsdf:
                return final_mix, compare_node

            # Decide whether to reuse existing Principled or make a new one
            should_add_principled = True
            if is_refine and refine_preserve and last_node is not None:
                # Final Principled is at last_node's output if BSDF
                final_principled = last_node.outputs[0].links[0].to_node
                if final_principled.type == "BSDF_PRINCIPLED":
                    should_add_principled = False

            if should_add_principled:
                final_principled = nodes_collection.new("ShaderNodeBsdfPrincipled")
                final_principled.location = (x_offset, y_offset)
                final_principled.inputs["Roughness"].default_value = 1.0

            links.new(final_mix.outputs[0], final_principled.inputs[0])
            return final_principled, compare_node

        # ---------------------------------------------------------------------
        # Recursive case: pair up shaders and merge them in a tree
        # ---------------------------------------------------------------------
        new_shaders = []
        new_weight_nodes = []
        i = 0

        while i < len(shaders):
            if i + 1 < len(shaders):
                vert_offset = -200 * (i // 2)

                # Sum the weights
                sum_node = nodes_collection.new("ShaderNodeMath")
                sum_node.operation = "ADD"
                sum_node.location = (x_offset - (800 if scene.early_priority else 600), y_offset + vert_offset)
                links.new(weight_nodes[i].outputs[0], sum_node.inputs[0])
                links.new(weight_nodes[i + 1].outputs[0], sum_node.inputs[1])

                # Compute mix factor: weight_B / (weight_A + weight_B)
                div_node = nodes_collection.new("ShaderNodeMath")
                div_node.operation = "DIVIDE"
                div_node.location = (
                    x_offset - (600 if scene.early_priority else 400),
                    y_offset + vert_offset,
                )
                links.new(weight_nodes[i + 1].outputs[0], div_node.inputs[0])
                links.new(sum_node.outputs[0], div_node.inputs[1])

                if scene.early_priority:
                    map_range_node = nodes_collection.new("ShaderNodeMapRange")
                    map_range_node.location = (x_offset - 400, y_offset + vert_offset)
                    map_range_node.inputs[1].default_value = scene.early_priority_strength
                    links.new(div_node.outputs[0], map_range_node.inputs[0])
                    factor_node = map_range_node
                else:
                    factor_node = div_node

                # Mix node for RGBA colors
                mix_node = nodes_collection.new("ShaderNodeMixRGB")
                mix_node.location = (x_offset - 200, y_offset + vert_offset)
                mix_node.use_clamp = True

                links.new(shaders[i].outputs[0], mix_node.inputs["Color1"])
                links.new(shaders[i + 1].outputs[0], mix_node.inputs["Color2"])
                links.new(factor_node.outputs[0], mix_node.inputs["Fac"])

                new_shaders.append(mix_node)
                new_weight_nodes.append(sum_node)
                i += 2
            else:
                new_shaders.append(shaders[i])
                new_weight_nodes.append(weight_nodes[i])
                i += 1

        return build_mix_tree(
            new_shaders,
            new_weight_nodes,
            nodes_collection,
            links,
            last_node=last_node,
            level=level + 1,
            stop_index=stop_index,
        )

    # -------------------------------------------------------------------------
    # Camera list and UI refresh
    # -------------------------------------------------------------------------
    cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]
    cameras.sort(key=lambda x: x.name)

    # Force refresh of the UI
    for area in context.screen.areas:
        area.tag_redraw()

    # -------------------------------------------------------------------------
    # UV Projection + baking (per-camera, per-object)
    # -------------------------------------------------------------------------
    for i, camera in enumerate(cameras):
        for obj in to_project:
            # We can skip UV Project in some sequential cases
            if (not is_sequential) or stop_index == 0 or bake_texture_flag:
                # Deselect all objects
                bpy.ops.object.select_all(action="DESELECT")

                # Select object as active (needed for applying the modifier)
                context.view_layer.objects.active = obj
                obj.select_set(True)

                # Make object data single-user before applying modifier
                bpy.ops.object.make_single_user(object=True, obdata=True)

                if obj.data.users > 1:
                    print("Warning: Cannot make object data single user. Making a copy.")
                    obj.data = obj.data.copy()
                    obj.data.name = f"{obj.name}_data"
                    obj.data.update()
                    if obj.data.users > 1:
                        print("Error: Cannot make object data single user. Exiting.")
                        return Exception("Cannot make object data single user. Exiting.")

                # Determine UV map to use
                uv_map = None
                if overwrite_material and not bake_texture_flag:
                    for uv in obj.data.uv_layers:
                        if uv.name == f"ProjectionUV_{i}_{mat_id}":
                            uv_map = uv
                            break

                # If object has no UV map and we are baking textures, create a new one
                if not obj.data.uv_layers and bake_texture_flag:
                    obj.data.uv_layers.new(name="BakeUV")

                if uv_map is None:
                    uv_map = obj.data.uv_layers.new(name=f"ProjectionUV_{i}_{mat_id}")

                # Add UV Project modifier
                uv_project_mod = obj.modifiers.new(name="UVProject", type="UV_PROJECT")

                if not camera:
                    return False

                uv_project_mod.projectors[0].object = camera

                # Set UV layer
                try:
                    uv_project_mod.uv_layer = uv_map.name
                except Exception:
                    raise Exception(
                        "Not enough UV map slots. Please remove some UV maps."
                    )

                # Aspect ratio for projection
                aspect_ratio = render.resolution_x / render.resolution_y
                uv_project_mod.aspect_x = aspect_ratio if aspect_ratio > 1 else 1
                uv_project_mod.aspect_y = 1 / aspect_ratio if aspect_ratio < 1 else 1

                bpy.ops.object.modifier_apply(modifier=uv_project_mod.name)
                original_uv_map = obj.data.uv_layers[0]  # saved but used later

            # Baking path
            if bake_texture_flag:
                if stop_index > 0 and is_sequential:
                    bpy.ops.object.select_all(action="DESELECT")
                    context.view_layer.objects.active = obj
                    obj.select_set(True)

                if i <= stop_index and (not is_sequential or i == stop_index):
                    simple_project_bake(context, i, obj, mat_id)

                # Remove last UV map (the one we just added for this projection)
                obj.data.uv_layers.remove(obj.data.uv_layers[-1])

    # -------------------------------------------------------------------------
    # Shader + OSL network setup per object
    # -------------------------------------------------------------------------
    # Switch render engine to Cycles + OSL (same as before, just not per-camera)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.shading_system = True

    for obj in to_project:
        bpy.ops.object.select_all(action="DESELECT")
        context.view_layer.objects.active = obj
        obj.select_set(True)

        # Choose or create material
        if (
            is_refine
            and refine_preserve
            and not overwrite_material
        ):
            mat = obj.active_material.copy()
            obj.data.materials.append(mat)
            to_switch = True
        elif obj.active_material and (
            overwrite_material
            or (is_refine and refine_preserve)
            or (is_sequential and stop_index > 0)
            or regenerate_selected
        ):
            mat = obj.active_material
            to_switch = False
        else:
            mat = bpy.data.materials.new(name="ProjectionMaterial")
            obj.data.materials.append(mat)
            obj.active_material_index = obj.material_slots.find(mat.name)
            to_switch = True

        if to_switch:
            original_materials = obj.data.materials[:]
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            for original_mat in original_materials:
                if original_mat != mat:
                    obj.data.materials.append(original_mat)

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        previous_node = None
        output = None
        original_uv_map = obj.data.uv_layers[0]

        # ---------------------------------------------------------------------
        # Sequential regenerate path: remove compare nodes for a specific view
        # ---------------------------------------------------------------------
        if (is_sequential and stop_index > 0) or regenerate_selected:
            script_node = None

            # Find script node for this stop index/material id
            for node in nodes:
                if node.type == "SCRIPT" and node.label == f"{stop_index}-{mat_id}":
                    script_node = node
                    break

            compare_output_sockets = set()
            if script_node:
                # Remove compare nodes feeding from this script
                for link in list(script_node.outputs[0].links):
                    if (
                        link.to_node.type == "MATH"
                        and link.to_node.operation == "LESS_THAN"
                    ):
                        for link2 in list(link.to_node.outputs[0].links):
                            compare_output_sockets.add(link2.to_socket)
                        nodes.remove(link.to_node)

            # Reconnect script directly
            for to_socket in compare_output_sockets:
                links.new(script_node.outputs[0], to_socket)

            # Update TEX_IMAGE node for this view
            for node in nodes:
                if node.type == "TEX_IMAGE" and node.label == f"{stop_index}-{mat_id}":
                    if not bake_texture_flag:
                        image_path = get_file_path(
                            context,
                            "generated",
                            camera_id=stop_index,
                            material_id=mat_id,
                        )
                        if (
                            gen_method in ("sequential", "separate")
                            and scene.sequential_ipadapter
                            and scene.sequential_ipadapter_regenerate
                            and not scene.use_ipadapter
                            and stop_index == 0
                            and scene.sequential_ipadapter_mode == "first"
                        ):
                            image_path = image_path.replace(".png", "_ipadapter.png")
                    else:
                        image_path = get_file_path(
                            context,
                            "generated_baked",
                            camera_id=stop_index,
                            material_id=mat_id,
                            object_name=obj.name,
                        )

                    image = get_or_load_image(
                        image_path, force_reload=overwrite_material
                    )
                    if image:
                        node.image = image
                    break

            continue  # go to next object

        # ---------------------------------------------------------------------
        # Non-refine or no-preserve path: clear nodes
        # ---------------------------------------------------------------------
        if not (is_refine and refine_preserve):
            nodes.clear()
            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (3000, 0)
        else:
            # Find material output and its upstream node
            for node in nodes:
                if node.type == "OUTPUT_MATERIAL":
                    output = node
                    break
            if output and output.inputs[0].links:
                from_node = output.inputs[0].links[0].from_node
                if from_node.type == "BSDF_PRINCIPLED" and from_node.inputs[0].links:
                    previous_node = from_node.inputs[0].links[0].from_node
                else:
                    previous_node = from_node

            if output is None:
                output = nodes.new("ShaderNodeOutputMaterial")
                output.location = (3000, 0)

        # Geometry node
        geometry = nodes.new("ShaderNodeNewGeometry")
        geometry.location = (-600, 0)

        tex_image_nodes = []
        uv_map_nodes = []
        subtract_nodes = []
        normalize_nodes = []
        script_nodes = []
        script_nodes_outputs = []
        add_camera_loc_nodes = []
        length_nodes = []
        camera_fov_nodes = []
        camera_aspect_ratio_nodes = []
        camera_direction_nodes = []
        camera_up_nodes = []

        # ---------------------------------------------------------------------
        # Per-camera nodes
        # ---------------------------------------------------------------------
        for i, camera in enumerate(cameras):
            # Image node
            tex_image = nodes.new("ShaderNodeTexImage")
            tex_image.location = (0, -200 * i)
            tex_image.extension = "CLIP"
            tex_image.label = f"{i}-{mat_id}"

            if i <= stop_index:
                if not bake_texture_flag:
                    image_path = get_file_path(
                        context, "generated", camera_id=i, material_id=mat_id
                    )
                    if (
                        gen_method in ("sequential", "separate")
                        and scene.sequential_ipadapter
                        and scene.sequential_ipadapter_regenerate
                        and not scene.use_ipadapter
                        and i == 0
                        and scene.sequential_ipadapter_mode == "first"
                    ):
                        image_path = image_path.replace(".png", "_ipadapter.png")
                else:
                    image_path = get_file_path(
                        context,
                        "generated_baked",
                        camera_id=i,
                        material_id=mat_id,
                        object_name=obj.name,
                    )

                image = get_or_load_image(image_path, force_reload=overwrite_material)
                if image:
                    tex_image.image = image

            tex_image_nodes.append(tex_image)

            # UV map
            uv_map_node = nodes.new("ShaderNodeUVMap")
            uv_map_node.location = (-200, -200 * (i + 1))
            uv_map_node.uv_map = (
                f"ProjectionUV_{i}_{mat_id}"
                if not bake_texture_flag
                else original_uv_map.name
            )
            uv_map_nodes.append(uv_map_node)

            # Direction vector (from camera to position -> normalized)
            subtract = nodes.new("ShaderNodeVectorMath")
            subtract.operation = "SUBTRACT"
            subtract.location = (-400, -300 + (-800) * i)
            subtract.inputs[1].default_value = camera.location
            subtract_nodes.append(subtract)

            normalize = nodes.new("ShaderNodeVectorMath")
            normalize.operation = "NORMALIZE"
            normalize.location = (-400, -500 + (-800) * i)
            normalize_nodes.append(normalize)

            # Script node (OSL raycast)
            script = nodes.new("ShaderNodeScript")
            script.location = (-400, -800 * i)
            script.mode = "EXTERNAL"
            script.filepath = os.path.join(os.path.dirname(__file__), "raycast.osl")
            script.label = f"{i}-{mat_id}"

            # Angle / power from UI
            script.inputs["AngleThreshold"].default_value = scene.discard_factor
            script.inputs["Power"].default_value = scene.weight_exponent

            # Frustum feather controls from UI
            if scene.visibility_vignette:
                script.inputs["EdgeFeather"].default_value = scene.visibility_vignette_width
                script.inputs["EdgeGamma"].default_value = scene.visibility_vignette_softness
            else:
                script.inputs["EdgeFeather"].default_value = 0.0
                script.inputs["EdgeGamma"].default_value = 1.0

            script_nodes.append(script)

            if i > stop_index:
                less_than = nodes.new("ShaderNodeMath")
                less_than.operation = "LESS_THAN"
                less_than.location = (-200, -800 * i)
                less_than.inputs[1].default_value = -1
                links.new(script.outputs[0], less_than.inputs[0])
                script_nodes_outputs.append(less_than)
            else:
                script_nodes_outputs.append(script)

            # Camera FOV (corrected for vertical aspect when needed)
            camera_fov = nodes.new("ShaderNodeValue")
            camera_fov.location = (-600, 200 + 300 * i)
            fov = camera.data.angle_x
            if render.resolution_y > render.resolution_x:
                fov = 2 * atan(
                    tan(fov / 2) * render.resolution_x / render.resolution_y
                )
            camera_fov.outputs[0].default_value = fov
            camera_fov_nodes.append(camera_fov)

            # Camera aspect ratio
            camera_aspect_ratio = nodes.new("ShaderNodeValue")
            camera_aspect_ratio.location = (-600, 200 + 300 * i)
            camera_aspect_ratio.outputs[0].default_value = (
                render.resolution_x / render.resolution_y
            )
            camera_aspect_ratio_nodes.append(camera_aspect_ratio)

            # Camera direction (forward) and up vectors
            cam_quat = camera.matrix_world.to_quaternion()

            camera_direction = nodes.new("ShaderNodeCombineXYZ")
            camera_direction.location = (-600, 200 + 300 * i)
            forward = cam_quat @ Vector((0, 0, -1))
            camera_direction.inputs[0].default_value = forward.x
            camera_direction.inputs[1].default_value = forward.y
            camera_direction.inputs[2].default_value = forward.z
            camera_direction_nodes.append(camera_direction)

            camera_up = nodes.new("ShaderNodeCombineXYZ")
            camera_up.location = (-600, 200 + 300 * i)
            up_vec = cam_quat @ Vector((0, 1, 0))
            camera_up.inputs[0].default_value = up_vec.x
            camera_up.inputs[1].default_value = up_vec.y
            camera_up.inputs[2].default_value = up_vec.z
            camera_up_nodes.append(camera_up)

            # Camera origin (Combine XYZ)
            add_camera_loc = nodes.new("ShaderNodeCombineXYZ")
            add_camera_loc.location = (-600, 200 + 300 * i)
            add_camera_loc.inputs[0].default_value = camera.location.x
            add_camera_loc.inputs[1].default_value = camera.location.y
            add_camera_loc.inputs[2].default_value = camera.location.z
            add_camera_loc_nodes.append(add_camera_loc)

            # Distance (length of ray from camera to position)
            length = nodes.new("ShaderNodeVectorMath")
            length.operation = "LENGTH"
            length.location = (-400, 200 * (i + 1))
            length_nodes.append(length)

        # ---------------------------------------------------------------------
        # Mix textures together with weight tree
        # ---------------------------------------------------------------------
        mix_node, _ = build_mix_tree(
            tex_image_nodes,
            script_nodes_outputs,
            nodes,
            links,
            last_node=previous_node,
            stop_index=stop_index,
        )
        links.new(mix_node.outputs[0], output.inputs["Surface"])
        output.location = (mix_node.location[0] + 400, mix_node.location[1])

        # ---------------------------------------------------------------------
        # Wire up common inputs to each camera's nodes
        # ---------------------------------------------------------------------
        for i, _camera in enumerate(cameras):
            tex_image = tex_image_nodes[i]
            uv_map_node = uv_map_nodes[i]
            subtract = subtract_nodes[i]
            normalize = normalize_nodes[i]
            script = script_nodes[i]
            add_camera_loc = add_camera_loc_nodes[i]
            length = length_nodes[i]
            camera_fov = camera_fov_nodes[i]
            camera_aspect_ratio = camera_aspect_ratio_nodes[i]
            camera_direction = camera_direction_nodes[i]
            camera_up = camera_up_nodes[i]

            links.new(uv_map_node.outputs["UV"], tex_image.inputs["Vector"])
            links.new(geometry.outputs["Position"], subtract.inputs[0])
            links.new(subtract.outputs["Vector"], normalize.inputs[0])
            links.new(normalize.outputs["Vector"], script.inputs["Direction"])
            links.new(add_camera_loc.outputs["Vector"], script.inputs["Origin"])
            links.new(length.outputs["Value"], script.inputs["threshold"])
            links.new(geometry.outputs["Normal"], script.inputs["SurfaceNormal"])
            links.new(camera_fov.outputs[0], script.inputs["CameraFOV"])
            links.new(camera_aspect_ratio.outputs[0], script.inputs["CameraAspect"])
            links.new(camera_direction.outputs[0], script.inputs["CameraDir"])
            links.new(camera_up.outputs[0], script.inputs["CameraUp"])
            links.new(subtract.outputs["Vector"], length.inputs[0])

        # Material index node (kept as-is, though unused in your snippet)
        subtract_node = nodes.new("ShaderNodeMath")
        subtract_node.operation = "SUBTRACT"
        subtract_node.inputs[0].default_value = mat_id
        subtract_node.location = (-1000, 0)

    return True
    
def simple_project_bake(context, camera_id, obj, mat_id):
    scene = context.scene
    view_layer = context.view_layer

    # -------------------------------------------------------------------------
    # Create and assign temporary projection material
    # -------------------------------------------------------------------------
    mat = bpy.data.materials.new(name="ProjectionMaterialTemp")
    obj.data.materials.append(mat)

    # Make the new material active and assign it to all faces
    obj.active_material_index = len(obj.material_slots) - 1
    view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.material_slot_assign()
    bpy.ops.object.mode_set(mode='OBJECT')

    # -------------------------------------------------------------------------
    # Unwrap (only for the first camera)
    # -------------------------------------------------------------------------
    if camera_id == 0:
        unwrap(obj, scene.bake_unwrap_method, scene.bake_unwrap_overlap_only)

    # -------------------------------------------------------------------------
    # Node setup for temporary material
    # -------------------------------------------------------------------------
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Image texture
    tex_image = nodes.new("ShaderNodeTexImage")

    # Resolve file path (with sequential IPAdapter override if applicable)
    file_path = get_file_path(
        context,
        "generated",
        camera_id=camera_id,
        material_id=mat_id,
    )

    if (
        scene.generation_method in {'sequential', 'separate'}
        and scene.sequential_ipadapter
        and scene.sequential_ipadapter_regenerate
        and not scene.use_ipadapter
        and camera_id == 0
        and scene.sequential_ipadapter_mode == 'first'
    ):
        file_path = file_path.replace(".png", "_ipadapter.png")

    image = get_or_load_image(file_path, force_reload=scene.overwrite_material)
    if image:
        tex_image.image = image

    # UV map node
    uv_map_node = nodes.new("ShaderNodeUVMap")
    uv_map_node.uv_map = f"ProjectionUV_{camera_id}_{mat_id}"

    # BSDF + output
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputMaterial")

    # Wiring: UV → Texture → BSDF/Output
    links.new(uv_map_node.outputs["UV"], tex_image.inputs["Vector"])

    if scene.apply_bsdf:
        links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    else:
        links.new(tex_image.outputs["Color"], output.inputs["Surface"])

    # -------------------------------------------------------------------------
    # Baking
    # -------------------------------------------------------------------------
    texture_size = scene.bake_texture_size
    original_engine = scene.render.engine

    prepare_baking(context)

    # Ensure object has at least one UV map for baking
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")

    bake_texture(
        context,
        obj,
        texture_size,
        suffix=f"{camera_id}-{mat_id}",
        output_dir=get_dir_path(context, "generated_baked"),
    )

    # Restore original render engine
    scene.render.engine = original_engine

    # Remove the temporary material slot (the active one we just used)
    bpy.ops.object.material_slot_remove()

def get_or_load_image(filepath, force_reload=False):
    """
    Prevents duplicate image datablocks by default.
    If force_reload is True, it finds the existing datablock 
    and reloads it from the specified filepath.
    """
    if not filepath:
        print("Error: No filepath provided to get_or_load_image.")
        return None

    filename = os.path.basename(filepath)
    image = bpy.data.images.get(filename)
    
    # Verify if the found image actually points to the requested file
    if image:
        # Normalize paths for comparison (handle // prefix and OS separators)
        try:
            # bpy.path.abspath resolves // relative paths to absolute paths
            img_path = os.path.normpath(bpy.path.abspath(image.filepath))
            req_path = os.path.normpath(bpy.path.abspath(filepath))
            
            if img_path != req_path:
                # Name collision: found an image with the same name but different path.
                # This is NOT the image we want.
                image = None
                
                # Try to find if the correct image is already loaded under a different name
                for img in bpy.data.images:
                    if img.filepath:
                        try:
                            if os.path.normpath(bpy.path.abspath(img.filepath)) == req_path:
                                image = img
                                break
                        except:
                            continue
        except Exception as e:
            print(f"Warning: Error comparing image paths: {e}")
            image = None

    if image and force_reload:
        # Image exists, but we are forced to reload (overwrite).
        try:
            # IMPORTANT: Update the filepath property of the existing
            # datablock to the new file path.
            image.filepath = filepath
            
            # Reload the image data from that path.
            image.reload()
        except RuntimeError as e:
            # Reload can fail if the file isn't found, etc.
            print(f"Reload failed for {filename}. Removing old datablock. Error: {e}")
            # Remove the bad datablock so we can try loading it fresh.
            bpy.data.images.remove(image)
            image = None # Set to None to trigger the load block below

    if image is None:
        # Image does not exist in .data, or it failed to reload.
        try:
            image = bpy.data.images.load(filepath)
            # Only set the name if it's a new datablock to avoid renaming existing ones
            # Blender will handle naming collisions (e.g. .001) automatically
            if not bpy.data.images.get(filename):
                image.name = filename 
        except RuntimeError as e:
            # Load can fail if the file isn't found.
            print(f"Warning: Could not load image file: {filepath}. Error: {e}")
            return None
            
    return image


def reinstate_compare_nodes(context, to_project, stop_id_mat_id_pairs):
    """
    Reinstates the 'LESS_THAN' compare nodes that were removed for sequential generation.
    This will esentially revert given views to not-generated state for viewpoint regeneration.
    """

    for obj in to_project:
        if not obj.active_material:
            continue

        mat = obj.active_material
        if not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for stop_id, mat_id in stop_id_mat_id_pairs:
            script_node = None
            # Find the script node with the specific label
            for node in nodes:
                if node.type == 'SCRIPT' and node.label == f"{stop_id}-{mat_id}":
                    script_node = node
                    break
            
            if not script_node:
                continue

            # Store links to disconnect and reconnect later
            links_to_recreate = []
            for link in list(script_node.outputs[0].links):
                links_to_recreate.append((link.from_socket, link.to_socket))
                links.remove(link)

            # For each original connection, insert a 'LESS_THAN' node
            for from_socket, to_socket in links_to_recreate:
                # Create a new 'LESS_THAN' math node
                less_than_node = nodes.new(type='ShaderNodeMath')
                less_than_node.operation = 'LESS_THAN'
                less_than_node.inputs[1].default_value = -1
                # Position it between the script node and its original destination
                less_than_node.location = (script_node.location.x + 200, script_node.location.y)

                # Connect script_node -> less_than_node -> original destination
                links.new(from_socket, less_than_node.inputs[0])
                links.new(less_than_node.outputs[0], to_socket)
