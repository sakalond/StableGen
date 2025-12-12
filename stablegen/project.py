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
    def build_mix_tree(shaders, weight_nodes, nodes_collection, links, last_node=None, level=0, stop_index=1000000):
        """
        Recursively builds a binary tree of mix shader nodes.
        The mix factor between two shader nodes is computed dynamically using
        the outputs of weight_nodes assumed to come from OSL/maths nodes.
        
        For each pair, the total weight is created using an ADD node.
        Then a DIVIDE node computes:
            mix_factor = weight_A / (weight_A + weight_B)
        which is connected to the Mix Shader node's Fac input.
        
        Params:
        shaders: list of shader nodes (e.g., Principled BSDF nodes).
        weight_nodes: list of nodes whose outputs provide the dynamic weight values.
                        (e.g. the less_than nodes outputs)
        nodes_collection: the node tree's nodes collection.
        links: the node tree's links collection.
        Returns:
        A tuple (final_shader, final_weight_node).
        """
        # Compute offsets based on recursion level
        if context.scene.refine_preserve:
            x_offset = 1000 + level * 800 + last_node.location[0] + 200 if context.scene.early_priority else 1000 + level * 600 + last_node.location[0] + 200
        else:
            x_offset = 1000 + level * 800 if context.scene.early_priority else 1000 + level * 600
        y_offset = 0
        if len(shaders) == 1:
            # Base-case: Mix single color with fallback magenta
            final_mix = nodes_collection.new("ShaderNodeMixRGB")
            final_mix.location = (x_offset - 200, y_offset)
            # Convert fallback_color from 3 to 4 components by appending alpha 1.0
            final_mix.inputs["Color2"].default_value = (*context.scene.fallback_color, 1.0)
            # Connect the single color output to Color1
            links.new(shaders[0].outputs[0], final_mix.inputs["Color1"])
            if last_node:
                # Connect the final mix node to the last node
                links.new(last_node.outputs[0], final_mix.inputs["Color2"])
            # Create a compare node to drive the mix factor
            
            if context.scene.generation_method == 'refine' and context.scene.refine_preserve:
                compare_node = nodes_collection.new("ShaderNodeValToRGB")
                compare_node.location = (x_offset - 500, y_offset)
                compare_node.color_ramp.elements[0].position = 0.0
                compare_node.color_ramp.elements[0].color = (1, 1, 1, 1)
                compare_node.color_ramp.elements[1].position = 0.6
                compare_node.color_ramp.elements[1].color = (0, 0, 0, 1)
                compare_node.color_ramp.interpolation = 'LINEAR'
            else:
                compare_node = nodes_collection.new("ShaderNodeMath")
                compare_node.operation = 'COMPARE'
                # Value to compare against (Value 2)
                compare_node.inputs[1].default_value = 0.0
                # Epsilon for comparison
                compare_node.inputs[2].default_value = 0.0
                compare_node.location = (x_offset - 500, y_offset)
            # Connect final add node to compare node
            links.new(weight_nodes[0].outputs[0], compare_node.inputs[0])
            links.new(compare_node.outputs[0], final_mix.inputs["Fac"])
            
            if not context.scene.apply_bsdf:
                return final_mix, compare_node
                 
            # Add a principle shader for the mixed color
            should_add_principled = True
            if context.scene.generation_method == 'refine' and context.scene.refine_preserve:
                # Final principled is at last_node's output if it is BSDF
                final_principled = last_node.outputs[0].links[0].to_node
                if final_principled.type == 'BSDF_PRINCIPLED':
                    should_add_principled = False
            
            if should_add_principled:
                final_principled = nodes_collection.new("ShaderNodeBsdfPrincipled")
                final_principled.location = (x_offset, y_offset)
                final_principled.inputs["Roughness"].default_value = 1.0
                
            links.new(final_mix.outputs[0], final_principled.inputs[0])
            return final_principled, compare_node

        new_shaders = []
        new_weight_nodes = []
        i = 0
        while i < len(shaders):
            if i + 1 < len(shaders):
                # Sum the weights
                vert_offset = -200 * (i // 2)
                sum_node = nodes_collection.new("ShaderNodeMath")
                sum_node.operation = 'ADD'
                sum_node.location = (x_offset - 800, y_offset + vert_offset) if context.scene.early_priority else (x_offset - 600, y_offset + vert_offset)
                links.new(weight_nodes[i].outputs[0], sum_node.inputs[0])
                links.new(weight_nodes[i+1].outputs[0], sum_node.inputs[1])

                # Compute mix factor: weight_A / (weight_A+weight_B)
                div_node = nodes_collection.new("ShaderNodeMath")
                div_node.operation = 'DIVIDE'
                div_node.location = (x_offset - 600, y_offset + vert_offset) if context.scene.early_priority else (x_offset - 400, y_offset + vert_offset)
                links.new(weight_nodes[i+1].outputs[0], div_node.inputs[0])
                links.new(sum_node.outputs[0], div_node.inputs[1])

                if context.scene.early_priority:
                    # Add map range node
                    map_range_node = nodes_collection.new("ShaderNodeMapRange")
                    map_range_node.location = (x_offset - 400, y_offset + vert_offset)
                    # to_min, to_max, from_max are set to 0.0 and 1.0 by default
                    # set to prioritize earlier images
                    map_range_node.inputs[1].default_value = context.scene.early_priority_strength
                    links.new(div_node.outputs[0], map_range_node.inputs[0])

                # Create a MixRGB node for color blending of the two inputs
                mix_node = nodes_collection.new("ShaderNodeMixRGB")
                mix_node.location = (x_offset - 200, y_offset + vert_offset)
                mix_node.use_clamp = True
                # Connect first color to Color1 and second color to Color2
                links.new(shaders[i].outputs[0], mix_node.inputs["Color1"])
                links.new(shaders[i+1].outputs[0], mix_node.inputs["Color2"])
                # Use the computed mix factor
                to_connect = div_node if not context.scene.early_priority else map_range_node
                links.new(to_connect.outputs[0], mix_node.inputs["Fac"])

                new_shaders.append(mix_node)
                # The new effective weight is the sum (stored in sum_node)
                new_weight_nodes.append(sum_node)
                i += 2
            else:
                new_shaders.append(shaders[i])
                new_weight_nodes.append(weight_nodes[i])
                i += 1
        return build_mix_tree(new_shaders, new_weight_nodes, nodes_collection, links, last_node, level+1)

    cameras = [obj for obj in context.scene.objects if obj.type == 'CAMERA']
    cameras.sort(key=lambda x: x.name)
    
    # Force refresh of the UI
    for area in context.screen.areas:
        area.tag_redraw()
    
    # Apply projection to all mesh objects for each camera
    for i, camera in enumerate(cameras):
        for x, obj in enumerate(to_project):
            # We can skip the UV Projection step in sequential mode for i > 0
            if context.scene.generation_method != 'sequential' or stop_index == 0 or context.scene.bake_texture:
                # Deselect all objects
                bpy.ops.object.select_all(action='DESELECT')
                # Select object as active (needed for applying the modifier)
                context.view_layer.objects.active = obj
                obj.select_set(True)
                    
                # Make object data single-user before applying modifier
                bpy.ops.object.make_single_user(object=True, obdata=True)

                # Check if the data is now single user
                if obj.data.users > 1:
                    # If not, we need to make it single user again
                    print("Warning: Cannot make object data single user. Making a copy.")
                    obj.data = obj.data.copy()
                    obj.data.name = f"{obj.name}_data"
                    obj.data.update()
                    # Check again
                    if obj.data.users > 1:
                        print("Error: Cannot make object data single user. Exiting.")
                        return Exception("Cannot make object data single user. Exiting.")

                # Add new UV map if not present (with same name)
                # Check if "ProjectionUV" UV map already exists
                uv_map = None

                if context.scene.overwrite_material and not context.scene.bake_texture:
                    for uv in obj.data.uv_layers:
                        if uv.name == f"ProjectionUV_{i}_{mat_id}":
                            uv_map = uv
                            break
                        
                # If the objects has no UV map and we are baking textures, create a new one
                if not obj.data.uv_layers and context.scene.bake_texture:
                    obj.data.uv_layers.new(name="BakeUV")

                if not uv_map:
                    uv_map = obj.data.uv_layers.new(name=f"ProjectionUV_{i}_{mat_id}")

                # Add the UV Project Modifier if not present
                uv_project_mod = obj.modifiers.new(name="UVProject", type='UV_PROJECT')

                # Assign the active camera to the UV Project modifier
                if not camera:
                    return False

                uv_project_mod.projectors[0].object = camera

                # Set the UV map for the modifier
                try:
                    uv_project_mod.uv_layer = uv_map.name
                except:
                    # Throw custom exception: Not enough UV map slots
                    raise Exception("Not enough UV map slots. Please remove some UV maps.")

                # Calculate and set the aspect ratio
                render = context.scene.render
                aspect_ratio = render.resolution_x / render.resolution_y
                uv_project_mod.aspect_x = aspect_ratio if aspect_ratio > 1 else 1
                uv_project_mod.aspect_y = 1 / aspect_ratio if aspect_ratio < 1 else 1

                # Apply the modifier
                bpy.ops.object.modifier_apply(modifier=uv_project_mod.name)

                original_uv_map = obj.data.uv_layers[0]

                # If we are running in sequential mode, we already have baked textures for i < stop_index
            if context.scene.bake_texture:
                if (stop_index > 0 and context.scene.generation_method == 'sequential'):
                    # Deselect all objects
                    bpy.ops.object.select_all(action='DESELECT')
                    # Select object as active (needed for applying the modifier)
                    context.view_layer.objects.active = obj
                    obj.select_set(True)
                if i <= stop_index and (not context.scene.generation_method == 'sequential' or i == stop_index):
                    simple_project_bake(context, i, obj, mat_id)
                obj.data.uv_layers.remove(obj.data.uv_layers[-1]) # Remove the last UV map

    for obj in to_project:

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = obj
        obj.select_set(True)

        # Create the material
        to_switch = False
        if context.scene.generation_method == "refine" and context.scene.refine_preserve and not context.scene.overwrite_material:
            # Copy active material
            mat = obj.active_material.copy()
            obj.data.materials.append(mat)
            to_switch = True
        elif obj.active_material and (context.scene.overwrite_material or (context.scene.generation_method == "refine" and context.scene.refine_preserve) \
                                      or (context.scene.generation_method == 'sequential' and stop_index > 0) or context.scene.generation_mode == 'regenerate_selected'):
            # Use active material
            mat = obj.active_material
        else:
            mat = bpy.data.materials.new(name="ProjectionMaterial")
            obj.data.materials.append(mat)
            # Mark as active material
            obj.active_material_index = obj.material_slots.find(mat.name)
            to_switch = True
        
        if to_switch:
            original_materials = obj.data.materials[:]
            # Clear existing materials
            obj.data.materials.clear()
            # Add the new material
            obj.data.materials.append(mat)
            # Add the rest of the original materials
            for original_mat in original_materials:
                if original_mat != mat:
                    obj.data.materials.append(original_mat)

        # Enable use of nodes
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        previous_node = None
        output = None

        original_uv_map = obj.data.uv_layers[0]


        if (context.scene.generation_method == 'sequential' and stop_index > 0) or context.scene.generation_mode == 'regenerate_selected':
            # We just need to remove the compare nodes which are connected to script node at stop_index
            script_node = None
            # First find all script nodes with label {stop_index}-{mat_id}
            for node in nodes:
                if node.type == 'SCRIPT' and node.label == f"{stop_index}-{mat_id}":
                    script_node = node
                    break
            compare_output_sockets = set()
            if script_node:
                # Find all compare nodes connected to the script node
                for link in script_node.outputs[0].links:
                    if link.to_node.type == 'MATH' and link.to_node.operation == 'LESS_THAN':
                        # Save all outputs (nodes connected to) of compare nodes
                        for link2 in link.to_node.outputs[0].links:
                            compare_output_sockets.add(link2.to_socket)
                        # Remove the compare node
                        nodes.remove(link.to_node)
            # Connect the script node to all outputs
            for output in compare_output_sockets:
                links.new(script_node.outputs[0], output)
            # We also need to set the generated image to the texture node with label {stop_index}-{mat_id}
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.label == f"{stop_index}-{mat_id}":
                    if not context.scene.bake_texture:
                        image_path = get_file_path(context, "generated", camera_id=stop_index, material_id=mat_id)
                        if (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate') and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate \
                        and not context.scene.use_ipadapter and stop_index == 0 and context.scene.sequential_ipadapter_mode == 'first':
                            image_path = get_file_path(context, "generated", camera_id=stop_index, material_id=mat_id).replace(".png", "_ipadapter.png")
                    else:
                        # Use baked texture
                        image_path = get_file_path(context, "generated_baked", camera_id=stop_index, material_id=mat_id, object_name=obj.name)

                    image = get_or_load_image(image_path, force_reload=context.scene.overwrite_material)
                    if image:
                        node.image = image
                    break
            # Now we can continue to the next object
            continue
                        
        elif not context.scene.refine_preserve or not context.scene.generation_method == 'refine':
            # Clear existing nodes
            for node in nodes:
                nodes.remove(node)
        else:
            # Find the node connected to the output node
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    output = node
                    break
            if output:
                # Save the node connected to the output node into a variable
                if output.inputs[0].links[0].from_node.type == 'BSDF_PRINCIPLED':
                    previous_node = output.inputs[0].links[0].from_node.inputs[0].links[0].from_node
                else:
                    previous_node = output.inputs[0].links[0].from_node
            
        if not (context.scene.generation_method == 'refine' and context.scene.refine_preserve):
            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (3000, 0)
        
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

        for i, camera in enumerate(cameras):
            # Add image texture node
            tex_image = nodes.new("ShaderNodeTexImage")
            if i <= stop_index:
                if not context.scene.bake_texture:
                    image_path = get_file_path(context, "generated", camera_id=i, material_id=mat_id)
                    if (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate') and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate \
                    and not context.scene.use_ipadapter and i == 0 and context.scene.sequential_ipadapter_mode == 'first':
                        image_path = get_file_path(context, "generated", camera_id=i, material_id=mat_id).replace(".png", "_ipadapter.png")
                else:
                    # Use baked texture
                    image_path = get_file_path(context, "generated_baked", camera_id=i, material_id=mat_id, object_name=obj.name)

                image = get_or_load_image(image_path, force_reload=context.scene.overwrite_material)
                if image:
                    tex_image.image = image

            tex_image.location = (0, -200 * i)
            tex_image.extension = 'CLIP'
            tex_image.label = f"{i}-{mat_id}"
            tex_image_nodes.append(tex_image)
                
            # Add UV map node
            uv_map_node = nodes.new("ShaderNodeUVMap")
            if not context.scene.bake_texture:
                uv_map_node.uv_map = f"ProjectionUV_{i}_{mat_id}"
            else:
                # Use the original UV map
                uv_map_node.uv_map = original_uv_map.name
            uv_map_node.location = (-200, -200 * (i+1))
            uv_map_nodes.append(uv_map_node)
            
            # Compute the dot product of the direction vector and geometry normal

            subtract = nodes.new("ShaderNodeVectorMath")
            subtract.operation = 'SUBTRACT'
            subtract.location = (-400, -300 + (-800) * i)
            subtract.inputs[1].default_value = camera.location
            subtract_nodes.append(subtract)

            normalize = nodes.new("ShaderNodeVectorMath")
            normalize.operation = 'NORMALIZE'
            normalize.location = (-400, -500 + (-800) * (i))
            normalize_nodes.append(normalize)

        
            # Switch to Cycles for OSL support
            context.scene.render.engine = 'CYCLES'
            context.scene.cycles.device = 'CPU'
            context.scene.cycles.shading_system = True

            # Add a script node
            script = nodes.new("ShaderNodeScript")
            script.location = (-400, (-800) * i)
            script.mode = 'EXTERNAL'
            script.filepath = os.path.join(os.path.dirname(__file__), "raycast.osl")
            script.inputs["AngleThreshold"].default_value = context.scene.discard_factor
            script.inputs["Power"].default_value = context.scene.weight_exponent
            script.label = f"{i}-{mat_id}"
            script_nodes.append(script)
            if i > stop_index:
                # Connect a temporary less than node to the script node
                less_than = nodes.new("ShaderNodeMath")
                less_than.operation = 'LESS_THAN'
                less_than.location = (-200, (-800) * i)
                less_than.inputs[1].default_value = -1
                links.new(script.outputs[0], less_than.inputs[0])
                script_nodes_outputs.append(less_than)
            else:
                script_nodes_outputs.append(script)

            # Add additional add node, which will contain camera's FOV in first default value, and camera's aspect ratio in second default value
            camera_fov = nodes.new("ShaderNodeValue")
            camera_fov.location = (-600, 200 + 300 * i)
            fov = camera.data.angle_x
            # Correct the FOV for vertical aspect ratio
            if context.scene.render.resolution_y > context.scene.render.resolution_x:
                fov = 2 * atan(tan(fov / 2) * context.scene.render.resolution_x / context.scene.render.resolution_y)
            camera_fov.outputs[0].default_value = fov
            camera_fov_nodes.append(camera_fov)
            camera_aspect_ratio = nodes.new("ShaderNodeValue")
            camera_aspect_ratio.location = (-600, 200 + 300 * i)
            
            # Add camera direction and up nodes (combine XYZ)
            camera_direction = nodes.new("ShaderNodeCombineXYZ")
            camera_direction.location = (-600, 200 + 300 * i)
            # Get the camera direction vector
            camera_direction.inputs[0].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))).x
            camera_direction.inputs[1].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))).y
            camera_direction.inputs[2].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))).z
            camera_direction_nodes.append(camera_direction)
            camera_up = nodes.new("ShaderNodeCombineXYZ")
            camera_up.location = (-600, 200 + 300 * i)
            # Get the camera up vector
            camera_up.inputs[0].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))).x
            camera_up.inputs[1].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))).y
            camera_up.inputs[2].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))).z
            camera_up_nodes.append(camera_up)
            
            camera_aspect_ratio.outputs[0].default_value = context.scene.render.resolution_x / context.scene.render.resolution_y
            camera_aspect_ratio_nodes.append(camera_aspect_ratio)

            # Add combine XYZ node
            add_camera_loc = nodes.new("ShaderNodeCombineXYZ")
            add_camera_loc.location = (-600, 200 + 300 * i)
            add_camera_loc.inputs[0].default_value = camera.location.x
            add_camera_loc.inputs[1].default_value = camera.location.y
            add_camera_loc.inputs[2].default_value = camera.location.z
            add_camera_loc_nodes.append(add_camera_loc)

            # Add length node
            length = nodes.new("ShaderNodeVectorMath")
            length.operation = 'LENGTH'
            length.location = (-400, 200 * (i+1))
            length_nodes.append(length)

        # Build mix shader tree
        mix_node, _ = build_mix_tree(tex_image_nodes, script_nodes_outputs, nodes, links, previous_node, stop_index=stop_index)
        links.new(mix_node.outputs[0], output.inputs["Surface"])

        # Move output node right to the mix_node
        output.location = (mix_node.location[0] + 400, mix_node.location[1])

        for i, camera in enumerate(cameras):
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

            # Connect common nodes
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

        # Add material index node (subtract node)
        subtract_node = nodes.new("ShaderNodeMath")
        subtract_node.operation = 'SUBTRACT'
        subtract_node.inputs[0].default_value = mat_id
        subtract_node.location = (-1000, 0)
    return True
    
def simple_project_bake(context, camera_id, obj, mat_id):
    # Create a temporary material for the projection
    mat = bpy.data.materials.new(name="ProjectionMaterialTemp")
    obj.data.materials.append(mat)

    # Switch the active material to the new material (Switch to edit mode, select all, assign the material)
    obj.active_material_index = len(obj.material_slots) - 1
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.material_slot_assign()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Unwrap (only for the first camera)
    if camera_id == 0:
        unwrap(obj, context.scene.bake_unwrap_method, context.scene.bake_unwrap_overlap_only)

    # Enable use of nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Add image texture node
    tex_image = nodes.new("ShaderNodeTexImage")
    
    file_path = get_file_path(context, "generated", camera_id=camera_id, material_id=mat_id)
    if (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate') and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate \
        and not context.scene.use_ipadapter and camera_id == 0 and context.scene.sequential_ipadapter_mode == 'first':
        file_path = get_file_path(context, "generated", camera_id=camera_id, material_id=mat_id).replace(".png", "_ipadapter.png")
    
    image = get_or_load_image(file_path, force_reload=context.scene.overwrite_material)
    if image:
        tex_image.image = image

    # Add UV map node
    uv_map_node = nodes.new("ShaderNodeUVMap")
    uv_map_node.uv_map = f"ProjectionUV_{camera_id}_{mat_id}"
    
    # Add BSDF node
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 1.0

    # Add output node
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(uv_map_node.outputs["UV"], tex_image.inputs["Vector"])
    # Connect the nodes
    if context.scene.apply_bsdf:
        links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    else:
        links.new(tex_image.outputs["Color"], output.inputs["Surface"])
 
    # Bake texture using BakeTextures.bake_texture
    texture_size = context.scene.bake_texture_size
    original_engine = context.scene.render.engine
    prepare_baking(context)
    
    # If the object has no UV map, create one
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    
    bake_texture(context, obj, texture_size, suffix=f"{camera_id}-{mat_id}", output_dir=get_dir_path(context, "generated_baked"))
    context.scene.render.engine = original_engine

    # Remove the temporary material
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
