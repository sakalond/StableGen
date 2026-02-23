import bpy
import os
from datetime import datetime

class AddHDRI(bpy.types.Operator):
	"""Add HDRI Global Illumination to the scene."""
	bl_idname = "object.add_hdri"
	bl_label = "Add HDRI Global Illumination"
	bl_options = {'REGISTER', 'UNDO'}

	hdri_path: bpy.props.StringProperty(
		name="HDRI Path",
		description="Path to the HDRI image",
		default="",
		subtype='FILE_PATH'
	) # type: ignore

	strength: bpy.props.FloatProperty(
		name="Strength",
		description="Brightness of the HDRI environment lighting",
		default=1.0,
		min=0.01,
		max=10.0
	) # type: ignore

	def execute(self, context):
		world = context.scene.world
		if not world:
			world = bpy.data.worlds.new("World")
			context.scene.world = world
		world.use_nodes = True
		tree = world.node_tree
		nodes = tree.nodes
		links = tree.links

		# Clear existing nodes
		nodes.clear()

		# Create Environment Texture node
		env_node = nodes.new("ShaderNodeTexEnvironment")
		env_node.name = "Environment Texture"
		env_node.location = (-300, 200)
		try:
			env_node.image = bpy.data.images.load(self.hdri_path)
		except Exception as e:
			self.report({'ERROR'}, f"Failed to load HDRI: {e}")
			return {'CANCELLED'}

		# Create Light Path node to separate background from lighting
		light_path = nodes.new('ShaderNodeLightPath')
		light_path.location = (-300, -100)

		# Create Mix Shader node to mix background color with HDRI lighting
		mix_shader = nodes.new('ShaderNodeMixShader')
		mix_shader.location = (200, 0)

		# Create Background nodes - one for HDRI lighting and one for a solid background
		bg_hdri = nodes.new("ShaderNodeBackground")
		bg_hdri.name = "HDRI Background"
		bg_hdri.location = (0, 100)
		bg_hdri.inputs['Strength'].default_value = self.strength
		
		bg_solid = nodes.new("ShaderNodeBackground")
		bg_solid.name = "Solid Background"
		bg_solid.location = (0, -100)
		bg_solid.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1)  # Default gray

		# Create Output node
		output_node = nodes.new("ShaderNodeOutputWorld")
		output_node.location = (400, 0)

		# Connect nodes
		links.new(env_node.outputs['Color'], bg_hdri.inputs['Color'])
		links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
		links.new(bg_hdri.outputs['Background'], mix_shader.inputs[1])
		links.new(bg_solid.outputs['Background'], mix_shader.inputs[2])
		links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])

		self.report({'INFO'}, f"HDRI lighting added (strength={self.strength:.2f}, background hidden).")
		return {'FINISHED'}

	def invoke(self, context, event):
		wm = context.window_manager
		return wm.invoke_props_dialog(self)

class ApplyModifiers(bpy.types.Operator):
    """Applies every modifier on every mesh in the scene and converts all instanced geometry into real meshes.

    Some modifiers (and instanced geometry) can interfere with automated texture generation
    or baking workflows. Running this operator ensures that:
    
     - All modifier stacks on every mesh object are applied in turn.
     - Any instanced or duplicated geometry (particle instances, dupliverts, collection instances,
        geometry‑nodes instances, etc.) is converted into real, editable mesh data.

    Use this as a preprocessing step to guarantee clean, final meshes before generating or baking textures."""
    bl_idname = "object.apply_all_mesh_modifiers"
    bl_label = "Apply All Modifiers on All Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Operator can only run in Object Mode
        return context.mode == 'OBJECT'

    def execute(self, context):
        if context.mode != 'OBJECT':
            self.report({'WARNING'}, "Operator requires Object Mode")
            return {'CANCELLED'}

        scene = context.scene
        view_layer = context.view_layer

        # Store original state
        original_active_object = view_layer.objects.active
        original_selected_objects = context.selected_objects[:] # Make a copy

        applied_count = 0
        object_count = 0

        # Deselect all initially to avoid issues with operator context
        bpy.ops.object.select_all(action='DESELECT')

        # Iterate through all objects in the scene
        for obj in scene.objects:
            # Check if the object is a mesh and has modifiers
            if (obj.type == 'MESH') and obj.modifiers:
                object_count += 1
                self.report({'INFO'}, f"Processing object: {obj.name}")

                # Set the object as the active object in the view layer
                view_layer.objects.active = obj
                obj.select_set(True)
                
                # First, make any instances real on this object
                # This should correspond to the "Make Instances Real" button
                try:
                    bpy.ops.object.duplicates_make_real(use_base_parent=True,
                                                        use_hierarchy=True)
                    self.report({'INFO'}, f"  Made instances real on {obj.name}")
                except RuntimeError:
                    # if there were no instances, or operator failed, ignore
                    pass

                # Loop through modifiers and apply them
                num_modifiers = len(obj.modifiers)
                applied_modifiers_on_obj = 0

                # Use a while loop as applying modifies the collection
                while obj.modifiers:
                    modifier = obj.modifiers[0] # Always target the first one
                    modifier_name = modifier.name
                    try:
                        bpy.ops.object.modifier_apply(modifier=modifier_name)
                        self.report({'INFO'}, f"  Applied modifier '{modifier_name}' on {obj.name}")
                        applied_modifiers_on_obj += 1
                        applied_count += 1
                    except RuntimeError as e:
                        self.report({'ERROR'}, f"Failed to apply modifier '{modifier_name}' on {obj.name}: {e}")
                        # If applying fails, break the loop for this object
                        break

                # Deselect the object after processing
                obj.select_set(False)

        # Restore original selection and active object
        view_layer.objects.active = original_active_object
        for sel_obj in original_selected_objects:
            # Check if the originally selected object still exists
            if sel_obj.name in scene.objects:
                sel_obj.select_set(True)
            else:
                 self.report({'WARNING'}, f"Originally selected object '{sel_obj.name}' no longer exists.")


        self.report({'INFO'}, f"Finished applying {applied_count} modifiers on {object_count} mesh objects.")
        return {'FINISHED'}
    
class CurvesToMesh(bpy.types.Operator):
	"""Convert all curve objects in the scene into meshes.
 
 - Mesh geometry is required for texturing and baking operations."""
	bl_idname = "object.curves_to_mesh"
	bl_label = "Convert Curves to Mesh"
	bl_options = {'REGISTER', 'UNDO'}

	def execute(self, context):
		# Select all objects
		bpy.ops.object.select_all(action='SELECT')
		for obj in context.scene.objects:
			if obj.type == 'CURVE':
				bpy.context.view_layer.objects.active = obj
				bpy.ops.object.convert(target='MESH')
		# Deselect all objects
		bpy.ops.object.select_all(action='DESELECT')
		self.report({'INFO'}, "All curves converted to meshes.")
		return {'FINISHED'}
	
def get_last_material_index(obj):
	"""     
	Get the index of the last material of the object. 
	The index is hidden inside default value of (the only) subtract node.
	If there are no subtract nodes, return -1. 
	:param obj: Blender object.     
	:return: Index of the last material.     
	"""
 
	highest_index = -1
 
	if obj.data.materials:
		for mat in obj.data.materials:
			if mat and mat.use_nodes:
				for node in mat.node_tree.nodes:
					if node.type == 'MATH' and node.operation == 'SUBTRACT':
						if node.inputs[0].default_value > highest_index:
							highest_index = node.inputs[0].default_value

	return int(highest_index)

def get_generation_dirs(context):
	"""
	Gets the directory structure for the current generation session.
	Creates a dictionary with paths to all required subdirectories for output files.
	:param context: Blender context containing addon preferences and scene data.
	:return: Dictionary with paths to all subdirectories (revision, controlnet, generated, baked, inpaint).
	"""
	base_dir = context.preferences.addons[__package__].preferences.output_dir
	scene_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
	# Use scene name as fallback if the blend file hasn't been saved yet
	if not scene_name:
		scene_name = context.scene.name
	timestamp = context.scene.output_timestamp
	
	if not timestamp:
		# If no timestamp set, use current time
		timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
		context.scene.output_timestamp = timestamp
	
	# Define directory structure
	revision_dir = os.path.join(base_dir, scene_name, timestamp)
 
	return {
		"revision": revision_dir,
        "controlnet_root": os.path.join(revision_dir, "controlnet"),
		"controlnet": {
			"depth": os.path.join(revision_dir, "controlnet", "depth"),
			"canny": os.path.join(revision_dir, "controlnet", "canny"),
			"normal": os.path.join(revision_dir, "controlnet", "normal"),
			"workbench": os.path.join(revision_dir, "controlnet", "workbench"),
			"viewport": os.path.join(revision_dir, "controlnet", "viewport"),
		},
		"generated": os.path.join(revision_dir, "generated"),
		"generated_baked": os.path.join(revision_dir, "generated_baked"),
		"baked": os.path.join(revision_dir, "baked"),
		"inpaint": {
			"render": os.path.join(revision_dir, "inpaint", "render"),
			"visibility": os.path.join(revision_dir, "inpaint", "visibility"),
		},
  		"inpaint_root": os.path.join(revision_dir, "inpaint"),
		"uv_inpaint": {
			"visibility": os.path.join(revision_dir, "uv_inpaint", "uv_visibility"),
		},
  		"uv_inpaint_root": os.path.join(revision_dir, "uv_inpaint"),
		"misc" : os.path.join(revision_dir, "misc"),
		"pbr": os.path.join(revision_dir, "pbr"),
	}

def ensure_dirs_exist(dirs_dict):
	"""
	Ensure that all required directories exist.
	:param dirs_dict: Dictionary of directory paths.
	:return: None
	"""
	
	# Create the main directories
	os.makedirs(dirs_dict["revision"], exist_ok=True)
	os.makedirs(dirs_dict["generated"], exist_ok=True)
	os.makedirs(dirs_dict["generated_baked"], exist_ok=True)
	os.makedirs(dirs_dict["baked"], exist_ok=True)

	# Create controlnet subdirectories
	for key, path in dirs_dict["controlnet"].items():
		os.makedirs(path, exist_ok=True)
	
	# Create inpaint subdirectories	
	for key, path in dirs_dict["inpaint"].items():
		os.makedirs(path, exist_ok=True)
  
	# Create uv_inpaint subdirectories
	for key, path in dirs_dict["uv_inpaint"].items():
		os.makedirs(path, exist_ok=True)
	# Create misc directory
	os.makedirs(dirs_dict["misc"], exist_ok=True)
	# Create pbr directory
	os.makedirs(dirs_dict["pbr"], exist_ok=True)

def get_file_path(context, file_type, subtype=None, filename=None, camera_id=None, object_name=None, material_id=None, legacy=False):
	"""
	Generate the full file path for saving images based on the type of file and other parameters.
	:param context: Blender context
	:param file_type: The type of file (controlnet, generated, baked, inpaint)
	:param subtype: Subtype for controlnet or inpaint (depth, canny, normal, render, etc.)
	:param filename: Base filename without extension
	:param camera_id: Optional camera ID for camera-specific files
	:param object_name: Optional object name for object-specific files
	:param material_id: Optional material ID for material-specific files
	:param legacy: If True, always include frame suffix (0001) regardless of Blender version.
	               Used for filenames written by CompositorNodeOutputFile which appends
	               0001 on Blender 4.x but not on 5.x+.
	:return: The full file path
	"""
	dirs = get_generation_dirs(context)

	# Determine frame suffix: Blender 5.x compositor output nodes no longer append "0001"
	frame_suffix = "0001" if legacy or bpy.app.version < (5, 0, 0) else ""
 
	# Ensure the directories exist
	ensure_dirs_exist(dirs)
	
	if file_type == "controlnet" and subtype:
		base_dir = dirs["controlnet"][subtype]
		if not filename:
			if subtype == "depth":
				filename = f"depth_map{camera_id}{frame_suffix}" if camera_id is not None else "depth_map_grid"
			elif subtype == "canny":  
				filename = f"canny{camera_id}{frame_suffix}" if camera_id is not None else "canny_grid"
			elif subtype == "normal":
				filename = f"normal_map{camera_id}{frame_suffix}" if camera_id is not None else "normal_grid"
			elif subtype == "workbench":
				filename = f"render{camera_id}{frame_suffix}" if camera_id is not None else "render_grid"
			elif subtype == "viewport":
				filename = f"viewport{camera_id}" if camera_id is not None else "viewport_grid"
		return os.path.join(base_dir, f"{filename}.png")
	
	elif file_type == "generated":
		base_dir = dirs["generated"]
		material_suffix = f"-{material_id}" if material_id is not None else ""
		return os.path.join(base_dir, f"generated_image{camera_id}{material_suffix}-0001.png" if camera_id is not None else "generated_image.png")
	
	elif file_type == "generated_baked":
		base_dir = dirs["generated_baked"]
		if object_name:
			material_suffix = f"{camera_id}-{material_id}" if material_id is not None else ""
			return os.path.join(base_dir, f"{object_name}_baked{material_suffix}.png")
	
	elif file_type == "baked":
		base_dir = dirs["baked"]
		if not filename:
			filename = f"{object_name}" if object_name else "baked_texture"
		return os.path.join(base_dir, f"{filename}.png")
	
	elif file_type == "inpaint" and subtype:
		base_dir = dirs["inpaint"][subtype]
		if subtype == "render":
			filename = f"ctx_render{camera_id}{frame_suffix}" if not filename else filename
		elif subtype == "visibility":
			filename = f"ctx_render{camera_id}_visibility{frame_suffix}" if not filename else filename
		return os.path.join(base_dir, f"{filename}.png")

	elif file_type == "uv_inpaint" and subtype:
		base_dir = dirs["uv_inpaint"][subtype]
		if subtype == "visibility":
			filename = f"{object_name}_baked_visibility" if not filename else filename
		return os.path.join(base_dir, f"{filename}.png")

	elif file_type == "pbr" and subtype:
		# PBR decomposition maps: subtype is the map name (albedo, roughness, metallic, etc.)
		base_dir = dirs["pbr"]
		material_suffix = f"-{material_id}" if material_id is not None else ""
		if camera_id is not None:
			return os.path.join(base_dir, f"pbr_{subtype}_cam{camera_id}{material_suffix}.png")
		else:
			return os.path.join(base_dir, f"pbr_{subtype}{material_suffix}.png")

	# Fallback to revision directory
	return os.path.join(dirs["revision"], f"{filename or 'file'}.png")

def get_dir_path(context, file_type):
	"""
	Get the directory path for a specific file type.
	:param context: Blender context
	:param file_type: The type of file (controlnet, generated, baked, inpaint)
	:return: The directory path
	"""
	dirs = get_generation_dirs(context)
 
	# Ensure the directories exist
	ensure_dirs_exist(dirs)
	
	if file_type == "revision":
		return dirs["revision"]
	elif file_type == "controlnet":
		return dirs["controlnet"]
	elif file_type == "generated":
		return dirs["generated"]
	elif file_type == "generated_baked":
		return dirs["generated_baked"]
	elif file_type == "baked":
		return dirs["baked"]
	elif file_type == "inpaint":
		return dirs["inpaint"]
	elif file_type == "uv_inpaint":
		return dirs["uv_inpaint"]
	else:
		return dirs["misc"]
	
	return None

def remove_empty_dirs(context, dirs_obj = None):
	"""
	Remove empty directories from the generation structure.
	:param context: Blender context
	"""
	if dirs_obj is None:
		dirs_obj = get_generation_dirs(context)
	
	for key, value in dirs_obj.items():
			if isinstance(value, dict):
				remove_empty_dirs(context, dirs_obj=value)
			else:
				if os.path.exists(value) and not os.listdir(value):
					os.rmdir(value)


def get_compositor_node_tree(scene):
	"""
	Get the compositor node tree, handling API differences between Blender versions.
	Blender 5.0+ uses scene.compositing_node_group, older versions use scene.node_tree.
	In 5.0+ the group may be None on first use, so we create it.
	"""
	if hasattr(scene, 'compositing_node_group'):
		if scene.compositing_node_group is None:
			new_tree = bpy.data.node_groups.new(name="Compositing", type='CompositorNodeTree')
			scene.compositing_node_group = new_tree
		return scene.compositing_node_group
	if hasattr(scene, 'node_tree'):
		return scene.node_tree
	return None


def configure_output_node_paths(node, directory, filename):
	"""Configure output node paths using Blender 5.0+ image-mode semantics with 4.x fallback."""

	# 1. Force single-image mode so format enums unlock in Blender 5
	if hasattr(node.format, "media_type"):
		node.format.media_type = 'IMAGE'

	# 2. Set PNG format
	node.format.file_format = "PNG"
	node.format.color_depth = '8'  # Force 8-bit; Blender 5.x defaults to 16-bit

	# 3. Configure directory/base path
	if hasattr(node, "directory"):
		node.directory = directory
	else:
		node.base_path = directory

	# 4. Clear prefix
	if hasattr(node, "file_name"):
		node.file_name = ""

	# 5. Update the visible slot label/path for both APIs
	slot = None
	if hasattr(node, "file_output_items"):
		items = node.file_output_items
		if items and len(items) > 0:
			slot = items[0]
			slot.name = filename
			if hasattr(slot, "path"):
				slot.path = filename
		else:
			slot = items.new(name=filename, socket_type='RGBA')
			if hasattr(slot, "path"):
				slot.path = filename
	elif hasattr(node, "file_slots"):
		slots = node.file_slots
		if slots and len(slots) > 0:
			slot = slots[0]
			if hasattr(slot, "path"):
				slot.path = filename
		else:
			slot = slots.new(filename)
	else:
		print("Warning: Output node API lacks slot accessors; output may fail.")
	return slot


def get_eevee_engine_id():
	"""
	Return the correct Eevee engine identifier for the current Blender version.
	Blender 5.0+ uses 'BLENDER_EEVEE', older versions use 'BLENDER_EEVEE_NEXT'.
	"""
	if bpy.app.version >= (5, 0, 0):
		return 'BLENDER_EEVEE'
	return 'BLENDER_EEVEE_NEXT'
