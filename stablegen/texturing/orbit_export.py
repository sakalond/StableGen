"""Orbit GIF/MP4 export operator."""

import os
import math
import time
import tempfile
import shutil
from ..core.property_help import ORBIT_HELP

import bpy  # pylint: disable=import-error
import numpy as np
import mathutils
from ..utils import get_dir_path, get_eevee_engine_id, sg_modal_active
import imageio
import imageio_ffmpeg

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
        description=ORBIT_HELP['engine'],
        items=[
            ('BLENDER_WORKBENCH', "Workbench", "Use Workbench render engine"),
            ('EEVEE', "Eevee", "Use Eevee render engine"),
            ('CYCLES', "Cycles", "Use Cycles render engine")
        ],
        default='CYCLES'
    ) # type: ignore

    interpolation: bpy.props.EnumProperty(
        name="Rotation Curve",
        description=ORBIT_HELP['interpolation'],
        items=[
            ('LINEAR', "Linear", "Constant speed throughout the orbit"),
            ('BEZIER', "Ease In/Out", "Smooth acceleration and deceleration"),
        ],
        default='LINEAR'
    ) # type: ignore

    # ── PBR showcase settings ─────────────────────────────────────────
    use_hdri: bpy.props.BoolProperty(
        name="HDRI Environment",
        description=ORBIT_HELP['use_hdri'],
        default=False
    ) # type: ignore

    hdri_rotation: bpy.props.FloatProperty(
        name="Environment Rotation",
        description=ORBIT_HELP['hdri_rotation'],
        default=0.0,
        min=0.0,
        max=360.0,
        subtype='ANGLE'
    ) # type: ignore

    env_mode: bpy.props.EnumProperty(
        name="Environment Mode",
        description=ORBIT_HELP['env_mode'],
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

    use_denoiser: bpy.props.BoolProperty(
        name="Denoise",
        description=ORBIT_HELP['use_denoiser'],
        default=True
    ) # type: ignore

    use_gpu: bpy.props.BoolProperty(
        name="GPU Compute",
        description="Use GPU for Cycles rendering (Blender 5.1+ only). "
                    "Disable to force CPU rendering",
        default=True
    ) # type: ignore

    filename_suffix: bpy.props.StringProperty(
        name="Filename Suffix",
        description="Internal: appended to the output filename (e.g. '_no_pbr')",
        default="",
        options={'HIDDEN'}
    ) # type: ignore

    _timer = None
    _rendering = False
    _cancelled = False # Added flag to track cancellation
    _handle_complete = None
    _handle_cancel = None
    _initial_settings = {}
    _temp_empty = None # Added for the pivot empty
    _env_mapping_node = None   # For HDRI rotation animation
    _original_world_data = None  # Store entire world setup for restore
    _output_path = "" # Internal variable to store the final GIF path
    _output_path_mp4 = "" # Internal variable to store the final MP4 path
    _temp_dir = "" # Temporary directory for frames
    _frame_paths = [] # List to store paths of rendered frames
    _original_camera_parent = None # Store original camera parent
    _original_camera_matrix = None # Store original camera matrix
    _original_compute_device_type = None  # For GPU/CPU toggle restore

    @classmethod
    def poll(cls, context):
        try:
            import imageio
        except ImportError:
            cls.poll_message_set("Python module 'imageio' not installed")
            return False
        if context.active_object is None or context.active_object.type not in {'MESH', 'EMPTY'}:
            cls.poll_message_set("Select a mesh or empty object first")
            return False
        if context.scene.camera is None:
            cls.poll_message_set("No active camera in the scene")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

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
            suffix = self.filename_suffix or ""
            self._output_path = os.path.join(revision_dir, f"orbit{suffix}.gif")
            self._output_path_mp4 = os.path.join(revision_dir, f"orbit{suffix}.mp4") # MP4 path
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
            # No HDRI world set up — create a *separate* procedural sky world
            # so the original world node tree is never destroyed.
            print("[StableGen] Orbit GIF: No HDRI found, creating sky fallback "
                  "(original world preserved)")
            fallback_world = bpy.data.worlds.new("SG_OrbitSkyFallback")
            fallback_world.use_nodes = True
            fb_tree = fallback_world.node_tree
            fb_tree.nodes.clear()

            tex_coord = fb_tree.nodes.new("ShaderNodeTexCoord")
            tex_coord.location = (-800, 300)

            mapping = fb_tree.nodes.new("ShaderNodeMapping")
            mapping.location = (-600, 300)
            mapping.inputs["Rotation"].default_value[2] = self.hdri_rotation
            self._env_mapping_node = mapping

            sky_tex = fb_tree.nodes.new("ShaderNodeTexSky")
            sky_tex.location = (-300, 300)
            # Set sky type — Blender 5.x replaced 'NISHITA' with new names
            for sky_type in ('NISHITA', 'HOSEK_WILKIE', 'PREETHAM'):
                try:
                    sky_tex.sky_type = sky_type
                    break
                except TypeError:
                    continue
            # Sun attributes vary by sky type / version
            try:
                sky_tex.sun_elevation = math.radians(30)
                sky_tex.sun_rotation = math.radians(45)
            except AttributeError:
                pass

            bg_node = fb_tree.nodes.new("ShaderNodeBackground")
            bg_node.location = (0, 300)

            output_node = fb_tree.nodes.new("ShaderNodeOutputWorld")
            output_node.location = (200, 300)

            fb_tree.links.new(tex_coord.outputs["Generated"],
                           mapping.inputs["Vector"])
            fb_tree.links.new(mapping.outputs["Vector"],
                           sky_tex.inputs["Vector"])
            fb_tree.links.new(sky_tex.outputs["Color"],
                           bg_node.inputs["Color"])
            fb_tree.links.new(bg_node.outputs["Background"],
                           output_node.inputs["Surface"])

            # Swap the scene world to the fallback; original is stored for restore
            context.scene.world = fallback_world
            self._original_world_data['created_fallback'] = True
            self._original_world_data['fallback_world'] = fallback_world
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

        # When called via EXEC_DEFAULT (e.g. from the queue) invoke()
        # is skipped, so initialise paths / temp dir here if needed.
        if not self._temp_dir:
            try:
                revision_dir = get_dir_path(context, "revision")
                os.makedirs(revision_dir, exist_ok=True)
                suffix = self.filename_suffix or ""
                self._output_path = os.path.join(revision_dir, f"orbit{suffix}.gif")
                self._output_path_mp4 = os.path.join(revision_dir, f"orbit{suffix}.mp4")
            except Exception as e:
                self.report({'ERROR'}, f"Could not determine output directory: {e}")
                return {'CANCELLED'}
            try:
                self._temp_dir = tempfile.mkdtemp(prefix="blender_gif_")
            except Exception as e:
                self.report({'ERROR'}, f"Could not create temporary directory: {e}")
                return {'CANCELLED'}

        # Store Initial Settings
        self._initial_settings = {
            'frame_start': scene.frame_start,
            'frame_end': scene.frame_end,
            'fps': render.fps,
            'camera': scene.camera, # Store the camera object itself
            'filepath': render.filepath,
            'file_format': render.image_settings.file_format,
            'color_mode': render.image_settings.color_mode,
            'resolution_x': render.resolution_x,
            'resolution_y': render.resolution_y,
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
        # Use a fixed 1024×1024 base so GIF resolution is independent of
        # whatever the generation pipeline left in resolution_x/y.
        render.resolution_x = 1024
        render.resolution_y = 1024
        render.filepath = os.path.join(self._temp_dir, "frame_") # Base path for frames
        render.image_settings.file_format = 'PNG' # Render as PNG sequence
        render.image_settings.color_mode = 'RGBA' # Use RGBA (needed for potential alpha, even if we make background opaque)
        render.resolution_percentage = self.resolution_percentage
        render.use_overwrite = True
        render.use_placeholder = False
        render.film_transparent = True 
        cycles.samples = self.samples # Set samples for rendering

        # ── GPU / CPU Compute (Blender 5.1+) ───────────────────────────
        if bpy.app.version >= (5, 1, 0) and scene.render.engine == 'CYCLES':
            try:
                cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
                self._original_compute_device_type = cycles_prefs.compute_device_type
                self._initial_settings['cycles_device'] = scene.cycles.device
                if self.use_gpu:
                    # Probe GPU backends, pick the first that has devices
                    for backend in ('CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'):
                        try:
                            cycles_prefs.compute_device_type = backend
                            cycles_prefs.get_devices()
                            gpu_devs = [d for d in cycles_prefs.devices if d.type == backend]
                            if gpu_devs:
                                for d in gpu_devs:
                                    d.use = True
                                scene.cycles.device = 'GPU'
                                print(f"[GIF Export] Using {backend} GPU: "
                                      f"{', '.join(d.name for d in gpu_devs)}")
                                break
                        except Exception:
                            continue
                    else:
                        # No GPU found, fall back to CPU
                        cycles_prefs.compute_device_type = 'NONE'
                        scene.cycles.device = 'CPU'
                        print("[GIF Export] No GPU backend found, falling back to CPU")
                else:
                    cycles_prefs.compute_device_type = 'NONE'
                    scene.cycles.device = 'CPU'
            except Exception as e:
                print(f"[GIF Export] Could not configure compute device: {e}")
                self._original_compute_device_type = None

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
        ExportOrbitGIF._rendering = True   # class-level flag for external polling
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
                    print("[StableGen] Warning: Camera parent changed during render, restoring world matrix only.")
                    original_camera.matrix_world = self._original_camera_matrix
                # Clear potentially stale parent inverse matrix
                original_camera.matrix_parent_inverse.identity()

            except Exception as e:
                print(f"[StableGen] Warning: Could not restore camera state for '{original_camera.name}': {e}")

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
                elif key == 'cycles_device':
                    cycles.device = value
                elif key == 'engine': # Restore engine
                    setattr(render, key, value)
                elif key == 'film_transparent': # Restore film transparency
                    setattr(render, key, value)
                elif key == 'use_compositing': # Restore compositor pipeline state
                    render.use_compositing = value
                elif key in ('resolution_x', 'resolution_y'):
                    setattr(render, key, value)
                elif hasattr(render, key):
                    setattr(render, key, value)
                elif hasattr(scene, key):
                    setattr(scene, key, value)
            except Exception as e:
                print(f"[StableGen] Warning: Could not restore setting '{key}': {e}")

        self._initial_settings = {} # Clear stored settings
        self._original_camera_parent = None
        self._original_camera_matrix = None

        # Restore GPU/CPU compute device type (Blender 5.1+)
        if self._original_compute_device_type is not None:
            try:
                cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
                cycles_prefs.compute_device_type = self._original_compute_device_type
            except Exception as e:
                print(f"[StableGen] Warning: Could not restore compute device type: {e}")
            self._original_compute_device_type = None

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

        # Restore the world environment (remove injected mapping nodes)
        if self._original_world_data:
            try:
                world = self._original_world_data['world_ref']

                if self._original_world_data.get('created_fallback'):
                    # We swapped in a separate fallback world — restore original
                    fallback_world = self._original_world_data.get(
                        'fallback_world')

                    # Clean up animation data on the fallback world
                    if (fallback_world and fallback_world.node_tree
                            and fallback_world.node_tree.animation_data):
                        action = fallback_world.node_tree.animation_data.action
                        if action and action.name in bpy.data.actions:
                            bpy.data.actions.remove(action)
                        fallback_world.node_tree.animation_data_clear()

                    # Restore the original world
                    if world and world.name in bpy.data.worlds:
                        bpy.context.scene.world = world
                    # Remove the temporary fallback world datablock
                    if (fallback_world
                            and fallback_world.name in bpy.data.worlds):
                        bpy.data.worlds.remove(fallback_world)
                elif world and world.name in bpy.data.worlds:
                    # Remove HDRI animation keyframes on the original world
                    if world.node_tree and world.node_tree.animation_data:
                        action = world.node_tree.animation_data.action
                        if action and action.name in bpy.data.actions:
                            bpy.data.actions.remove(action)
                        world.node_tree.animation_data_clear()

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
                print(f"[StableGen] Warning: Could not restore world environment: {e}")
            self._original_world_data = None
        self._env_mapping_node = None


        self._rendering = False # Ensure rendering flag is reset
        ExportOrbitGIF._rendering = False  # class-level flag for external polling
        self._cancelled = False # Reset cancellation flag

        # Clean up temporary directory if it exists
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                print(f"[StableGen] Removed temporary directory: {self._temp_dir}")
            except Exception as e:
                print(f"[StableGen] Warning: Could not remove temporary directory '{self._temp_dir}': {e}")
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
                     print(f"[StableGen] Warning: Frame file not found after render: {filepath} or {alt_filepath}")


    def render_complete_handler(self, scene, _):
        try:
            # Only respond if we're still in the rendering phase
            if not getattr(self, "_rendering", False):
                return

            print("[StableGen] Frame rendering complete.")
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
             print(f"[StableGen] Error in render_complete_handler: {e}") # Log other potential errors


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

        print(f"[StableGen] Assembling output files from {len(self._frame_paths)} frames...")
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
                    print(f"[StableGen] Warning: Frame file disappeared before processing: {filename}")
                    continue # Skip missing frame

            if not images: # Check if any images were actually loaded
                 self.report({'ERROR'}, "No valid frame images found to create output files.")
                 return False

            # Calculate duration per frame for imageio (in seconds)
            frame_duration = 1.0 / self.frame_rate

            # --- Create GIF ---
            print(f"[StableGen] Creating GIF: {self._output_path}")
            try:
                 # imageio v3+ uses 'duration' in seconds per frame
                 imageio.mimsave(self._output_path, images, fps=self.frame_rate, loop=0, disposal=2) # loop=0 means infinite loop
                 print(f"[StableGen] GIF saved successfully.")
                 gif_success = True
            except TypeError:
                      # Fallback if 'duration' is not accepted 
                      imageio.mimsave(self._output_path, images, duration=frame_duration, loop=0)
                      print(f"[StableGen] GIF saved successfully (fallback duration).")
                      gif_success = True
            except Exception as e:
                 self.report({'ERROR'}, f"Failed to create GIF: {e}")
                 # Print traceback for debugging GIF errors
                 import traceback
                 traceback.print_exc()

            # --- Create MP4 ---
            print(f"[StableGen] Creating MP4: {self._output_path_mp4}")
            try:
                imageio.mimsave(self._output_path_mp4, images, format='mp4', fps=self.frame_rate, quality=8)
                print(f"[StableGen] MP4 saved successfully.")
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
