"""TRELLIS.2 on ComfyUI's native nodes (ComfyUI 0.42 or newer) as an API-format graph.

Built like Comfy-Org's TRELLIS.2 image-to-model template: the sparse structure, the 512 shape, the cascade upsample
and its shape pass, then optionally the texture pass, each on a KSampler, followed by remeshing, decimation, UV
unwrapping and baking with the native mesh nodes. The installer's package 8 provides the model files below.
"""

TRELLIS2_MODEL = "trellis_2_bf16.safetensors"
TRELLIS2_SHAPE_VAE = "trellis_2_shape_vae_bf16.safetensors"
TRELLIS2_TEXTURE_VAE = "trellis_2_texture_vae_bf16.safetensors"
TRELLIS2_IMAGE_ENCODER = "dino_v3_vit_l.safetensors"
TRELLIS2_BG_REMOVAL = "birefnet.safetensors"
MODEL_FILES = (TRELLIS2_MODEL, TRELLIS2_SHAPE_VAE, TRELLIS2_TEXTURE_VAE, TRELLIS2_IMAGE_ENCODER, TRELLIS2_BG_REMOVAL)

# A core node every native TRELLIS.2 graph uses: its presence means the server can run TRELLIS.2
PROBE_NODE = "Trellis2Conditioning"
# StableGen's voxel budget node (written by the installer) until Trellis2UpsampleStage has max_tokens upstream
BUDGET_NODE = "StableGenTrellis2TokenBudget"

BACKGROUND_COLORS = {'black': '#000000', 'gray': '#808080', 'white': '#FFFFFF'}
CASCADE_TARGETS = {'1024_cascade': 1024, '1536_cascade': 1536}

# What each role's node is called in progress messages
LABELS = {
    'input_image': 'Loading Image', 'unet': 'Loading Models', 'remove_bg': 'Removing Background',
    'crop': 'Cropping Image', 'conditioning': 'Conditioning', 'ss_sampler': 'Sampling Structure',
    'structure': 'Decoding Structure', 'shape_sampler': 'Sampling Shape', 'budget': 'Voxel Budget',
    'upsample': 'Upsampling Shape', 'hr_sampler': 'Sampling Shape (High Resolution)', 'shape': 'Decoding Shape',
    'remesh': 'Remeshing', 'decimate': 'Decimating', 'tex_sampler': 'Sampling Texture',
    'texture': 'Decoding Texture', 'unwrap': 'Unwrapping UVs', 'bake': 'Baking Texture', 'apply': 'Applying Texture',
    'export': 'Exporting GLB',
}


def build_trellis2_prompt(image_name, *, resolution='1024_cascade', seed=0, ss_steps=12, ss_guidance=7.5,
                          shape_steps=12, shape_guidance=7.5, texture=False, tex_steps=12, tex_guidance=3.0,
                          texture_size=4096, remove_background=True, background='black', remesh=False,
                          target_faces=None, max_tokens=49152, budget=None, prefix='trellis2'):
    """The prompt and {role: node id} of one TRELLIS.2 run.

    remesh rebuilds the surface on the server; target_faces (None: keep every face) decimates it. budget is how
    the cascade honours max_tokens: 'node' (StableGen's budget node), 'input' (a Trellis2UpsampleStage with the
    max_tokens input) or None (no budget).
    """
    prompt, nodes = {}, {}

    def node(role, cls, **inputs):
        nid = str(len(prompt) + 1)
        prompt[nid] = {'class_type': cls, 'inputs': inputs}
        nodes[role] = nid
        return nid

    image = node('input_image', 'LoadImage', image=image_name)
    unet = node('unet', 'UNETLoader', unet_name=TRELLIS2_MODEL, weight_dtype='default')
    shape_vae = node('shape_vae', 'VAELoader', vae_name=TRELLIS2_SHAPE_VAE)
    encoder = node('encoder', 'CLIPVisionLoader', clip_name=TRELLIS2_IMAGE_ENCODER)
    if remove_background:
        model = node('bg_model', 'LoadBackgroundRemovalModel', bg_removal_name=TRELLIS2_BG_REMOVAL)
        mask = [node('remove_bg', 'RemoveBackground', bg_removal_model=[model, 0], image=[image, 0]), 0]
    else:
        # LoadImage's mask is the transparent area; an image without alpha gives an empty one, so the whole
        # image becomes the subject
        mask = [node('mask', 'InvertMask', mask=[image, 1]), 0]
    crop = node('crop', 'ImageCropToMask', images=[image, 0], masks=mask, width=1024, height=1024, pad_factor=1.0,
                grow_mask=0, background=BACKGROUND_COLORS.get(background, background))
    cond = node('conditioning', 'Trellis2Conditioning', clip_vision_model=[encoder, 0], image=[crop, 0])

    # sparse structure, with the original pipeline's guidance interval and rescale (the template's CFGOverride
    # and RescaleCFG)
    ss_model = node('ss_model', 'ModelSamplingSD3', shift=5.0, model=[node(
        'ss_rescale', 'RescaleCFG', multiplier=0.7, model=[node(
            'ss_interval', 'CFGOverride', model=[unet, 0], cfg=1.0, start_percent=0.667, end_percent=1.0), 0]), 0])
    empty = node('empty', 'EmptyTrellis2LatentStructure', batch_size=1)
    ss = node('ss_sampler', 'KSampler', model=[ss_model, 0], positive=[cond, 0], negative=[cond, 1],
              latent_image=[empty, 0], seed=seed, steps=ss_steps, cfg=ss_guidance, sampler_name='euler',
              scheduler='normal', denoise=1.0)
    structure = node('structure', 'VaeDecodeStructureTrellis2', samples=[ss, 0], vae=[shape_vae, 0], resolution='32')

    shape_model = node('shape_rescale', 'RescaleCFG', multiplier=0.5, model=[node(
        'shape_interval', 'CFGOverride', model=[unet, 0], cfg=1.0, start_percent=0.769, end_percent=1.0), 0])
    stage = node('shape_stage', 'Trellis2ShapeStage', positive=[cond, 0], negative=[cond, 1], voxel=[structure, 0])
    shape = node('shape_sampler', 'KSampler', model=[shape_model, 0], positive=[stage, 0], negative=[stage, 1],
                 latent_image=[stage, 2], seed=seed, steps=shape_steps, cfg=shape_guidance, sampler_name='euler',
                 scheduler='normal', denoise=1.0)
    stage_out, shape_out = stage, shape
    if resolution in CASCADE_TARGETS:
        target = CASCADE_TARGETS[resolution]
        if budget == 'node':
            target = [node('budget', BUDGET_NODE, shape_latent=[shape, 0], vae=[shape_vae, 0],
                           target_resolution=target, max_tokens=max_tokens), 0]
        up = node('upsample', 'Trellis2UpsampleStage', positive=[stage, 0], negative=[stage, 1],
                  shape_latent=[shape, 0], vae=[shape_vae, 0], target_resolution=target)
        if budget == 'input':
            prompt[up]['inputs']['max_tokens'] = max_tokens
        shape_out = node('hr_sampler', 'KSampler', model=[shape_model, 0], positive=[up, 0], negative=[up, 1],
                         latent_image=[up, 2], seed=seed, steps=shape_steps, cfg=shape_guidance,
                         sampler_name='euler', scheduler='simple', denoise=1.0)
        stage_out = up
    decoded = node('shape', 'VaeDecodeShapeTrellis', samples=[shape_out, 0], vae=[shape_vae, 0])

    mesh = _remesh_and_decimate(node, [decoded, 0], remesh, target_faces)

    if texture:
        size = max(256, min(8192, int(texture_size) // 256 * 256))    # UnwrapMesh's atlas steps
        tex_vae = node('tex_vae', 'VAELoader', vae_name=TRELLIS2_TEXTURE_VAE)
        tstage = node('tex_stage', 'Trellis2TextureStage', positive=[stage_out, 0], negative=[stage_out, 1],
                      shape_latent=[shape_out, 0])
        tex = node('tex_sampler', 'KSampler', model=[unet, 0], positive=[tstage, 0], negative=[tstage, 1],
                   latent_image=[tstage, 2], seed=seed, steps=tex_steps, cfg=tex_guidance, sampler_name='euler',
                   scheduler='normal', denoise=1.0)
        voxels = node('texture', 'VaeDecodeTextureTrellis', samples=[tex, 0], vae=[tex_vae, 0],
                      shape_subdivides=[decoded, 1])
        unwrapped = node('unwrap', 'UnwrapMesh', mesh=mesh, segmenter='pec', resolution=size, padding=1,
                         weld_distance=0.0002)
        baked = node('bake', 'BakeTextureFromVoxel', mesh=[unwrapped, 0], voxel_colors=[voxels, 0],
                     reference_mesh=[decoded, 0], texture_size=size)
        mesh = [node('apply', 'ApplyTextureToMesh', mesh=[unwrapped, 0], base_color=[baked, 0],
                     metallic=[baked, 1], roughness=[baked, 2]), 0]
    node('export', 'SaveGLB', mesh=mesh, filename_prefix=prefix)
    return prompt, nodes


def _remesh_and_decimate(node, mesh, remesh, target_faces):
    """The server's post-processing of a MESH: QDC remeshing (RemeshMesh), then decimation to target_faces."""
    if remesh:
        mesh = [node('remesh', 'RemeshMesh', **{
            'mesh': mesh, 'resolution': 512, 'band': 1.0, 'project_back': 0.0, 'fix_poles': False,
            'smooth_iters': 0, 'drop_small_components': 0.01, 'precluster_max_verts': 20000000,
            'sign_mode': 'udf', 'sign_mode.qef': False, 'sign_mode.drop_inverted_components': False,
            'sign_mode.drop_enclosed_components': False}), 0]
    if target_faces:
        mesh = [node('decimate', 'DecimateMesh', mesh=mesh, target_face_count=int(target_faces),
                     placement_mode='midpoint'), 0]
    return mesh


def build_mesh_postprocess_prompt(model_file, *, remesh=False, target_faces=None, prefix='postprocess'):
    """An existing mesh through the server post-processing TRELLIS.2 generation uses: Load 3D (Advanced) reads the
    uploaded GLB (model_file, relative to ComfyUI's input folder; its viewer state is unused here), Get 3D Components
    makes it a MESH, then remesh and decimate as above, and SaveGLB writes the result. Returns (prompt, nodes)."""
    prompt, nodes = {}, {}

    def node(role, cls, **inputs):
        nid = str(len(prompt) + 1)
        prompt[nid] = {'class_type': cls, 'inputs': inputs}
        nodes[role] = nid
        return nid

    loaded = node('load', 'Load3DAdvanced', model_file=model_file, viewport_state={}, width=1024, height=1024)
    mesh = [node('components', 'Get3DComponents', model_3d=[loaded, 0]), 0]
    node('export', 'SaveGLB', mesh=_remesh_and_decimate(node, mesh, remesh, target_faces), filename_prefix=prefix)
    return prompt, nodes


# Core nodes the server post-processing of an existing mesh needs (ComfyUI 0.42 or newer)
POSTPROCESS_NODES = ('Load3DAdvanced', 'Get3DComponents', 'RemeshMesh', 'DecimateMesh', 'SaveGLB')


def budget_mode(upsample_info, budget_node_info):
    """How this server can honour the voxel budget, from the /object_info entries of Trellis2UpsampleStage and
    StableGen's budget node (None when absent)."""
    if upsample_info:
        inputs = {**upsample_info.get('input', {}).get('required', {}),
                  **upsample_info.get('input', {}).get('optional', {})}
        if 'max_tokens' in inputs:
            return 'input'
    return 'node' if budget_node_info else None


def progress_weights(nodes):
    """Phase progress (percent) when each node starts, for the progress bar."""
    order = ['input_image', 'unet', 'remove_bg', 'conditioning', 'ss_sampler', 'shape_sampler', 'budget',
             'upsample', 'hr_sampler', 'shape', 'remesh', 'decimate', 'tex_sampler', 'texture', 'unwrap', 'bake',
             'export']
    # time each step takes, roughly (sampling dominates; the high-resolution pass and texturing most of all)
    cost = {'input_image': 1, 'unet': 3, 'remove_bg': 2, 'conditioning': 2, 'ss_sampler': 6, 'shape_sampler': 12,
            'budget': 2, 'upsample': 3, 'hr_sampler': 25, 'shape': 4, 'remesh': 8, 'decimate': 4,
            'tex_sampler': 18, 'texture': 4, 'unwrap': 8, 'bake': 6, 'export': 2}
    present = [role for role in order if role in nodes]
    total = sum(cost[role] for role in present)
    weights, done = {}, 0
    for role in present:
        weights[nodes[role]] = round(100 * done / total)
        done += cost[role]
    return weights
