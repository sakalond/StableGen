# StableGen documentation

StableGen is a Blender add-on that generates and textures 3D assets through a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server. These pages describe how to install, configure and use it.

## Getting started

| Page | Contents |
|---|---|
| [Installation](installation.md) | Requirements, ComfyUI dependencies, installing the add-on and configuring its preferences. |
| [Manual installation](MANUAL_INSTALLATION.md) | Downloading ComfyUI custom nodes and models by hand instead of using `installer.py`. |
| [Quick start](quick-start.md) | Texturing an existing model and generating a mesh with TRELLIS.2 in the StableGen panel. |

## Texturing

| Page | Contents |
|---|---|
| [Architectures and presets](architectures-and-presets.md) | Supported model architectures, the built-in presets and saving your own. |
| [Generation modes](generation-modes.md) | The texture workflows: Separately, Sequentially, Grid, Refine, Local Edit, Masked Edit and UV Inpaint. |
| [Masked Edit](masked-edit.md) | Painting a selection on the model and regenerating only that region. |
| [Cameras](cameras.md) | Adding and placing cameras, per-camera resolution, generation order and view prompts. |
| [PBR maps](pbr.md) | Extracting roughness, metallic, normal and other maps, and baking them. |

## Mesh generation and rigging

| Page | Contents |
|---|---|
| [TRELLIS.2 mesh generation](trellis2.md) | Image and prompt to 3D, texture modes, the preview gallery, folder batches and third-party licenses. |
| [Rigging and animation](rigging-and-animation.md) | SkinTokens auto-rigging and retargeting FBX or BVH animation onto a rig. |

## Output and automation

| Page | Contents |
|---|---|
| [Export](export.md) | Baking textures, exporting for game engines, orbit GIF/MP4 renders and 3D printing. |
| [Batch processing and the Scene Queue](batch-and-queue.md) | Queuing scenes, TRELLIS.2 folder batches and DAE/GLB batch texturing. |
| [Output files](output-files.md) | Layout of the output directory. |
| [Troubleshooting](troubleshooting.md) | Common problems and how to resolve them. |
