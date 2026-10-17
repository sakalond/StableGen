<h1 align="center"><img src="docs/img/brand/stablegen-logo.svg" width="44" align="absmiddle" alt="StableGen logo">&ensp;StableGen</h1>
<p align="center">Generate, texture and rig 3D assets in Blender.</p>
<p align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-007ec6" alt="License: GPL-3.0"></a>
<img src="https://img.shields.io/badge/Blender-4.2--4.5%20%7C%205.1%2B-e5722c" alt="Blender 4.2 to 4.5, or 5.1 and later">
<a href="https://github.com/sakalond/stablegen/releases"><img src="https://img.shields.io/github/downloads/sakalond/stablegen/total" alt="Downloads"></a>
</p>
<p align="center"><a href="https://github.com/sakalond/StableGen/wiki">Documentation</a> · <a href="https://github.com/sakalond/StableGen/wiki/Installation">Installation</a> · <a href="https://github.com/sakalond/StableGen/wiki/Quick-start">Quick start</a> · <a href="https://github.com/sakalond/stablegen/releases">Releases</a></p>

StableGen is a Blender add-on that textures meshes and whole scenes with diffusion models, and generates new meshes from a prompt or an image. It renders guidance images from cameras in your scene, sends the work to a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server on the same computer or elsewhere on your network, and projects the results back onto the meshes.

<p>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/hero-dark.webp">
  <img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/hero-light.webp" alt="A clay room textured in three interior styles: Scandinavian, spaceship and wizard's study">
</picture>
</p>

## One prompt to a rigged asset

Describe a character and pick one of the preview images; StableGen does the rest in one run. TRELLIS.2 builds the mesh, SkinTokens rigs it, and views generated around it are projected into its texture. A run can also start from an image, skip the rig, or keep the texture TRELLIS.2 makes itself. [TRELLIS.2](https://github.com/sakalond/StableGen/wiki/TRELLIS.2-mesh-generation)

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-phone-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-phone-light.webp" alt="The StableGen panel: a prompt, the preview gallery, then the mesh, rig and texture generated in Blender"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-light.webp" alt="The StableGen panel: a prompt, the preview gallery, then the mesh, rig and texture generated in Blender"></picture>
</p>

## What it does

### Texture any model

Place cameras around one object or a whole scene. StableGen generates a view from each camera and projects it onto every surface that camera sees; each view is generated with the earlier ones as context, so the texture stays consistent all the way round. [Generation modes](https://github.com/sakalond/StableGen/wiki/Generation-modes) · [Cameras](https://github.com/sakalond/StableGen/wiki/Cameras)

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/texture-phone-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/texture-phone-light.webp" alt="A floating island textured view by view from ten generated camera views"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/texture-wide-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/texture-wide-light.webp" alt="A floating island textured view by view from ten generated camera views"></picture>
</p>

### Edit part of a texture

Paint over the part you want to change and describe the change. Masked Edit regenerates only the painted region, in every view that sees it, and leaves the rest of the texture as it was. [Masked Edit](https://github.com/sakalond/StableGen/wiki/Masked-Edit)

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/edit-phone-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/edit-phone-light.webp" alt="A courier's yellow jacket replaced in turn by a black neon tech jacket, desert camo and a chrome racing jacket"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/edit-wide-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/edit-wide-light.webp" alt="A courier's yellow jacket replaced in turn by a black neon tech jacket, desert camo and a chrome racing jacket"></picture>
</p>

### PBR materials

Marigold estimates albedo, roughness, metallic and normal maps from the generated views. StableGen projects them like the color and bakes them into standard maps for other software. [PBR maps](https://github.com/sakalond/StableGen/wiki/PBR-maps)

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pbr-phone-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pbr-phone-light.webp" alt="A log house under a moving light, with its base color, roughness, metallic and normal maps"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pbr-wide-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pbr-wide-light.webp" alt="A log house under a moving light, with its base color, roughness, metallic and normal maps"></picture>
</p>

### Rig and animate

SkinTokens adds a skeleton and skin weights to a character, during a TRELLIS.2 run or on any mesh you select. Motion from FBX or BVH files, such as Mixamo animations, can then be retargeted onto the rig. [Rigging and animation](https://github.com/sakalond/StableGen/wiki/Rigging-and-animation)

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/rig-phone-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/rig-phone-light.webp" alt="A generated character with its skeleton, then walking with a retargeted Mixamo animation"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/rig-wide-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/rig-wide-light.webp" alt="A generated character with its skeleton, then walking with a retargeted Mixamo animation"></picture>
</p>

### Export to a game engine

Bake the result, PBR maps included, into UV textures and export it with its rig and animations: glTF for Godot and the web, or FBX for Unreal Engine and Unity. Below, a village in Unreal Engine 5.7 built only from StableGen exports: the houses, props, trees and ground as GLB with PBR maps, and five rigged characters as FBX with their walk. Orbit GIFs and MP4s, multi-color 3MF files for 3D printing, and the scene queue for unattended batches are in the same place. [Export](https://github.com/sakalond/StableGen/wiki/Export) · [Batch processing](https://github.com/sakalond/StableGen/wiki/Batch-processing-and-the-Scene-Queue)

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/export-phone-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/export-phone-light.webp" alt="A village of StableGen assets in Unreal Engine 5.7: the level editor with the imported assets, then aerial, market, street and village shots with five walking characters"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/export-wide-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/export-wide-light.webp" alt="A village of StableGen assets in Unreal Engine 5.7: the level editor with the imported assets, then aerial, market, street and village shots with five walking characters"></picture>
</p>

## Working by hand

Each step can also be done by hand: reorder the cameras and give views their own prompts, change how the views blend and reproject without generating again, extract PBR maps, and repaint a region with Masked Edit.

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/tools-phone-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/tools-phone-light.webp" alt="The StableGen panel: camera order and view prompts, blending, PBR maps and Masked Edit"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/tools-dark.webp"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/tools-light.webp" alt="The StableGen panel: camera order and view prompts, blending, PBR maps and Masked Edit"></picture>
</p>

## How it works

StableGen runs inside Blender and sends the model work to a ComfyUI server, on the same computer or another one on the network. You start from a prompt, an image or your own model, and the results come back into the scene as materials, meshes and armatures.

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-narrow-dark.svg"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-narrow-light.svg" width="100%" alt="In Blender: prompt, mesh, cameras, projection, bake and export. On the ComfyUI server: mesh generation with TRELLIS.2, rigging with SkinTokens, diffusion with SDXL or Qwen Image, and PBR maps with Marigold."></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-diagram-dark.svg"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/pipeline-diagram-light.svg" width="100%" alt="In Blender: prompt, mesh, cameras, projection, bake and export. On the ComfyUI server: mesh generation with TRELLIS.2, rigging with SkinTokens, diffusion with SDXL or Qwen Image, and PBR maps with Marigold."></picture>
</p>

Texturing works by projection. Each camera gets a generated view that follows the geometry through a depth guide, and the views are blended onto the surface by how directly each camera sees it.

<p>
<picture><source media="(min-width: 768px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/projection-narrow-dark.svg"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/projection-narrow-light.svg" width="100%" alt="Projection in three steps: place cameras, generate a view for each from a depth guide, project and blend the views"></picture><picture><source media="(max-width: 767px)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/blank.svg"><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/projection-diagram-dark.svg"><img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/projection-diagram-light.svg" width="100%" alt="Projection in three steps: place cameras, generate a view for each from a depth guide, project and blend the views"></picture>
</p>

### Built on

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) runs every model, locally or on another machine.
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) generates meshes, through [ComfyUI-TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2).
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with ControlNet and IPAdapter, and [Qwen Image](https://github.com/QwenLM/Qwen-Image), generate the texture views.
- [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) rigs characters.
- [Marigold](https://github.com/prs-eth/Marigold) and [StableDelight](https://github.com/Stable-X/StableDelight) estimate the PBR maps.

## Gallery

<p align="center">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g0.webp" width="150" alt="Inventor with goggles and a leather apron">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g1.webp" width="150" alt="Wizard in purple and gold robes with a crystal staff">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g2.webp" width="150" alt="Stone windmill with wooden sails">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g3.webp" width="150" alt="Brass astrolabe">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g4.webp" width="150" alt="Adventurer in steel armor">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g5.webp" width="150" alt="Market stall with a striped awning and baskets of produce">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g6.webp" width="150" alt="White mecha robot">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g7.webp" width="150" alt="Treasure chest full of gold coins">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g8.webp" width="150" alt="Cyberpunk vending machine covered in graffiti">
<img src="https://raw.githubusercontent.com/sakalond/stablegen-media/main/readme/g9.webp" width="150" alt="Hollow tree stump with glowing mushrooms">
</p>

## Requirements

- Blender 4.2 to 4.5, or 5.1 and later. Blender 5.0 is not supported.
- A ComfyUI server, preferably with an NVIDIA GPU: about 8 GB of VRAM for SDXL, and 16 GB or more for Qwen Image.
- Windows, Linux, or macOS on Apple Silicon.
- 10 GB to over 50 GB of disk space for models, depending on what you install.

## Installation

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI#installing) and check that it runs.
2. Install the custom nodes and models StableGen uses with the installer in this repository, then restart ComfyUI:
   ```bash
   python installer.py <path to ComfyUI>
   ```
   Package 3 sets up SDXL texturing, package 13 Qwen Image 2.1, package 8 mesh generation and package 14 auto-rigging.
3. Download the add-on ZIP from [Releases](https://github.com/sakalond/stablegen/releases). In Blender, open **Edit > Preferences > Get Extensions**, choose **Install from Disk** in the top-right menu, and select the ZIP.
4. In the add-on preferences, set **Output Directory** and **Server Address** (`127.0.0.1:8188` by default). Under **System > Network**, enable **Allow Online Access**.

[Installation](https://github.com/sakalond/StableGen/wiki/Installation) covers every installer package, unattended installs, a remote server and manual setup.

## Quick start

Press <kbd>N</kbd> in the 3D Viewport and open the **StableGen** tab.

**Texture a model**

1. Select the mesh and click **Add cameras**.
2. Choose an **Architecture** and a **Preset**, then a **Checkpoint** on the **Texture** page.
3. Describe the surface in **Texture prompt** and click **Generate textures**.
4. Switch the viewport to **Rendered** shading to see the result. **Bake** and **Export** turn it into ordinary textures and files.

**Generate an asset**

1. Set **Architecture** to **TRELLIS.2** and choose a **(Mesh + Texture)** preset.
2. On the **Mesh** page, set **Generate mesh from** to **Prompt**, describe the object and enable **Preview gallery**.
3. To rig a character in the same run, enable **Auto-rig** under **Rigging & Animation**.
4. Click **Generate asset**, then pick a preview image.

[Quick start](https://github.com/sakalond/StableGen/wiki/Quick-start) goes through both in detail.

## Documentation

[Installation](https://github.com/sakalond/StableGen/wiki/Installation) · [Quick start](https://github.com/sakalond/StableGen/wiki/Quick-start) · [Architectures and presets](https://github.com/sakalond/StableGen/wiki/Architectures-and-presets) · [Generation modes](https://github.com/sakalond/StableGen/wiki/Generation-modes) · [Cameras](https://github.com/sakalond/StableGen/wiki/Cameras) · [Masked Edit](https://github.com/sakalond/StableGen/wiki/Masked-Edit) · [PBR maps](https://github.com/sakalond/StableGen/wiki/PBR-maps) · [TRELLIS.2](https://github.com/sakalond/StableGen/wiki/TRELLIS.2-mesh-generation) · [Rigging and animation](https://github.com/sakalond/StableGen/wiki/Rigging-and-animation) · [Export](https://github.com/sakalond/StableGen/wiki/Export) · [Batch processing](https://github.com/sakalond/StableGen/wiki/Batch-processing-and-the-Scene-Queue) · [Output files](https://github.com/sakalond/StableGen/wiki/Output-files) · [Troubleshooting](https://github.com/sakalond/StableGen/wiki/Troubleshooting)

## License

StableGen is released under the [GNU General Public License v3.0](LICENSE). The models and nodes it runs on ComfyUI have their own licenses, and two of them are for non-commercial use only:

- Qwen Image 2.1 is released under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE). Qwen Image Edit and SDXL are not affected.
- The TRELLIS.2 **Native** texture mode uses NVIDIA's nvdiffrast and nvdiffrec. The other modes do not use them. See [TRELLIS.2 third-party licenses](https://github.com/sakalond/StableGen/wiki/TRELLIS.2-mesh-generation#third-party-licenses).

## Acknowledgements

StableGen began as a bachelor's thesis by Ondřej Sakala at the Faculty of Information Technology, Czech Technical University in Prague, supervised by Ing. Radek Richtr, Ph.D. ([full text](https://dspace.cvut.cz/handle/10467/123567)). Part of its development took place during a study stay at Kanazawa University in summer 2026; travel costs were supported by a grant from Nadace „Nadání Josefa, Marie a Zdeňky Hlávkových“.

It builds on [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [ComfyUI-TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2) and [Blender](https://www.blender.org/), and was inspired by the [Dream Textures](https://github.com/carson-katri/dream-textures) and [Diffused Texture](https://github.com/FrederikHasecke/diffused-texture-addon) add-ons.

<details>
<summary>Research StableGen draws on</summary>

- Ho et al. (2020), Denoising Diffusion Probabilistic Models, [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Rombach et al. (2022), Latent Diffusion Models, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Zhang et al. (2023), ControlNet, [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
- Ye et al. (2023), IP-Adapter, [arXiv:2308.06721](https://arxiv.org/abs/2308.06721)
- Chen et al. (2023), Text2Tex, [arXiv:2303.11396](https://arxiv.org/abs/2303.11396)
- Richardson et al. (2023), TEXTure, [arXiv:2302.01721](https://arxiv.org/abs/2302.01721)
- Zeng et al. (2023), Paint3D, [arXiv:2312.13913](https://arxiv.org/abs/2312.13913)
- Le et al. (2024), EucliDreamer, [arXiv:2311.15573](https://arxiv.org/abs/2311.15573)
- Ceylan et al. (2024), MatAtlas, [arXiv:2404.02899](https://arxiv.org/abs/2404.02899)
- Siddiqui et al. (2022), Texturify, [arXiv:2204.02411](https://arxiv.org/abs/2204.02411)
- Bokhovkin et al. (2023), Mesh2Tex, [arXiv:2304.05868](https://arxiv.org/abs/2304.05868)
- Levin and Fried (2024), Differential Diffusion, [arXiv:2306.00950](https://arxiv.org/abs/2306.00950)

</details>

## Contact

Ask questions and share results in [Discussions](https://github.com/sakalond/StableGen/discussions), and report bugs in [Issues](https://github.com/sakalond/StableGen/issues). Ondřej Sakala: `sakalaondrej@gmail.com`, [@sakalond](https://x.com/sakalond) on X.
