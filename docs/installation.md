# Installation

StableGen has two parts: the Blender add-on and a ComfyUI server that runs the models. The server can run on the same computer or on another machine on your network.

Installation takes four steps:

1. [Install ComfyUI](#1-install-comfyui).
2. [Install the custom nodes and models StableGen needs](#2-install-comfyui-dependencies).
3. [Install the add-on in Blender](#3-install-the-add-on).
4. [Configure the add-on preferences](#4-configure-the-preferences).

A video walkthrough of an earlier version is available from Polynox: [StableGen installation and basic usage](https://www.youtube.com/watch?v=EVNYAMnn_oQ). Some labels have changed since it was recorded.

## Requirements

| Component | Requirement |
|---|---|
| Blender | 4.2 to 4.5, or 5.1 and later. Blender 5.0 is not supported. |
| Operating system | Windows (x64), Linux (x64) or macOS on Apple Silicon. These are the platforms the add-on package declares. |
| GPU | An NVIDIA GPU with CUDA is recommended for the ComfyUI server. See the [ComfyUI documentation](https://github.com/comfyanonymous/ComfyUI) for other hardware. |
| VRAM | About 8 GB for SDXL. Plan for 16 GB or more for Qwen Image Edit. |
| ComfyUI | A working ComfyUI installation. |
| Installer script | Python 3 with the `requests` package (`tqdm` is optional and adds progress bars), and Git on the `PATH`. |
| Disk space | ComfyUI, its models (roughly 10 GB to over 50 GB depending on the packages) and generated output. |

On Blender 4.2 to 4.5, projection uses Open Shading Language and StableGen switches Cycles to CPU rendering while it builds projection materials. Blender 5.1 and later use native Raycast shader nodes and do not need OSL.

TRELLIS.2 depends on compiled CUDA extensions. The installer warns that pre-built wheels are not available for Python 3.13 and recommends running ComfyUI with Python 3.12 if you use TRELLIS.2.

## 1. Install ComfyUI

Follow the [official ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI#installing). Install it in a dedicated directory, referred to below as `<ComfyUI>`, and check that ComfyUI starts and runs before continuing.

If ComfyUI will run on a separate machine, do steps 1 and 2 on that machine.

## 2. Install ComfyUI dependencies

`installer.py` in the root of this repository downloads the ComfyUI custom nodes and models StableGen uses and places them in the correct folders.

```bash
pip install requests tqdm
python installer.py <ComfyUI>
```

If you omit the path, the script asks for it. It checks that the folder contains `models` and `custom_nodes`, then shows a menu of packages. After you choose a package, it lists every component and asks for confirmation. Files that already exist are skipped, so you can run it again to add packages. Restart ComfyUI afterwards so it loads the new custom nodes.

| # | Package | Enables | Approx. download |
|---|---|---|---|
| 1 | Minimal Core | IPAdapter and SDXL Lightning LoRA. You supply an SDXL checkpoint and ControlNet models. | 7.3 GB |
| 2 | Core + Preset Essentials | Package 1 plus the depth ControlNet the built-in SDXL presets use. You supply an SDXL checkpoint. | 9.8 GB |
| 3 | Full SDXL Setup (recommended) | Package 2 plus optional ControlNet and LoRA models and the PBR nodes (Marigold, StableDelight). You supply an SDXL checkpoint. | 19.3 GB |
| 4 | Complete SDXL | Package 3 plus the RealVisXL V5.0 checkpoint. | 26.3 GB |
| 5 | Qwen Core | Qwen Image Edit 2509 (GGUF), its VAE and text encoder, a Lightning LoRA and the GGUF loader node. | 20.3 GB |
| 6 | Qwen Extras | Package 5 plus additional Qwen Lightning LoRAs. | 22.6 GB |
| 7 | Qwen Nunchaku | Package 5 plus the Nunchaku nodes and the Int4 Nunchaku model. | 33.0 GB |
| 8 | TRELLIS.2 | The ComfyUI-TRELLIS2 custom node and its Python environment. The models (about 15.4 GB) download on first use. | 20.4 GB |
| 9 | Marigold IID | The Marigold node for PBR extraction. Its models download on first use. | 0.01 GB |
| 10 | StableDelight | The StableDelight node and model, plus the Marigold node. | 3.3 GB |
| 11 | FLUX.2 Klein | The FLUX.2 Klein 4B model, text encoder and VAE. FLUX.2 Klein is deprecated; see [Architectures and presets](architectures-and-presets.md). | 12.4 GB |
| 12 | Marigold V2 | Optional Marigold V2 albedo and normal models: the int8 Qwen Image Edit 2509 base plus LoRAs, VAEs and conditionings. Uses ComfyUI core nodes (September 2026 or newer). See [PBR maps](pbr.md). | 23.3 GB |

Sizes include first-use downloads where noted. The first run that needs those models takes longer.

Common combinations:

| Goal | Packages |
|---|---|
| Texturing with SDXL | 3, or 4 to include a checkpoint |
| Texturing with Qwen Image Edit | 5, 6 or 7 |
| Mesh generation and SDXL texturing | 3 (or 4) and 8 |
| Mesh generation and Qwen texturing | 6 and 8 |
| Add PBR extraction to a Qwen setup | 10 (packages 3 and 4 already include it) |

When you choose the TRELLIS.2 package, the installer shows a license notice about the NVIDIA libraries used by its native texture mode. See [TRELLIS.2 third-party licenses](trellis2.md#third-party-licenses).

### Components the installer does not cover

- **Qwen Image 2.1.** Place the diffusion model in `<ComfyUI>/models/diffusion_models/`, the text encoder `qwen3vl_8b_w4a8.safetensors` in `models/text_encoders/` and the VAE `qwen_image_2.1_vae_bf16.safetensors` in `models/vae/`. StableGen expects these file names for the text encoder and VAE. The files are published in the [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) repository. Qwen Image 2.1 is released under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), which allows non-commercial use only (research or evaluation). Qwen Image Edit 2509 is Apache 2.0.
- **FLUX.1.** See [Manual installation](MANUAL_INSTALLATION.md). FLUX.1 is deprecated.
- **SkinTokens auto-rigging.** Requires a ComfyUI node providing `StableGenSkinTokensAutoRig`. See [Rigging and animation](rigging-and-animation.md). The [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) code and [model weights](https://huggingface.co/VAST-AI/SkinTokens) are MIT-licensed.

To install everything by hand instead, follow [Manual installation](MANUAL_INSTALLATION.md).

### After updating ComfyUI

TRELLIS.2 runs in an isolated environment that can fall behind a ComfyUI update. Re-run the installer, or repair without the menu, then restart ComfyUI:

```
python installer.py <ComfyUI> --repair
```

`--check` only reports problems. The installer pins the TRELLIS.2 node to a tested commit, so updating the node through ComfyUI-Manager can break it until the next StableGen installer update.

## 3. Install the add-on

1. Download the add-on ZIP file from the [Releases page](https://github.com/sakalond/stablegen/releases). Do not extract it.
2. In Blender, open **Edit > Preferences > Get Extensions** (or **Add-ons**).
3. Open the drop-down menu in the top-right corner and choose **Install from Disk**.
4. Select the ZIP file. StableGen is enabled after installation; if it is not, enable it in the **Add-ons** list.

You can also drag the ZIP file into the Blender window.

The add-on bundles its Python dependencies as wheels, so nothing else needs to be installed into Blender.

## 4. Configure the preferences

Open **Edit > Preferences > Add-ons > StableGen**. In the StableGen panel, the gear icon in the panel header opens the same page.

| Preference | Purpose |
|---|---|
| Output Directory | Folder where StableGen saves generated images, bakes and exports. Generation is disabled until this is set to an existing folder. See [Output files](output-files.md). |
| Server Address | Host and port of the ComfyUI server. Default `127.0.0.1:8188`. The refresh button next to it checks the connection and reloads the model lists. |
| Use Legacy Panel | Show the original sidebar panel instead of the current one. **Reset Tips** shows dismissed hints again. |
| Save Blend File | Save a copy of the `.blend` file with packed textures in the output folder after each generation. |
| Enable 3D Printing | Show the 3D printing palette and exporters. See [Export](export.md#3d-printing). |
| Enable Experimental Texturing Features | Show research controls and let their saved values take effect. |
| Enable Deprecated Architectures | Show FLUX.1 and FLUX.2 Klein and their presets. |
| Enable Scene Queue | Show the Scene Queue. See [Batch processing](batch-and-queue.md#scene-queue). |
| ControlNet Model Assignments | Which guidance types (Depth, Canny, Normal) each ControlNet model on the server supports. Refresh the list after adding models. StableGen guesses the types from the file names; correct any that are wrong. |

**Advanced Preferences** contains the viewport camera overlay color, **Enable Debug Settings** (diagnostic tools) and network timeouts. Raise **Mesh Generation Timeout** if TRELLIS.2 post-processing on a slow server times out.

### Allow online access

StableGen communicates with ComfyUI over HTTP and WebSocket, so Blender's online access must be enabled even when the server runs locally. Open **Edit > Preferences > System > Network** and enable **Allow Online Access**. Without it, the generate button reads **Enable online access in preferences**.

## Using a remote ComfyUI server

Set **Server Address** to the remote machine's address and port, for example `192.168.1.20:8188`. ComfyUI only accepts connections from the local machine by default, so start it with `--listen` (for example `python main.py --listen 0.0.0.0`) and allow the port through the firewall.

StableGen uploads its renders to the server and downloads the results through the ComfyUI API, so the two machines do not need a shared file system. Large transfers are subject to **Transfer Timeout** in the advanced preferences.
