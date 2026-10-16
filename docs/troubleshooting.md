# Troubleshooting

When something fails, check two places first:

- **Blender's system console.** On Windows, open it with **Window > Toggle System Console**; on Linux and macOS, start Blender from a terminal. StableGen prefixes its messages with `[StableGen]`.
- **The ComfyUI console.** Errors about missing nodes, missing models or memory appear here.

## The StableGen tab is missing

Check that the add-on is installed and enabled under **Edit > Preferences > Add-ons**, then press **N** in the 3D Viewport and look for the **StableGen** tab. If you prefer the original interface, enable **Use Legacy Panel** in the add-on preferences.

## The generate button is disabled

The button, or the status line above it in the StableGen panel, states the reason.

| Message | Fix |
|---|---|
| Output Path Invalid | Set **Output Directory** in the preferences to a folder that exists. |
| Server Address Missing | Set **Server Address** in the preferences. |
| ComfyUI offline | Start ComfyUI, check **Server Address**, then click the refresh button next to the generate button. |
| Enable online access in preferences | Enable **Edit > Preferences > System > Network > Allow Online Access**. See [Installation](installation.md#allow-online-access). |
| Model Directory Empty | No model for the selected architecture was found on the server. See below. |
| TRELLIS.2 nodes not found | Install the TRELLIS.2 package and restart ComfyUI. See below. |
| Select an image first | Choose an **Input Image** for TRELLIS.2, or switch **Generate From** to **Prompt**. |

While the StableGen tab is open, the add-on checks the server every 10 seconds, so the status updates shortly after ComfyUI starts or stops. The refresh button checks immediately and also reloads the model lists.

## Connection problems

- Confirm that ComfyUI is running and that you can open its web interface at the configured address.
- For a server on another machine, start ComfyUI with `--listen` and allow the port through the firewall. See [Using a remote ComfyUI server](installation.md#using-a-remote-comfyui-server).
- On slow connections or busy servers, raise the timeouts under **Advanced Preferences**. **Generation Heartbeat** only decides how long StableGen waits for progress messages before checking the ComfyUI queue; it does not cancel a generation.

## Models are missing

The **Checkpoint** list is read from different ComfyUI folders depending on the architecture:

| Architecture | Folder |
|---|---|
| SDXL | `models/checkpoints` |
| Qwen Image Edit, Qwen Image 2.1 | `models/diffusion_models` (and `models/unet` for GGUF files) |

After adding models, click the refresh button next to **Checkpoint**. If ComfyUI reports a missing model during generation, the preset expects a file you do not have, usually a ControlNet or LoRA. Run `installer.py` with the matching package, or check the file names in [Manual installation](MANUAL_INSTALLATION.md). Restart ComfyUI after adding custom nodes.

If a ControlNet unit uses the wrong model type, review **ControlNet Model Assignments** in the preferences.

## TRELLIS.2 or PBR nodes are not detected

StableGen looks for specific nodes on the server: `Trellis2ImageToShape` for TRELLIS.2, and `MarigoldModelLoader` and `LoadStableDelightModel` for PBR extraction. If they are missing:

1. Run `installer.py` with package 8 (TRELLIS.2) or packages 9 and 10 (PBR).
2. Restart ComfyUI and check its console for errors while it loads the custom nodes.
3. Click the refresh button in the StableGen panel.

TRELLIS.2 uses compiled CUDA extensions without pre-built wheels for Python 3.13. If its nodes fail to load, run ComfyUI with Python 3.12.

If TRELLIS.2 times out during mesh post-processing, raise **Mesh Generation Timeout** in **Advanced Preferences**.

## TRELLIS.2 stops working after a ComfyUI update

TRELLIS.2 runs in its own isolated Python environment managed by comfy-env, but it still imports ComfyUI's core code. When ComfyUI updates and raises a package requirement, that environment falls behind. The ComfyUI console then shows errors such as `No module named 'comfy_aimdo.storage'` or `cannot import name 'ColorPrimaries' from 'av.video.reformatter'`.

Run the installer's repair mode, then restart ComfyUI:

```
python installer.py <path to ComfyUI> --repair
```

It re-applies StableGen's patches to the TRELLIS.2 node, brings the isolated environment in line with ComfyUI's `requirements.txt`, and checks that ComfyUI core imports inside it. `--check` reports the same without changing anything. Installs made with this installer also re-check automatically on every ComfyUI start.

## ComfyUI crashes or hangs while loading a model

Symptoms include ComfyUI exiting with `Windows fatal exception: access violation` during a model load, `The paging file is too small for this operation to complete`, `aimdo` read errors with Windows error 1450, or a sampler that stops at the first step while the GPU shows full load at low power.

These point to the system running out of memory, not to a broken model. Qwen Image 2.1 with its text encoder, TRELLIS.2 and Blender together need a lot of RAM and page file.

- Restart Windows if it has been running for a long time, and close memory-heavy programs (WSL and Docker reserve several GB).
- Give Windows a larger page file: a fixed 32–64 GB on a fast drive.
- Start ComfyUI with `--reserve-vram 2` (or more) when Blender and other GPU programs are open, so models are offloaded instead of overflowing into shared memory.
- If crashes continue, try `--disable-dynamic-vram` and `--disable-pinned-memory`.

## Out of GPU memory

- Lower **Target megapixels**, or keep **Fit camera resolution** enabled, in **Output & materials**.
- Use fewer cameras.
- For Qwen Image Edit, use a smaller GGUF model or the Nunchaku presets.
- For TRELLIS.2, choose a lower **Mesh generation resolution** or another **Attention Backend**.
- Close other applications that use the GPU.

## The texture is not visible

The projection material is evaluated by Cycles. Switch the viewport to **Rendered** shading. In Solid or Material Preview shading, the texture appears only after you bake it.

Blender 5.0 is not supported. Use Blender 4.2 to 4.5 or Blender 5.1 and later.

## All meshes are textured

**Target objects** defaults to **All Visible**. Set it to **Selected** and select the meshes to texture, or hide the others.

## Poor results

- Start from a built-in preset for your architecture. See [Architectures and presets](architectures-and-presets.md).
- Check camera coverage. Every surface you care about should face at least one camera directly. Try another placement mode or more cameras; see [Cameras](cameras.md).
- Add view prompts such as `rear view` to stop the model from drawing a front on every side.
- In **Generate Sequentially**, the first camera sets the look. Change the [generation order](cameras.md#generation-order) so a characteristic view comes first.
- Adjust ControlNet strength for SDXL. Too low ignores the geometry; too high can flatten the result.
- To fix individual views, select their cameras and use **Regenerate selected views**, or use [Local Edit or Masked Edit](generation-modes.md).

## A generation appears stuck

Click the stop button in the footer and wait for the current ComfyUI request to finish. If StableGen still reports that it is busy, enable **Enable Debug Settings** in the preferences and use **Reset stalled generation** in the **Recovery** group of the **Diagnostics** page.
