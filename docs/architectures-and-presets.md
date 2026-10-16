# Architectures and presets

## Architectures

The **Architecture** menu selects the model family. Each architecture needs its own models on the ComfyUI server; see [Installation](installation.md#2-install-comfyui-dependencies).

| Architecture | Checkpoint list comes from | Guidance | Notes |
|---|---|---|---|
| SDXL | `models/checkpoints` | ControlNet (Depth, Canny, Normal) and IPAdapter | Supports every [generation mode](generation-modes.md). |
| Qwen Image Edit | `models/diffusion_models` and GGUF models | Reference images: a guidance map, previous or initial views, or an external style image | Uses the Qwen workflows: Generate, Refine, Local Edit and Masked Edit. |
| Qwen Image 2.1 | `models/diffusion_models`, filtered to Qwen Image 2.1 files | As Qwen Image Edit | The presets prompt for RGBA views; transparent pixels are excluded from projection. Non-commercial use only ([Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)). |
| TRELLIS.2 | — | — | Generates a mesh from an image or prompt, then optionally textures it with one of the other architectures. See [TRELLIS.2](trellis2.md). |

The Qwen architectures use their own set of workflows and do not offer Grid or UV Inpaint. See [Generation modes](generation-modes.md#qwen-workflows).

### Deprecated architectures

**FLUX.1** and **FLUX.2 Klein** no longer receive support. They are hidden from the architecture menus, and their presets are hidden, unless **Enable Deprecated Architectures** is enabled in the add-on preferences. A file saved with a deprecated architecture keeps that selection and remains readable, but you must enable the preference to apply its presets.

## Presets

A preset sets many generation and blending parameters at once. The built-in presets cover the common cases and are the recommended starting point.

In the StableGen panel, the **Preset** menu lists only the presets for the current architecture, and choosing one applies it immediately. In the legacy panel, choose a preset and click **Apply Preset**. Hover over a preset to read its description.

When you change a setting so that it no longer matches any preset, the menu shows **Custom**. Choosing the preset again restores its values.

There are 38 built-in presets. By default, 33 are shown: one is visible only with **Enable Experimental Texturing Features**, and four only with **Enable Deprecated Architectures**.

### SDXL (8)

| Preset | Workflow | Use |
|---|---|---|
| Default | Sequentially | General-purpose texturing. |
| Model Is Important | Sequentially | As Default, with stronger depth ControlNet guidance. |
| Characters | Sequentially | Characters; uses the first view as an IPAdapter reference. |
| Characters (Alternative Masking) | Sequentially | Alternative masking for more consistent views; may produce more artifacts. Try it if Characters fails. |
| Quick Draft | Grid | Fast previews. |
| Uv Inpainting | UV Inpaint | Fills untextured areas in UV space. Bake textures first to control the unwrap. |
| Architecture | Separately | Buildings and flat surfaces; each surface point takes the most direct camera. |
| Local Refine | Local Edit | Adds detail to specific areas using cameras pointed at them. |

### Qwen Image Edit (12)

| Preset | Workflow | Use |
|---|---|---|
| Qwen Precise | Generate | Detailed results when cameras overlap well. |
| Qwen Precise+ | Generate | Qwen Precise with Confidence-Gated Context, Acquisition Quality Weighting and Depth/Part Projection Guard. |
| Qwen Safe | Generate | Uses the first view as a reference for every later view; more consistent when coverage is limited. |
| Qwen Alt | Generate | Combines context renders with sequential references. |
| Qwen Voronoi | Generate | Assigns each surface point to its closest camera during generation, then blends softly. |
| Qwen Precise / Safe / Alt / Voronoi (Nunchaku) | Generate | The same four strategies for the Nunchaku model. Requires the Nunchaku nodes. |
| Local Edit (Qwen) | Local Edit | Targeted changes in areas the cameras point at. |
| Refine (Qwen) | Refine | Restyles the whole existing texture. |
| Qwen 2509 Full Steps (Experimental) | Generate | Base-model comparison without the Lightning LoRA. Shown only with experimental features enabled. |

### Qwen Image 2.1 (6)

| Preset | Workflow | Use |
|---|---|---|
| Qwen 2.1 Precise | Generate | Projected context renders; best with overlapping cameras. |
| Qwen 2.1 Precise+ | Generate | Qwen 2.1 Precise with Confidence-Gated Context, Acquisition Quality Weighting and Depth/Part Projection Guard. |
| Qwen 2.1 Safe | Generate | Uses the first view as an appearance reference for every later view. |
| Qwen 2.1 Context + Reference | Generate | Combines projected context with the first view as a persistent reference. |
| Local Edit (Qwen 2.1) | Local Edit | Targeted changes in areas the cameras point at. |
| Refine (Qwen 2.1) | Refine | Restyles the whole existing texture. |

### TRELLIS.2 Pipeline (8)

| Preset | Texture mode | Use |
|---|---|---|
| Default (Mesh + Texture) | SDXL | General objects. |
| Trellis.2 (Mesh Only) | None | Geometry only. |
| Characters (Mesh + Texture) | SDXL | Characters; does not texture downward-facing surfaces. |
| Architecture (Mesh + Texture) | SDXL | Buildings and objects with flat walls and sharp angles. |
| Qwen Precise (Mesh + Texture) | Qwen Image Edit | Detailed results when cameras overlap well. |
| Qwen Safe (Mesh + Texture) | Qwen Image Edit | Uses the TRELLIS.2 input image as a style reference. |
| Qwen Alt (Mesh + Texture) | Qwen Image Edit | General-purpose Qwen pipeline with additional context renders. |
| Qwen Voronoi (Mesh + Texture) | Qwen Image Edit | Hard camera segmentation during generation, then soft blending. |

### FLUX.2 Klein (Deprecated) (4)

Klein Precise, Klein Safe, Klein Alt and Klein Voronoi follow the same strategies as the Qwen presets of the same names.

## Saving your own presets

1. Adjust the settings. The preset menu shows **Custom**.
2. Click **Save** (StableGen panel) or **Save Preset** (legacy panel).
3. Enter a **Preset Name** and an optional **Description**. **Include ControlNet Units** and **Include LoRA Units** control whether those lists are stored.

The name is stored in upper case. Saved presets appear in the menu alongside the built-in ones, and the delete button removes them. Built-in presets cannot be deleted.

Saved presets are kept in memory only and are lost when Blender is closed.
