# Output files

StableGen saves everything it produces under the **Output Directory** set in the add-on preferences. Each generation run gets its own folder:

```
<Output Directory>/
└── <file name>/
    └── <YYYY-MM-DDTHH-MM-SS>/
```

`<file name>` is the name of the `.blend` file without its extension, or the scene name if the file has not been saved. The timestamp is set when a new generation starts. **Reproject** and **Regenerate selected views** reuse the folder of the run they continue, so their files are added to it.

Folders are created as needed, and empty folders are removed when a generation finishes, so a run folder contains only the folders its workflow used.

## Run folder contents

| Path | Contents |
|---|---|
| `generated/` | The image generated for each camera, before projection. |
| `controlnet/depth/`, `canny/`, `normal/` | Depth, edge and normal guidance renders for each camera. |
| `controlnet/workbench/`, `viewport/` | Workbench and viewport guidance renders, used by Qwen guidance maps. |
| `inpaint/render/` | Renders of the current texture, used as context in sequential, refine and local edit workflows. |
| `inpaint/visibility/` | Visibility and feathering masks. |
| `uv_inpaint/uv_visibility/` | UV-space visibility masks for UV Inpaint. |
| `generated_baked/` | Per-object results of UV Inpaint. |
| `pbr/` | Extracted PBR maps per camera and the settings used to make them. |
| `baked/` | Results of **Bake** (`<object>.png`, `<object>_BaseColor.png` and the other PBR maps, `<object>_ORM.png`), game engine exports, and 3D print exports. |
| `misc/` | Other intermediate renders and temporary upload images. |
| `prompts/` | Every ComfyUI workflow submitted during the run, one JSON file per request. |
| `debug/` | Output of diagnostic tools, such as projection line reports. |
| `prompt.json` | The most recent ComfyUI workflow, useful for inspecting what was sent to ComfyUI. For Qwen Image 2.1, it is an editable workflow that can be dragged into ComfyUI, and the exact API graph is saved beside it as `prompt_api.json`. |
| `view_plan.json` | The cameras, order and prompts used for the run. |
| `orbit.gif`, `orbit.mp4` | Orbit exports. |
| `<file name>_<timestamp>.blend` | A copy of the scene with packed textures, written when **Save Blend File** is enabled in the preferences. |

TRELLIS.2 runs also write the generated mesh as `trellis2_<date>_<time>.glb` and, when generating from a prompt, the input image as `trellis2_input_<date>_<time>.png`. Auto-rigging writes `stablegen_autorig_source_<...>.glb` and `stablegen_autorig_rigged_<...>.glb`.

## Masked Edit runs

[Masked Edit](masked-edit.md) uses a separate folder next to the normal run folders, named `<timestamp>-masked-<id>`. It contains `masked-edit.json` and a `view-NNN/` folder for each edited view with the render, masks, generated image and composite.

## Other locations in the output directory

| Path | Contents |
|---|---|
| `queue_jobs/` | `.blend` snapshots of [Scene Queue](batch-and-queue.md#scene-queue) items. Queue runs produce run folders named after these snapshots. |
| `sg_scene_queue.json` | The saved Scene Queue. |
| `regression/` | Results of the debug texturing regression. |

DAE/GLB batch texturing writes its `.glb` files and `batch_failures.json` to the batch's own output directory, and TRELLIS.2 folder batches write their log to the image folder. See [Batch processing](batch-and-queue.md).
