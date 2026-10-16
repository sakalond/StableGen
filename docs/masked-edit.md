# Masked Edit

Masked Edit changes only a painted region of an existing texture. It is available
for the SDXL, Qwen Image Edit and Qwen Image 2.1 architectures. See
[Generation modes](generation-modes.md) for the other texture workflows.

In the StableGen panel, choose **Masked Edit** as the **Texture workflow** on the
**Texture** page. The prompt field becomes **Edit instructions**, the painting
commands appear below the workflow selector, and the remaining controls are in the
**Mode settings** section. The main button reads **Edit painted selection**. With
**Use Legacy Panel** enabled in the preferences, the same controls appear in a
separate **Masked Edit** sidebar panel. The controls remain available in Object
Mode and Texture Paint Mode so they can be used while painting the selection.

1. Set **Target objects** (**Paint Targets** in the legacy panel) to **All Visible** for a complete character, or **Selected** for
   selected mesh objects, then click **Paint Selection**. White selects a surface;
   black erases it, and X swaps the colors. The viewport displays the mask
   directly: white is selected, black is unselected. Use Blender's brush controls for brush falloff and the panel's
   size/strength controls. **Clear Mask** removes the selection from every current
   paint target without leaving paint mode. A temporary joined copy lets strokes cross component
   meshes, with a separate mask per original mesh. The original objects remain
   separate and are restored when you click **Finish Painting**. The panel shows
   how many meshes are included. Clear Selection follows the same target scope.
2. Describe the change in **Edit instructions** (**Edit** in the legacy panel), for example “add a stitched badge to the jacket”.
   Avoid instructions to zoom, crop or change the camera.
3. Click **Edit painted selection** (**Generate Masked Edit** in the legacy panel). It follows the normal All/Selected texture
   object scope and uses the currently selected model, sampler, steps and LoRAs.
   StableGen shows the current preparation or ComfyUI sampling stage and a second
   overall bar covering the camera visibility scan and sequential view edits.
4. Inspect the result and choose **Apply** or **Discard**. Apply retains the
   projection layers; normal StableGen baking can flatten them afterwards.
   Discard restores the original mesh and material datablocks.

Each mesh gets its own packed mask image and a separate non-overlapping mask UV
map. Existing texture UVs are preserved. **Mask Resolution** defaults to 4096 and
can be raised to 8192 for close detail. Existing masks are resized to the selected
resolution, while preserving their contents, when **Paint Selection** starts.
Painting and generation currently support meshes using one material
across their faces, with StableGen color graphs, Principled BSDF or emission.
At least one free UV-map slot is required. Paint masks can be reused for later edits.
Painting uses Solid shading with the active mask texture for immediate brush
   feedback. The previous viewport shading and color mode are restored after painting.
   The textured material is temporarily hidden by this mask view; it is not modified.
Ctrl+Z uses a Masked Edit-specific undo boundary while painting. Brush strokes can
be undone normally, but Blender is prevented from undoing only part of the temporary
multi-mesh proxy setup. Reaching the start of the paint session leaves the proxy intact.

## Generation

Existing cameras are ranked by visible selection area weighted by facing angle.
Invisible/tiny selections are skipped. Each view renders the current material
again after previous projections; completed regions become context rather than
being edited repeatedly. Partially covered boundaries can receive further views.
No cameras are created. Regions invisible to every existing camera remain unchanged.

Qwen receives the clean render as image1 and, by default, the same render with a
translucent magenta overlay as image2. **Send Selection Guide to Qwen** can be
disabled for instruction-only editing. With the guide enabled, the driving prompt
first identifies the highlighted region as the content to replace, then inserts the
user's Edit text as the requested result. It also tells Qwen to preserve the camera,
geometry and unselected content and not to reproduce the marker. This explicit
target-first form prevents short phrases such as "make this triangle golden" from
being applied to a similar unselected shape. The user does not need to mention the
marker or its color. The overlay is excluded from the reference latent chain.

Enable **Custom Guidance Prompt** to edit the complete driving prompt sent to
Qwen. Use `{main_prompt}` where the contents of the **Edit** field should be
inserted. The reset button restores the tested request-first template. The custom
template affects Qwen only; SDXL continues to receive the Edit text directly.

**Use Depth Map** renders the current camera's normalized depth and sends it as
Image3. It is disabled by default because depth can improve geometry-sensitive
replacement edits while weakening material changes or small additive details.
Masked Edit suppresses external style and previous-view reference slots for this
request without changing the user's saved options. It does not use Qwen
sampler-level masking.

SDXL receives the current render and an expanded inpainting mask. **Edit strength**
controls denoising. This path uses standard masked VAE inpainting, independently
of the base preset's generation mode, Differential Diffusion and ControlNet settings.

The generated image is projected with a frozen copy of the painted surface mask,
per-view write mask and visibility weights. Edge Softness feathers inward. The
**Expand Selection** control first grows the camera-space write mask outward, then
Edge Softness feathers inward from that expanded boundary. An expansion equal to
or greater than the softness makes the original painted edge receive full weight,
which helps cover jagged edges and traces of the old texture. At zero expansion,
the original painted UV selection remains a strict projection boundary. Positive
expansion intentionally permits projection into the immediately surrounding visible
surface. The generation mask can extend farther for model context. The original material's Principled or
emission shader remains around the edited color, preserving its other inputs.

Changes inside the selection can still be misplaced by the model. Spatial
alignment is not guaranteed by a language instruction. A mismatched output size
is rejected; other semantic/alignment errors require inspecting the preview.

**Stop After Current View** (or Esc) waits for the active backend request to finish
and keeps completed views available for Apply/Discard. Failures leave a recoverable
preview; use Discard to revert. Do not alter camera positions or geometry during a run.

Each run has a separate output directory with per-view clean renders, remaining
masks, generation masks, overlay guides, generated/composited images, workflow
graphs and `masked-edit.json`. The original generation output location is restored
after the run. Image assets used by edit layers are packed into the blend file.
