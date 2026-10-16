# Generation modes

The **Texture workflow** setting on the **Texture** page decides how StableGen creates or updates textures. (The legacy panel labels it **Generation Mode**.) Presets set it for you; change it when you want a different workflow.

Every workflow works from the scene cameras: StableGen renders guidance images from each camera, sends them to ComfyUI, and projects the returned images back onto the meshes. How the projections are combined is controlled on the **Blending** page. **Target objects** chooses whether all visible meshes or only selected meshes are textured.

The settings specific to the current workflow are in the **Mode settings** section of the **Texture** page.

## SDXL workflows

| Workflow | What it does |
|---|---|
| Generate Separately | Generates each camera's image independently. Earlier views are not used as context. |
| Generate Sequentially | Generates one camera at a time. Each new view inpaints the regions not yet covered by earlier views, using a render of the current texture as context. |
| Generate Using Grid | Generates all views together as a single grid image, then splits it into camera images. Fast, but at a lower resolution per view. |
| Refine/Restyle Texture (Img2Img) | Renders the existing texture from each camera and modifies it with the current prompt. Works on textures not made by StableGen. |
| Local Edit | Generates changes from existing views and blends them over the current projection with feathered edges. |
| Masked Edit | Regenerates only a region you paint on the model. See [Masked Edit](masked-edit.md). |
| UV Inpaint Missing Areas | Bakes the texture and inpaints untextured areas in UV space instead of camera space. |

### Generate Separately

Because views do not see each other, consistency depends on the shared prompt, geometry guidance and, optionally, IPAdapter. Enable **Use previous view as reference** to feed an earlier generated image to IPAdapter; **Reference source** selects the first or the most recent view, the original render, or the TRELLIS.2 input image. The **Architecture** preset uses this workflow with a high blending exponent, so each surface takes its texture from the most direct camera.

### Generate Sequentially

The camera [generation order](cameras.md#generation-order) matters in this workflow: the first view sets the appearance that later views continue. The visibility masks that decide what to inpaint are controlled by **Full-visibility tolerance**, **Coverage low point** and **Coverage high point**, and **Smooth previous-view coverage** produces soft rather than hard mask edges. The **Inpainting Options** section holds the masking settings shared with UV Inpaint, such as **Differential Diffusion**, **Blur Mask** and **Mask expansion (px)**.

**Use previous view as reference** works as in Separately. **Regenerate First Image** regenerates the first view with IPAdapter applied.

### Generate Using Grid

Enable **Refine generated views** to run an image-to-image pass over each view after the grid is split. The refinement has its own sampler, scheduler, **Edit strength**, steps, CFG and an optional **Refine Prompt**.

### Refine/Restyle Texture (Img2Img)

**Edit strength** controls how far each view may move from the current texture. Low values add detail; high values restyle.

### Local Edit

Point one or more cameras at the area to change and describe the change. The new images blend over the existing material instead of replacing it. Blending is controlled by:

| Setting | Effect |
|---|---|
| Edit strength | How much each view may change. |
| Use Angle-Based Blending | Fades contributions on surfaces seen at a grazing angle, between **Angle ramp low point** and **Angle ramp high point**. |
| Image-frame fade | Fades contributions toward the edges of each camera image. |
| Projection silhouette fade | Fades contributions near the silhouette edges of the projection. |
| Match Colors to Viewport | Matches each new view's colors to the current texture before blending. Methods: MKL, Histogram, Reinhard, MVGD, HM–MVGD–HM and HM–MKL–HM. |

### UV Inpaint Missing Areas

UV Inpaint works on baked UV textures. Bake first with your preferred unwrap so that the inpainting does not have to unwrap during generation; the **Uv Inpainting** preset description recommends this. **Allow updating textured areas** lets the model also change areas that already have texture. **Ask for Object Specific Prompts** uses a separate prompt for each object instead of the shared prompt.

## Qwen workflows

With **Qwen Image Edit** or **Qwen Image 2.1**, the workflow menu offers four entries:

| Workflow | What it does |
|---|---|
| Generate | Generates new per-camera images. The Qwen presets generate the views sequentially. |
| Refine | Restyles the existing texture across all views. |
| Local Edit | Changes the areas the cameras point at and blends them over the existing material, as Local Edit above. |
| Masked Edit | Regenerates only a painted region. See [Masked Edit](masked-edit.md). |

Qwen workflows are guided by reference images rather than ControlNet and IPAdapter. They are configured in the **Image guidance** section:

| Setting | Purpose |
|---|---|
| Guidance Map | The structural image for each view: **Depth Map**, **Normal Map**, **Workbench Render** or **Viewport Render**. |
| Use External Image as Style | Use an image file as the style reference. **External for Initial Only** applies it to the first view only. |
| Use previous view as reference | Use the first or most recent generated view as a reference. |
| Context Render | In sequential generation, how the render of the current texture is supplied: **Disabled**, **Replace Style Image** or **Additional Context**. |
| Hard view segmentation | Assigns each surface point to its closest camera during generation. Used by the Voronoi presets. |

In Refine and Local Edit, **Use Previous Refined View** adds the last edited view as a reference and **Use Depth Map** adds the depth map as structural guidance.

Qwen sends instruction prompts to the model in addition to your text. Enable **Custom Guidance Prompts** to edit them; the reset buttons restore the defaults.

## Regenerating selected views

After a generation, select one or more cameras. The generate button changes to **Regenerate selected views**, which regenerates only those views and keeps the others. It requires a previous generation and is available when the workflow is Generate Separately or Generate Sequentially; Qwen **Generate** qualifies because the Qwen presets run it sequentially. Click **All views** in the footer to clear the camera selection.

## Reprojecting

**Reproject** rebuilds the projection from the saved camera images using the current blending settings. It does not generate new images, so it is the quickest way to try blending changes. In the StableGen panel, the **Bake** button is disabled until you reproject after changing a setting that affects projection.

**Mirror last projection** on the **Tools** page duplicates the last projection camera and its image across the chosen axes (**X** by default), which helps complete symmetric objects.
