# Rigging and animation

StableGen can add a skeleton to a character mesh with SkinTokens, running on the ComfyUI server, and retarget FBX or BVH animation onto a rig in Blender.

## Where the tools are

| Tool | Location in the StableGen panel |
|---|---|
| Auto-rig selected character | **Tools** page, **Mesh & rig utilities**. Available with any architecture. |
| Auto-Rig during TRELLIS.2 generation, rig display, reference poses and animation | **Rigging & Animation** section of the **Mesh** page. Shown only when **Architecture** is **TRELLIS.2**. |

In the legacy panel, the same controls are in the **Auto-Rigging (SkinTokens)** section, also only for TRELLIS.2.

## Auto-rigging with SkinTokens

Auto-rigging requires a ComfyUI node named `StableGenSkinTokensAutoRig`. It is not installed by `installer.py`. StableGen checks for it when connecting to the server; until it is found, the auto-rig controls are disabled with the message **StableGen SkinTokens node was not detected in ComfyUI**.

### Rigging an existing mesh

1. Select the character's mesh, or the empty or armature that is the root of an imported character. Multiple selected mesh parts are joined into a temporary copy for rigging.
2. Click **Auto-rig selected character** (**Auto-Rig Selected Character**).
3. Review the options in the dialog and confirm.

StableGen exports the mesh as a GLB, uploads it to ComfyUI, runs SkinTokens and imports the rigged GLB: an armature and a mesh skinned to it with vertex-group weights. **Preserve Original Source** (on by default) hides the original mesh; when off, the original is deleted after a successful rig.

### Rigging during TRELLIS.2 generation

Enable **Auto-Rig** in the **Rigging & Animation** section. The rigging step runs on the server in the same job, after mesh generation and before texturing. Local decimation, local remeshing and **Solidify Mesh** are skipped when auto-rigging is enabled, and local collapse decimation is performed on the server instead.

### Rig options

| Option | Default | Purpose |
|---|---|---|
| Transfer Original | On | Transfer the generated rig back to the original mesh, preserving its texture and scale. |
| Use Existing Skeleton | Off | If the input already has a skeleton, keep it and generate only skin weights. |
| Skin Postprocess | Off | SkinTokens voxel skin post-processing before export. |
| Clean Limb Weights | On | Remove weights that cross between limbs. |
| Garment Handling | Original Weights | **Original Weights**, **Stabilized Dress** or **Smooth Leg Follow**, post-import weighting for long dresses and skirts. |
| Bone Direction | Fortune / Child-Directed | How bone orientation is reconstructed on import: **Fortune / Child-Directed**, **Blender / Legacy Round-Trip** or **Temperance / Nearest Axis**. |
| Bone Display | Octahedral (Mixamo-like) | Viewport display of the bones. **Apply** changes it on existing armatures. |

The sampling options (**Rig candidate limit (top-k)**, **Rig probability cutoff (top-p)**, **Rig sampling temperature**, **Repetition Penalty** and **Rig beam count**) control how SkinTokens generates the skeleton. **Rig checkpoint path** and **Hugging Face model path** override the model; leave them empty to use the defaults.

## Animation retargeting

Retargeting bakes motion from an FBX or BVH file onto a humanoid rig, such as one made by auto-rigging. All commands act on the selected armature or rigged mesh.

1. **Analyze Rig.** Infers which bones form the spine, arms, legs, pelvis and head and stores this mapping on the armature.
2. **Check the mapping.** The eye button next to **Edit Mapping** shows the mapping in the viewport. **Edit Mapping** lets you correct the start and end bones of each chain. **Proxy** shows the collision capsules used to keep limbs out of the body.
3. **Set a reference pose.** The rig's pose should match the pose the animation starts from. Either:
   - pose the rig with **Rest**, **A-Pose** or **T-Pose**, adjust it if needed, and click **Set Ref Pose**; or
   - click **Match Source Ref**, choose the animation file, and let StableGen pose the rig to match the animation's reference frame. Each arm's upper arm, elbow bend, forearm and hand are turned to the source's directions, so the elbows and wrists move like the source's; on a rig without finger bones the hand follows its skinned mesh rather than its bone. **Protect Body Clearance** then moves a forearm or hand that cuts into the body out to the side, only as far as the skinned mesh needs. This stores the reference pose by default.
4. **Apply Animation.** Choose the FBX or BVH file and confirm the options. StableGen imports the source, retargets it and bakes a new action named `<armature>_<file>_<engine>` over the source's frame range.

Main options of **Apply Animation**:

| Option | Default | Purpose |
|---|---|---|
| Retarget Engine | Reference-Aware (Recommended) | **Reference-Aware** maps motion relative to the reference poses. **Legacy Compatibility** keeps the older behavior. |
| Reference Pose | Stored Ref Pose | The target pose matched to the source's neutral pose: **Stored Ref Pose**, **Current Pose** or **Rig Rest Pose**. |
| Source Neutral | First Frame | **First Frame** or **Source Rest Pose** of the animation. |
| Auto-Match First Frame | On | Poses the rig to the source's first frame automatically. |
| Key Every N Frames | 1 | Keyframe interval for the bake, 1 to 10. |
| Skeletal Proxy Collisions | On | Keeps forearms and hands out of the body. On a skinned rig every frame is checked against the mesh and the arms open sideways only where they would cut in, eased over the neighboring frames so a hand moves smoothly around a belt pouch. Collision capsules keep the arms apart and stand in for rigs without a skinned mesh. |
| Replace Existing Action | On | Replace the rig's current action. |
| Copy Root Motion | Off | Transfer root translation. |
| Show Source Preview | Off | Keep a preview of the source animation. The eye button next to **Apply Animation** shows or hides it. |

Further options control hand orientation and roll, arm corrections, collision margins and body clearance; hover over each for details.

If **Reference Pose** is **Stored Ref Pose** and no reference pose has been stored, Apply Animation stops with a message asking you to use **Set Ref Pose**, unless the reference-aware engine can match the first frame automatically.
