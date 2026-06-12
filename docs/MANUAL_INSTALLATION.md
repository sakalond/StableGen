# Manual Installation

If you prefer to set up dependencies manually or need to install specific components not covered by the script, follow these instructions.
**Estimated time for manual setup:** 15-30 minutes (excluding model download times, which can be significant).

**Prerequisites for manual installation:**
* **Git:** Required for cloning ComfyUI custom nodes. Download from [git-scm.com](https://git-scm.com/).
* **Sufficient Disk Space:** AI models are large. Ensure you have adequate free space (10GB to 50GB+).

**Manual Installation Steps:**

**1. Install Required ComfyUI Custom Nodes**

StableGen requires specific custom nodes for ComfyUI to function correctly, particularly for IPAdapter support.

1.  Open your system's terminal or command prompt.
2.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd <YourComfyUIDirectory>/custom_nodes/
    ```
3.  Clone the following repository:
    * **ComfyUI IPAdapter Plus:**
        ```bash
        git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
        ```
4.  **Restart ComfyUI** after installing new custom nodes to ensure they are loaded.

    *(You may find the [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) custom node helpful for managing other custom nodes in the future, though it's not a strict requirement for StableGen's core dependencies listed here.)*

**2. Download and Place Required AI Models for StableGen**

These models are essential for StableGen's core features and presets. Download each file and place it into the specified subdirectory within `<YourComfyUIDirectory>/models/`. **Create directories if they do not exist.** Pay close attention to **renaming instructions** where specified, as they are crucial.

* **a) IPAdapter Model (Core Functionality)**
    * **Directory:** `<YourComfyUIDirectory>/models/ipadapter/`
    * **Filename:** `ip-adapter-plus_sdxl_vit-h.safetensors`
    * **Download URL:** [https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors?download=true](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~850 MB

* **b) IPAdapter CLIP Vision Models (Dependencies for IPAdapter)**
    * **Directory:** `<YourComfyUIDirectory>/models/clip_vision/`
    * **File 1:**
        * **Target Filename:** `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
        * **Download URL:** [https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors)
        * **Action:** Download `model.safetensors` and **rename it** to `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`.
    * **File 2:**
        * **Target Filename:** `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors`
        * **Download URL:** [https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors)
        * **Action:** Download `model.safetensors` and **rename it** to `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors`.
    * **License (for both):** Apache 2.0
    * **Combined Size:** ~6 GB

* **c) SDXL Lighting LoRA (Necessary for Default Presets & Recommended for Speed)**
    * **Directory:** `<YourComfyUIDirectory>/models/loras/`
    * **Filename:** `sdxl_lightning_8step_lora.safetensors`
    * **Download URL:** [https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true](https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true)
    * **License:** OpenRAIL++
    * **Size:** ~400 MB

* **d) ControlNet Model (Depth - Crucial for Presets)**
    * **Directory:** `<YourComfyUIDirectory>/models/controlnet/`
    * **Target Filename:** `controlnet_depth_sdxl.safetensors`
    * **Download URL:** [https://huggingface.co/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true](https://huggingface.co/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true)
    * **Action:** Download `diffusion_pytorch_model.safetensors` and **rename it** to `controlnet_depth_sdxl.safetensors`.
    * **License:** Apache 2.0
    * **Size:** ~2.5 GB

**3. (Optional but Recommended) Additional AI Models**

While the models above are core/hardcoded for some presets, StableGen supports more.

* **Other SDXL Lighting & Hyper-SD LoRAs:**
    * Place in: `<YourComfyUIDirectory>/models/loras/`
    * No renaming needed.
    * **SDXL Lightning 4-step:** [Download Link](https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true) (License: OpenRAIL++)
    * **SDXL Lightning 2-step:** [Download Link](https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_2step_lora.safetensors?download=true) (License: OpenRAIL++)
    * **Hyper-SDXL 8-steps LoRA:** [Download Link](https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-lora.safetensors?download=true) (License: Unknown)
    * **Hyper-SDXL 4-steps LoRA:** [Download Link](https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-4steps-lora.safetensors?download=true) (License: Unknown)
    * **Hyper-SDXL 1-step LoRA:** [Download Link](https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-1step-lora.safetensors?download=true) (License: Unknown)

* **Other ControlNet Models:**
    * Place in: `<YourComfyUIDirectory>/models/controlnet/`
    * **Depth (alternative variant):**
        * Filename: `diffusion_pytorch_model.fp16.safetensors` (No rename needed)
        * URL: [https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true)
        * License: OpenRAIL++
        * Size: ~2.5 GB
    * **Union (Depth/Normal/Canny):**
        * Target Filename: `sdxl_promax.safetensors`
        * URL: [https://huggingface.co/brad-twinkl/controlnet-union-sdxl-1.0-promax/resolve/main/diffusion_pytorch_model.safetensors?download=true](https://huggingface.co/brad-twinkl/controlnet-union-sdxl-1.0-promax/resolve/main/diffusion_pytorch_model.safetensors?download=true)
        * Action: Download and **rename** to `sdxl_promax.safetensors`.
        * License: Apache 2.0
        * Size: ~2.5 GB

**4. Download a Base SDXL Model**

You need at least one main SDXL checkpoint. These are large and are user's choice.

* **Recommended Placement:** `<YourComfyUIDirectory>/models/checkpoints/`
* **Recommended Model (Used throughout development):**
    * **RealVisXL V5.0 (fp16):**
        * URL: [https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors?download=true](https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors?download=true)
        * License: OpenRAIL++
        * Size: ~7 GB
* **Alternative: Standard SDXL Base:**
    * **SDXL Base 1.0:**
        * URL: [https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true)
        * License: OpenRAIL++
        * Size: ~7 GB
* *Note: StableGen supports any SDXL-based model.*

**5. (Optional) FLUX.1 Setup**

FLUX.1 is a newer diffusion architecture and its usage in StableGen is optional.
**Important Considerations for FLUX.1:**
* **License:** The FLUX.1-dev model and its specific ControlNet have a **strict non-commercial license**. You must agree to its terms on Hugging Face before downloading.
* **Hugging Face Account:** Downloading the FLUX.1-dev model typically requires a Hugging Face account and being logged in.
* **IPAdapter:** Not currently supported for the FLUX.1 architecture within StableGen.
* **Size:** FLUX.1 models and their dependencies are very large.

* **a) FLUX.1-dev Model:**
    * **Directory:** `<YourComfyUIDirectory>/models/unet/`
    * **Filename:** `flux1-dev.safetensors`
    * **Download Page:** [https://huggingface.co/black-forest-labs/FLUX.1-dev/](https://huggingface.co/black-forest-labs/FLUX.1-dev/)
    * **License:** FLUX-1-dev Non-Commercial License
    * **Size:** ~24 GB

    **(Optional) You can also download a quantized version (GGUF format) instead for potentially improved performance:**
    * **Directory:** `<YourComfyUIDirectory>/models/unet/`
    * **Filename:** `flux1-dev.gguf`
    * **Download URL:** [https://huggingface.co/city96/FLUX.1-dev-gguf/tree/main](https://huggingface.co/city96/FLUX.1-dev-gguf/tree/main)
    * **License:** FLUX-1-dev Non-Commercial License
    * **Size:** ~5 to 12.7 GB (depending on quantization)

    For GGUF (quantized) model support, you will also need to install the following custom node into ComfyUI:

    1.  Navigate to your ComfyUI custom nodes directory:
        ```bash
        cd <YourComfyUIDirectory>/custom_nodes/
        ```
    2.  Clone the repository:
        ```bash
        git clone https://github.com/city96/ComfyUI-GGUF.git
        ```


* **b) FLUX CLIP Models (Required for FLUX.1-dev):**
    * **Directory:** `<YourComfyUIDirectory>/models/clip/`
    * **File 1:** `t5xxl_fp8_e4m3fn.safetensors` ([URL](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true))
    * **File 2:** `clip_l.safetensors` ([URL](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true))
    * **License (for both CLIP models):** Apache 2.0
    * **Combined Size:** ~5.5 GB

* **c) FLUX VAE Model (Required for FLUX.1-dev):**
    * **Directory:** `<YourComfyUIDirectory>/models/vae/`
    * **Filename:** `ae.safetensors`
    * **Target Filename:** `ae.sft` (Rename is necessary here!!)
    * **Download URL:** [https://huggingface.co/black-forest-labs/FLUX.1-dev/](https://huggingface.co/black-forest-labs/FLUX.1-dev/)
    * **License:** Apache 2.0
    * **Size:** ~2.5 GB

* **d) Guidance (ControlNet and/or FLUX.1-dev Depth LoRA):**
    
   *  FLUX ControlNet (Union - Depth/Canny):
        * **Directory:** `<YourComfyUIDirectory>/models/controlnet/`
        * **Target Filename:** `controlnet_flux1_union_pro.safetensors`
        * **Download URL:** [https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors?download=true](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors?download=true)
        * **Action:** Download and **rename** to `controlnet_flux1_union_pro.safetensors`.
        * **License:** FLUX-1-dev Non-Commercial License
        * **Size:** ~6.5 GB
    *  FLUX.1-dev Depth LoRA (alternative to ControlNet):
        * **Directory:** `<YourComfyUIDirectory>/models/loras/`
        * **Filename:** `flux1_depth_lora.safetensors`
        * **Download URL:** [https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/)
        * **License:** FLUX-1-dev Non-Commercial License
        * **Size:** ~1.2 GB
 
* **e) FLUX.1 IPAdapter Custom Node (Enables IPAdapter for FLUX)**
    * **Note:** This is required to use IPAdapter with the FLUX.1 model.
    1.  Navigate to your ComfyUI custom nodes directory:
        ```bash
        cd <YourComfyUIDirectory>/custom_nodes/
        ```
    2.  Clone the repository:
        ```bash
        git clone https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux.git
        ```
    3.  **Restart ComfyUI** after installation.

* **f) FLUX.1 IPAdapter Model**
    * **Directory:** `<YourComfyUIDirectory>/models/ipadapter-flux/`
    * **Filename:** `ip-adapter.bin`
    * **Download URL:** [https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/resolve/main/ip-adapter.bin?download=true](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/resolve/main/ip-adapter.bin?download=true)
    * **License:** FLUX-1-dev Non-Commercial License
    * **Size:** ~6.5 GB

**6. (Optional) Qwen Image Edit Setup**

StableGen can interface with the experimental Qwen Image Edit 2509 workflow for rapid texture refinements. These components are large and require the ComfyUI GGUF loader.

* **a) Install the ComfyUI GGUF Loader (Custom Node)**
    1. Navigate to your ComfyUI custom nodes directory:
        ```bash
        cd <YourComfyUIDirectory>/custom_nodes/
        ```
    2. Clone the GGUF loader repository:
        ```bash
        git clone https://github.com/city96/ComfyUI-GGUF.git
        ```
    3. Restart ComfyUI to load the new node.

* **b) Download the Qwen Image Edit UNet (GGUF)**
    * **Directory:** `<YourComfyUIDirectory>/models/unet/`
    * **Filename:** `Qwen-Image-Edit-2509-Q3_K_M.gguf`
    * **Download URL:** [https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/resolve/main/Qwen-Image-Edit-2509-Q3_K_M.gguf?download=true](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/resolve/main/Qwen-Image-Edit-2509-Q3_K_M.gguf?download=true)
    * **License:** Apache 2.0
    * **Size:** ~9.5 GB

* **c) Download the Qwen Image VAE**
    * **Directory:** `<YourComfyUIDirectory>/models/vae/`
    * **Filename:** `qwen_image_vae.safetensors`
    * **Download URL:** [https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors?download=true](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~254 MB

* **d) Download the Qwen Text Encoder (FP8)**
    * **Directory:** `<YourComfyUIDirectory>/models/clip/`
    * **Filename:** `qwen_2.5_vl_7b_fp8_scaled.safetensors`
    * **Download URL:** [https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors?download=true](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~9.2 GB

* **e) Install the Core Qwen Lightning LoRA (Required for StableGen Presets)**
    * **Directory:** `<YourComfyUIDirectory>/models/loras/`
    * **Filename:** `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`
    * **Download URL:** [https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors?download=true](https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~850 MB

* **f) (Optional) Additional Qwen Lightning LoRAs**
    * **Directory:** `<YourComfyUIDirectory>/models/loras/`
    * **Qwen Image Edit Lightning 8-Step (bf16):** [https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors?download=true](https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors?download=true)
    * **Qwen Image Lightning 4-Step (bf16):** [https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors?download=true](https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors?download=true)
    * Each LoRA is licensed under Apache 2.0 and is approximately 850 MB.

**7. (Optional) Nunchaku Qwen Setup**

For users with NVIDIA GPUs on Linux or Windows (WSL2 recommended for best performance, though native Windows support is improving), Nunchaku offers a highly optimized inference engine for Qwen models.

* **a) Install Required Custom Nodes**
    1. Navigate to your ComfyUI custom nodes directory:
        ```bash
        cd <YourComfyUIDirectory>/custom_nodes/
        ```
    2. Clone the Nunchaku repository:
        ```bash
        git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git
        ```
    3. Clone the Qwen Image LoRA Loader repository:
        ```bash
        git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git
        ```
    4. Restart ComfyUI.

* **b) Download the Nunchaku Quantized Model**
    * **Directory:** `<YourComfyUIDirectory>/models/diffusion_models/`
    * **Filename:** `svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors`
    * **Download URL:** [https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit-2509/resolve/main/lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors?download=true](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit-2509/resolve/main/lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~12.7 GB

**8. (Optional) TRELLIS.2 - Image/Text to 3D Setup**

Generate 3D meshes from a single image or text prompt. Models auto-download on first use (~2 GB each for shape and texture stages).

> **Note:** The TRELLIS.2 native textured output pipeline uses NVIDIA non-commercial libraries (nvdiffrast). Shape-only and SDXL/Qwen texture generation does not have this restriction.

> [!IMPORTANT]
> **Upgrade Action Required:** If you are upgrading from a previous version of StableGen, you **must manually delete** the old `ComfyUI-TRELLIS2` and `ComfyUI-Geometrypack` directories from your ComfyUI `custom_nodes/` directory first before cloning the new repository.

* **a) Install the ComfyUI-TRELLIS2 Custom Node**
    1.  Navigate to your ComfyUI custom nodes directory:
        ```bash
        cd <YourComfyUIDirectory>/custom_nodes/
        ```
    2.  Clone the repository at the tested commit:
        ```bash
        git clone https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git
        cd ComfyUI-TRELLIS2
        git checkout 6b0b2148f45bbafa0b86bfd25c63602b63a7aae0
        ```
    3.  Install its dependencies (from inside the cloned folder):
        ```bash
        pip install -r requirements.txt
        ```
    4.  Install the comfy-env dependency:
        ```bash
        pip install comfy-env==0.2.0
        ```
    5.  **Restart ComfyUI.**
    * **License:** MIT (node code)
    * **Size:** ~0.1 GB (models auto-download on first use)

* **b) Important: Apply Patches (Recommended)**

    The `installer.py` script automatically applies several critical patches to ComfyUI-TRELLIS2 that fix major VRAM leaks (up to ~16 GB savings) and improve stability. If you install manually, it is **strongly recommended** to run the installer for this step, or apply the patches yourself. Key patches include:
    * Wrapping DinoV3 in `torch.no_grad()` to prevent 5-9 GB GPU memory leak.
    * Unregistering models from comfy-env's registry to prevent ~7 GB VRAM leak.
    * Cleaning up temporary IPC files between generations.
    * Python 3.10 compatibility fix for `Path.is_junction`.

**9. (Optional) PBR Decomposition - Marigold IID Setup**

Decompose color textures into PBR material maps (roughness, metallic, normal, height, albedo). Models auto-download on first use.

* **a) Install the ComfyUI-Marigold Custom Node**
    1.  Navigate to your ComfyUI custom nodes directory:
        ```bash
        cd <YourComfyUIDirectory>/custom_nodes/
        ```
    2.  Clone the repository:
        ```bash
        git clone https://github.com/kijai/ComfyUI-Marigold.git
        ```
    3.  Install its dependencies:
        ```bash
        cd ComfyUI-Marigold
        pip install -r requirements.txt
        pip install "diffusers>=0.28"
        ```
    4.  **(Windows only)** Install triton:
        ```bash
        pip install triton-windows
        ```
    5.  **Restart ComfyUI.**
    * **License:** GPL-3.0
    * **Size:** ~0.01 GB (models auto-download on first use, ~2 GB each)

**10. (Optional) PBR Decomposition - StableDelight Setup**

Adds a specular-free albedo extraction option for PBR decomposition. **Requires Marigold IID (step 9) to also be installed.**

* **a) Install the ComfyUI_StableDelight_ll Custom Node**
    1.  Navigate to your ComfyUI custom nodes directory:
        ```bash
        cd <YourComfyUIDirectory>/custom_nodes/
        ```
    2.  Clone the repository:
        ```bash
        git clone https://github.com/lldacing/ComfyUI_StableDelight_ll.git
        ```
    3.  **Restart ComfyUI.**
    * **License:** Apache 2.0

* **b) Download the StableDelight Model**
    * **Directory:** `<YourComfyUIDirectory>/models/diffusers/Stable-X--yoso-delight-v0-4-base`
    * **Source:** [https://huggingface.co/Stable-X/yoso-delight-v0-4-base](https://huggingface.co/Stable-X/yoso-delight-v0-4-base)
    * You can download the full repo or use:
        ```bash
        pip install huggingface_hub
        python -c "from huggingface_hub import snapshot_download; snapshot_download('Stable-X/yoso-delight-v0-4-base', local_dir='<YourComfyUIDirectory>/models/diffusers/Stable-X--yoso-delight-v0-4-base')"
        ```
    * **License:** Apache 2.0
    * **Size:** ~3.3 GB

**11. (Optional) FLUX.2 Klein Setup (Experimental)**

> **⚠️ Experimental:** Klein is not yet fully stable - there are known issues with geometry guidance. Expect improvements in future updates.

FLUX.2 Klein is a multi-reference image editing architecture from Black Forest Labs. No custom nodes required - all nodes are built into ComfyUI.

* **a) Download the Klein 4B Diffusion Model (FP8)**
    * **Directory:** `<YourComfyUIDirectory>/models/diffusion_models/`
    * **Filename:** `flux-2-klein-base-4b-fp8.safetensors`
    * **Download URL:** [https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors?download=true](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~4.1 GB

* **b) Download the Qwen 3 4B Text Encoder (bf16)**
    * **Directory:** `<YourComfyUIDirectory>/models/text_encoders/`
    * **Filename:** `qwen_3_4b.safetensors`
    * **Download URL:** [https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors?download=true](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~8.1 GB

* **c) Download the FLUX.2 VAE**
    * **Directory:** `<YourComfyUIDirectory>/models/vae/`
    * **Filename:** `flux2-vae.safetensors`
    * **Download URL:** [https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors?download=true](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors?download=true)
    * **License:** Apache 2.0
    * **Size:** ~321 MB

* **Total FLUX.2 Klein download size:** ~12.4 GB
* **Estimated VRAM requirement:** ~13 GB

