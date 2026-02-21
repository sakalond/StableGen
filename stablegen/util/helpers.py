import random

# Prompt for ComfyUI in API format (SDXL)
prompt_text = """
        {
            "5": {
                "inputs": {
                    "image": "",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "Load Image"
                }
            },
            "6": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "Load Checkpoint"
                }
            },
            "9": {
                "inputs": {
                    "text": "a monkey",
                    "clip": [
                        "247",
                        0
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "10": {
                "inputs": {
                    "text": "",
                    "clip": [
                        "247",
                        0
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "15": {
                "inputs": {
                    "seed": 24,
                    "steps": 8,
                    "cfg": 1.5,
                    "sampler_name": "dpmpp_2s_ancestral",
                    "scheduler": "sgm_uniform",
                    "denoise": 1,
                    "model": [
                        "26",
                        0
                    ],
                    "positive": [
                        "14",
                        0
                    ],
                    "negative": [
                        "14",
                        1
                    ],
                    "latent_image": [
                        "16",
                        0
                    ]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            },
            "16": {
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "19": {
                "inputs": {
                    "samples": [
                        "15",
                        0
                    ],
                    "vae": [
                        "6",
                        2
                    ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "25": {
                "inputs": {
                    "images": [
                        "19",
                        0
                    ]
                },
                "class_type": "SaveImageWebsocket",
                "_meta": {
                    "title": "SaveImageWebsocket"
                }
            },
            "27": {
                "inputs": {
                    "image": "",
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "Load Image"
                }
            },
            "239": {
                "inputs": {
                    "type": "depth",
                    "control_net": [
                        "201",
                        0
                    ]
                },
                "class_type": "SetUnionControlNetType",
                "_meta": {
                    "title": "SetUnionControlNetType"
                }
            },
            "247": {
                "inputs": {
                  "stop_at_clip_layer": -1,
                  "clip": [
                    "",
                    1
                  ]
                },
                "class_type": "CLIPSetLastLayer",
                "_meta": {
                  "title": "CLIP Set Last Layer"
                }
            },
            "235": {
              "inputs": {
                "preset": "PLUS (high strength)",
                "model": [
                  "",
                  0
                ]
              },
              "class_type": "IPAdapterUnifiedLoader",
              "_meta": {
                "title": "IPAdapter Unified Loader"
              }
            },
            "236": {
              "inputs": {
                "weight": 1,
                "start_at": 0,
                "end_at": 1,
                "weight_type": "standard",
                "model": [
                  "235",
                  0
                ],
                "ipadapter": [
                  "235",
                  1
                ],
                "image": ["237", 0]
              },
              "class_type": "IPAdapter",
              "_meta": {
                "title": "IPAdapter"
              }
            },
            "237": {
              "inputs": {
                "upload": "image"
              },
              "class_type": "LoadImage",
              "_meta": {
                "title": "Load Image"
              }
            }
        }
        """

prompt_text_img2img = """
{
  "1": {
      "inputs": {
        "image": "",
        "upload": "image"
      },
      "class_type": "LoadImage",
      "_meta": {
        "title": "Load Image"
      }
    },
  "12": {
      "inputs": {
        "image": "",
        "upload": "image"
      },
      "class_type": "LoadImage",
      "_meta": {
        "title": "Load Image"
      }
    },
  "23": {
      "inputs": {
        "upscale_method": "nearest-exact",
        "width": 1024,
        "height": 1024,
        "crop": "disabled",
        "image": [
          "1",
          0
        ]
      },
      "class_type": "ImageScale",
      "_meta": {
        "title": "Upscale Image"
      }
    },
  "13": {
    "inputs": {
      "grow_mask_by": 0,
      "pixels": [
        "23",
        0
      ],
      "vae": [
        "38",
        2
      ],
      "mask": [
        "25",
        0
      ]
    },
    "class_type": "VAEEncodeForInpaint",
    "_meta": {
      "title": "VAE Encode (for Inpainting)"
    }
  },
  "25": {
    "inputs": {
      "channel": "red",
      "image": [
        "12",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "38": {
    "inputs": {
      "ckpt_name": "realvisxlV40_v40Bakedvae.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "102": {
    "inputs": {
      "text": "tifa lockhart, cosplay, final fantasy",
      "clip": [
        "247",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "103": {
    "inputs": {
      "text": "",
      "clip": [
        "247",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "105": {
    "inputs": {
      "seed": 42,
      "steps": 8,
      "cfg": 1.48,
      "sampler_name": "dpmpp_sde",
      "scheduler": "sgm_uniform",
      "denoise": 0.5,
      "model": [
        "",
        0
      ],
      "positive": [
        "104",
        0
      ],
      "negative": [
        "104",
        1
      ],
      "latent_image": [
        "116",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "108": {
    "inputs": {
      "image": "",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "110": {
    "inputs": {
      "samples": [
        "105",
        0
      ],
      "vae": [
        "38",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "111": {
    "inputs": {
      "images": [
        "110",
        0
      ]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": {
        "title": "SaveImageWebsocket"
    }
  },
  "116": {
    "inputs": {
      "pixels": [
        "118",
        0
      ],
      "vae": [
        "38",
        2
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "117": {
    "inputs": {
      "image": "",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "118": {
    "inputs": {
      "upscale_method": "lanczos",
      "width": 713,
      "height": 1080,
      "crop": "disabled",
      "image": [
        "117",
        0
      ]
    },
    "class_type": "ImageScale",
    "_meta": {
      "title": "Upscale Image"
    }
  },
  "224": {
    "inputs": {
      "expand": 0,
      "tapered_corners": true,
      "mask": [
        "25",
        0
      ]
    },
    "class_type": "GrowMask",
    "_meta": {
      "title": "GrowMask"
    }
  },
  "225": {
    "inputs": {
      "mask": [
        "224",
        0
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "226": {
    "inputs": {
      "blur_radius": 1,
      "sigma": 1,
      "image": [
        "225",
        0
      ]
    },
    "class_type": "ImageBlur",
    "_meta": {
      "title": "Image Blur"
    }
  },
  "227": {
    "inputs": {
      "channel": "red",
      "image": [
        "226",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "228": {
    "inputs": {
      "noise_mask": true,
      "positive": [
        "102",
        0
      ],
      "negative": [
        "103",
        0
      ],
      "vae": [
        "38",
        2
      ],
      "pixels": [
        "1",
        0
      ],
      "mask": [
        "227",
        0
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "229": {
    "inputs": {
      "model": [
        "",
        0
      ]
    },
    "class_type": "DifferentialDiffusion",
    "_meta": {
      "title": "Differential Diffusion"
    }
  },
  "235": {
    "inputs": {
      "preset": "PLUS (high strength)",
      "model": [
        "",
        0
      ]
    },
    "class_type": "IPAdapterUnifiedLoader",
    "_meta": {
      "title": "IPAdapter Unified Loader"
    }
  },
  "236": {
    "inputs": {
      "weight": 1,
      "start_at": 0,
      "end_at": 1,
      "weight_type": "standard",
      "model": [
        "235",
        0
      ],
      "ipadapter": [
        "235",
        1
      ],
      "image": ["1", 0]
    },
    "class_type": "IPAdapter",
    "_meta": {
      "title": "IPAdapter"
    }
  },
  "237": {
    "inputs": {
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "239": {
    "inputs": {
      "type": "depth",
      "control_net": [
        "201",
        0
      ]
    },
    "class_type": "SetUnionControlNetType",
    "_meta": {
      "title": "SetUnionControlNetType"
    }
  },
  "247": {
    "inputs": {
      "stop_at_clip_layer": -1,
      "clip": [
        "",
        1
      ]
    },
    "class_type": "CLIPSetLastLayer",
    "_meta": {
      "title": "CLIP Set Last Layer"
    }
  }
}
"""

prompt_text_flux = """ {
  "6": {
    "inputs": {
      "text": "a monkey",
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "10": {
    "inputs": {
      "vae_name": "ae.sft"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-dev.sft",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "13": {
    "inputs": {
      "noise": [
        "25",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "17",
        0
      ],
      "latent_image": [
        "30",
        0
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "simple",
      "steps": 21,
      "denoise": 1,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "12",
        0
      ],
      "conditioning": [
        "26",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "25": {
    "inputs": {
      "noise_seed": 42
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "26": {
    "inputs": {
      "guidance": 40,
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "30": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "32": {
    "inputs": {
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": {
      "title": "SaveImageWebsocket"
    }
  },
  "239": {
    "inputs": {
      "type": "",
      "control_net": [
        "201",
        0
      ]
    },
    "class_type": "SetUnionControlNetType",
    "_meta": {
      "title": "SetUnionControlNetType"
    }
  }
}
"""

prompt_text_img2img_flux = """ {
  "1": {
        "inputs": {
          "image": "",
          "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {
          "title": "Load Image"
        }
      },
  "6": {
    "inputs": {
      "text": "a monkey",
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "10": {
    "inputs": {
      "vae_name": "ae.sft"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-dev.sft",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "13": {
    "inputs": {
      "noise": [
        "25",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "17",
        0
      ],
      "latent_image": [
        "116",
        0
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "simple",
      "steps": 21,
      "denoise": 1,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "12",
        0
      ],
      "conditioning": [
        "26",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "25": {
    "inputs": {
      "noise_seed": 42
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "26": {
    "inputs": {
      "guidance": 40,
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "30": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "32": {
    "inputs": {
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": {
      "title": "SaveImageWebsocket"
    }
  },
  "42": {
      "inputs": {
        "image": "",
        "upload": "image"
      },
      "class_type": "LoadImage",
      "_meta": {
        "title": "Load Image"
      }
    },
  "43": {
      "inputs": {
        "upscale_method": "nearest-exact",
        "width": 1024,
        "height": 1024,
        "crop": "disabled",
        "image": [
          "1",
          0
        ]
      },
      "class_type": "ImageScale",
      "_meta": {
        "title": "Upscale Image"
      }
    },
  "44": {
    "inputs": {
      "grow_mask_by": 0,
      "pixels": [
        "43",
        0
      ],
      "vae": [
        "10",
        0
      ],
      "mask": [
        "45",
        0
      ]
    },
    "class_type": "VAEEncodeForInpaint",
    "_meta": {
      "title": "VAE Encode (for Inpainting)"
    }
  },
  "45": {
    "inputs": {
      "channel": "red",
      "image": [
        "42",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "50": {
    "inputs": {
      "model": [
        "12",
        0
      ]
    },
    "class_type": "DifferentialDiffusion",
    "_meta": {
      "title": "Differential Diffusion"
    }
  },
  "51": {
    "inputs": {
      "noise_mask": true,
      "positive": [
        "6",
        0
      ],
      "negative": [
        "6",
        0
      ],
      "vae": [
        "10",
        0
      ],
      "pixels": [
        "1",
        0
      ],
      "mask": [
        "227",
        0
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "116": {
    "inputs": {
      "pixels": [
        "118",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "117": {
    "inputs": {
      "image": "",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "118": {
    "inputs": {
      "upscale_method": "lanczos",
      "width": 713,
      "height": 1080,
      "crop": "disabled",
      "image": [
        "117",
        0
      ]
    },
    "class_type": "ImageScale",
    "_meta": {
      "title": "Upscale Image"
    }
  },
  "224": {
    "inputs": {
      "expand": 0,
      "tapered_corners": true,
      "mask": [
        "45",
        0
      ]
    },
    "class_type": "GrowMask",
    "_meta": {
      "title": "GrowMask"
    }
  },
  "225": {
    "inputs": {
      "mask": [
        "224",
        0
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "226": {
    "inputs": {
      "blur_radius": 1,
      "sigma": 1,
      "image": [
        "225",
        0
      ]
    },
    "class_type": "ImageBlur",
    "_meta": {
      "title": "Image Blur"
    }
  },
  "227": {
    "inputs": {
      "channel": "red",
      "image": [
        "226",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "239": {
    "inputs": {
      "type": "",
      "control_net": [
        "201",
        0
      ]
    },
    "class_type": "SetUnionControlNetType",
    "_meta": {
      "title": "SetUnionControlNetType"
    }
  }
}
"""

ipadapter_flux = """
{
  "242": {
      "inputs": {
        "ipadapter": "ip-adapter.bin",
        "clip_vision": "google/siglip-so400m-patch14-384",
        "provider": "cuda"
      },
      "class_type": "IPAdapterFluxLoader",
      "_meta": {
        "title": "Load IPAdapter Flux Model"
      }
    },
  "243": {
    "inputs": {
      "weight": 1,
      "start_percent": 0,
      "end_percent": 1,
      "model": [
        "12",
        0
      ],
      "ipadapter_flux": [
        "242",
        0
      ],
      "image": [
        "244",
        0
      ]
    },
    "class_type": "ApplyIPAdapterFlux",
    "_meta": {
      "title": "Apply IPAdapter Flux Model"
    }
  },
  "244": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  }
}
"""

depth_lora_flux = """
{
  "245": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "246": {
    "inputs": {
      "positive": [
        "6",
        0
      ],
      "negative": [
        "6",
        0
      ],
      "vae": [
        "10",
        0
      ],
      "pixels": [
        "245",
        0
      ]
    },
    "class_type": "InstructPixToPixConditioning",
    "_meta": {
      "title": "InstructPixToPixConditioning"
    }
  },
  "247": {
    "inputs": {
      "lora_name": "flux1-depth-dev-lora.safetensors",
      "strength_model": 1,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  }
}
"""

gguf_unet_loader = """
{
  "12": {
    "inputs": {
      "unet_name": ""
    },
    "class_type": "UnetLoaderGGUF",
    "_meta": {
      "title": "Unet Loader (GGUF)"
    }
  }
}
"""

prompt_text_qwen_image_edit = """
{
  "1": {
    "inputs": {
      "seed": 157165768450326,
      "steps": 4,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1,
      "model": [
        "7",
        0
      ],
      "positive": [
        "12",
        0
      ],
      "negative": [
        "11",
        0
      ],
      "latent_image": [
        "8",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "2": {
    "inputs": {
      "samples": [
        "1",
        0
      ],
      "vae": [
        "4",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "3": {
    "inputs": {
      "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
      "type": "qwen_image",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "Load CLIP"
    }
  },
  "4": {
    "inputs": {
      "vae_name": "qwen_image_vae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "5": {
    "inputs": {
      "images": [
        "2",
        0
      ]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": {
      "title": "SaveImageWebsocket"
    }
  },
  "6": {
    "inputs": {
      "shift": 3,
      "model": [
        "9",
        0
      ]
    },
    "class_type": "ModelSamplingAuraFlow",
    "_meta": {
      "title": "ModelSamplingAuraFlow"
    }
  },
  "7": {
    "inputs": {
      "strength": 1,
      "model": [
        "6",
        0
      ]
    },
    "class_type": "CFGNorm",
    "_meta": {
      "title": "CFGNorm"
    }
  },
  "8": {
    "inputs": {
      "pixels": [
        "14",
        0
      ],
      "vae": [
        "4",
        0
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  },
  "11": {
    "inputs": {
      "prompt": "",
      "clip": [
        "3",
        0
      ],
      "image1": [
        "14",
        0
      ],
      "image2": [
        "15",
        0
      ],
      "image3": [
        "16",
        0
      ]
    },
    "class_type": "TextEncodeQwenImageEditPlus",
    "_meta": {
      "title": "TextEncodeQwenImageEditPlus"
    }
  },
  "12": {
    "inputs": {
      "prompt": "a monkey",
      "clip": [
        "3",
        0
      ],
      "image1": [
        "14",
        0
      ],
      "image2": [
        "15",
        0
      ],
      "image3": [
        "16",
        0
      ]
    },
    "class_type": "TextEncodeQwenImageEditPlus",
    "_meta": {
      "title": "TextEncodeQwenImageEditPlus"
    }
  },
  "13": {
    "inputs": {
      "unet_name": "Qwen-Image-Edit-2509-Q3_K_M.gguf"
    },
    "class_type": "UnetLoaderGGUF",
    "_meta": {
      "title": "Unet Loader (GGUF)"
    }
  },
  "14": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "15": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "16": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  }
}
"""

# ---------------------------------------------------------------------------
# FLUX.2 Klein – multi-reference edit model (Apache 2.0)
# Based on official ComfyUI Klein image-edit workflow.
# Nodes: all built-in ComfyUI core (no custom nodes needed)
#   UNETLoader, CLIPLoader (type=flux2, Qwen 3 4B), VAELoader,
#   CLIPTextEncode (x2), EmptyFlux2LatentImage, Flux2Scheduler,
#   KSamplerSelect, RandomNoise, CFGGuider (cfg=1), SamplerCustomAdvanced,
#   VAEDecode, SaveImageWebsocket, LoadImage (x3 reference slots).
# Reference images are wired dynamically via:
#   VAEEncode → ReferenceLatent (pos + neg chain) → CFGGuider.
# ---------------------------------------------------------------------------
prompt_text_flux2_klein = """
{
  "1": {
    "inputs": {
      "unet_name": "flux-2-klein-base-4b-fp8.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": { "title": "Load Diffusion Model (Klein)" }
  },
  "2": {
    "inputs": {
      "clip_name": "qwen_3_4b.safetensors",
      "type": "flux2"
    },
    "class_type": "CLIPLoader",
    "_meta": { "title": "Load CLIP (Qwen 3 4B)" }
  },
  "3": {
    "inputs": {
      "vae_name": "flux2-vae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": { "title": "Load VAE (FLUX.2)" }
  },
  "4": {
    "inputs": {
      "text": "",
      "clip": ["2", 0]
    },
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "CLIP Text Encode (Positive)" }
  },
  "5": {
    "inputs": {
      "text": "",
      "clip": ["2", 0]
    },
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "CLIP Text Encode (Negative)" }
  },
  "6": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyFlux2LatentImage",
    "_meta": { "title": "Empty Flux 2 Latent" }
  },
  "7": {
    "inputs": {
      "steps": 20,
      "width": 1024,
      "height": 1024
    },
    "class_type": "Flux2Scheduler",
    "_meta": { "title": "Flux 2 Scheduler" }
  },
  "8": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": { "title": "KSampler Select" }
  },
  "9": {
    "inputs": {
      "noise_seed": 0
    },
    "class_type": "RandomNoise",
    "_meta": { "title": "Random Noise" }
  },
  "10": {
    "inputs": {
      "model": ["1", 0],
      "positive": ["4", 0],
      "negative": ["5", 0],
      "cfg": 1
    },
    "class_type": "CFGGuider",
    "_meta": { "title": "CFG Guider" }
  },
  "11": {
    "inputs": {
      "noise": ["9", 0],
      "guider": ["10", 0],
      "sampler": ["8", 0],
      "sigmas": ["7", 0],
      "latent_image": ["6", 0]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": { "title": "Sampler Custom Advanced" }
  },
  "12": {
    "inputs": {
      "samples": ["11", 0],
      "vae": ["3", 0]
    },
    "class_type": "VAEDecode",
    "_meta": { "title": "VAE Decode" }
  },
  "13": {
    "inputs": {
      "images": ["12", 0]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": { "title": "SaveImageWebsocket" }
  },
  "14": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": { "title": "Load Reference Image 1 (Structure)" }
  },
  "15": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": { "title": "Load Reference Image 2 (Style)" }
  },
  "16": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": { "title": "Load Reference Image 3 (Context)" }
  }
}
"""

# Workflow for TRELLIS.2 Image-to-3D generation via PozzettiAndrea/ComfyUI-TRELLIS2 nodes
# Pipeline: LoadImage → LoadModels → RemoveBackground → GetConditioning → ImageToShape → ShapeToTexturedMesh → ExportGLB
prompt_text_trellis2 = """
{
  "1": {
    "inputs": {
      "image": "",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Input Image"
    }
  },
  "2": {
    "inputs": {
      "resolution": "1024_cascade",
      "attn_backend": "flash_attn",
      "vram_mode": "disk_offload"
    },
    "class_type": "LoadTrellis2Models",
    "_meta": {
      "title": "Load TRELLIS.2 Models"
    }
  },
  "3": {
    "inputs": {
      "image": ["1", 0],
      "low_vram": true
    },
    "class_type": "Trellis2RemoveBackground",
    "_meta": {
      "title": "Remove Background"
    }
  },
  "4": {
    "inputs": {
      "model_config": ["2", 0],
      "image": ["3", 0],
      "mask": ["3", 1],
      "include_1024": true,
      "background_color": "black"
    },
    "class_type": "Trellis2GetConditioning",
    "_meta": {
      "title": "Get Conditioning"
    }
  },
  "5": {
    "inputs": {
      "model_config": ["2", 0],
      "conditioning": ["4", 0],
      "seed": 0,
      "ss_guidance_strength": 7.5,
      "ss_sampling_steps": 12,
      "shape_guidance_strength": 7.5,
      "shape_sampling_steps": 12,
      "max_tokens": 49152
    },
    "class_type": "Trellis2ImageToShape",
    "_meta": {
      "title": "Image to Shape"
    }
  },
  "6": {
    "inputs": {
      "model_config": ["2", 0],
      "conditioning": ["4", 0],
      "shape_result": ["5", 0],
      "seed": 0,
      "tex_guidance_strength": 7.5,
      "tex_sampling_steps": 12
    },
    "class_type": "Trellis2ShapeToTexturedMesh",
    "_meta": {
      "title": "Shape to Textured Mesh"
    }
  },
  "7": {
    "inputs": {
      "voxelgrid_path": ["6", 1],
      "decimation_target": 100000,
      "texture_size": 4096,
      "remesh": true,
      "filename_prefix": "trellis2"
    },
    "class_type": "Trellis2ExportGLB",
    "_meta": {
      "title": "Export GLB"
    }
  }
}
"""

prompt_text_trellis2_shape_only = """
{
  "1": {
    "inputs": {
      "image": "",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Input Image"
    }
  },
  "2": {
    "inputs": {
      "resolution": "1024_cascade",
      "attn_backend": "flash_attn",
      "vram_mode": "disk_offload"
    },
    "class_type": "LoadTrellis2Models",
    "_meta": {
      "title": "Load TRELLIS.2 Models"
    }
  },
  "3": {
    "inputs": {
      "image": ["1", 0],
      "low_vram": true
    },
    "class_type": "Trellis2RemoveBackground",
    "_meta": {
      "title": "Remove Background"
    }
  },
  "4": {
    "inputs": {
      "model_config": ["2", 0],
      "image": ["3", 0],
      "mask": ["3", 1],
      "include_1024": true,
      "background_color": "black"
    },
    "class_type": "Trellis2GetConditioning",
    "_meta": {
      "title": "Get Conditioning"
    }
  },
  "5": {
    "inputs": {
      "model_config": ["2", 0],
      "conditioning": ["4", 0],
      "seed": 0,
      "ss_guidance_strength": 7.5,
      "ss_sampling_steps": 12,
      "shape_guidance_strength": 7.5,
      "shape_sampling_steps": 12,
      "max_tokens": 49152
    },
    "class_type": "Trellis2ImageToShape",
    "_meta": {
      "title": "Image to Shape"
    }
  },
  "6": {
    "inputs": {
      "trimesh": ["5", 1],
      "target_face_count": 100000,
      "fill_holes": true,
      "remesh": false,
      "remesh_band": 1.0
    },
    "class_type": "Trellis2Simplify",
    "_meta": {
      "title": "Simplify Mesh"
    }
  },
  "7": {
    "inputs": {
      "trimesh": ["6", 0],
      "filename_prefix": "trellis2",
      "file_format": "glb"
    },
    "class_type": "Trellis2ExportTrimesh",
    "_meta": {
      "title": "Export Trimesh"
    }
  }
}
"""

