from __future__ import annotations

import fal
from fal.toolkit import clone_repository, download_file

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "torch==2.1.0",
        "torchsde",
        "einops",
        "transformers>=4.25.1",
        "safetensors>=0.3.0",
        "aiohttp",
        "accelerate",
        "pyyaml",
        "Pillow",
        "scipy",
        "tqdm",
        "psutil",
        "torchvision",
        "huggingface_hub",
        "accelerate==0.24.1",
        "xformers==0.0.22.post7",
    ],
    machine_type="GPU",
)
def comfy_sdxl(
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
) -> BenchmarkResults:
    import sys

    comfy_repo = clone_repository(
        "https://github.com/comfyanonymous/ComfyUI.git",
        commit_hash="2a23ba0b8c225b59902423ef08db0de39d2ed7e7",
    )
    sys.path.insert(0, str(comfy_repo))

    download_file(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true",
        comfy_repo / "models" / "checkpoints",
        file_name="sd_xl_base_1.0.safetensors",
    )

    import numpy as np
    import torch
    from nodes import (
        CheckpointLoaderSimple,
        CLIPTextEncode,
        EmptyLatentImage,
        KSampler,
        VAEDecode,
    )
    from PIL import Image

    checkpoint_loader_simple = CheckpointLoaderSimple()
    empty_latent_image = EmptyLatentImage()
    clip_text_encode = CLIPTextEncode()
    k_sampler = KSampler()
    vae_decode = VAEDecode()

    (
        model,
        clip,
        vae,
    ) = checkpoint_loader_simple.load_checkpoint(ckpt_name="sd_xl_base_1.0.safetensors")

    @torch.inference_mode
    def inference_func():
        (latent,) = empty_latent_image.generate(width=1024, height=1024, batch_size=1)
        (conditioning,) = clip_text_encode.encode(text="", clip=clip)
        (conditioning_2,) = clip_text_encode.encode(text=parameters.prompt, clip=clip)
        (latent_2,) = k_sampler.sample(
            seed=0,
            steps=parameters.steps,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=model,
            positive=conditioning_2,
            negative=conditioning,
            latent_image=latent,
        )
        (images,) = vae_decode.decode(samples=latent_2, vae=vae)
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    return benchmark_settings.apply(inference_func)


LOCAL_BENCHMARKS = [
    {
        "name": "Comfy (torch 2.1, xformers)",
        "category": "SDXL (End-to-end)",
        "function": comfy_sdxl,
    },
]
