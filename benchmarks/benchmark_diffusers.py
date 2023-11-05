from __future__ import annotations

import os
from functools import partial

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "accelerate==0.24.1",
        "diffusers==0.21.4",
        "torch==2.1.0",
        "transformers==4.35.0",
        "xformers==0.0.22.post7",
    ],
    machine_type="GPU",
)
def diffusers_any(
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
    model_name: str,
    enable_xformers: bool = False,
    use_compile: bool = False,
    use_nchw_channels: bool = False,
    tiny_vae: str = None,
) -> BenchmarkResults:
    # Some of these functionality might not be available in torch 2.1,
    # but setting just in case if in the future we upgrade to a newer
    # version of torch.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-cache"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

    import torch
    from diffusers import AutoencoderTiny, DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    if tiny_vae:
        pipeline.vae = AutoencoderTiny.from_pretrained(
            tiny_vae,
            torch_dtype=torch.float16,
        )

    pipeline.to("cuda")

    # Use XFormers memory efficient attention instead of Torch SDPA
    # which might also utilize memory efficient attention (alongside
    # flash attention).
    if enable_xformers:
        pipeline.enable_xformers_memory_efficient_attention()

    if use_nchw_channels:
        pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)

    # The mode here is reduce-overhead, which is a balanced compromise between
    # compilation time and runtime. The other modes might be a possible choice
    # for future benchmarks.
    if use_compile:
        pipeline.unet = torch.compile(
            pipeline.unet, fullgraph=True, mode="reduce-overhead"
        )

    inference_func = partial(
        pipeline, parameters.prompt, num_inference_steps=parameters.steps
    )
    return benchmark_settings.apply(inference_func)


LOCAL_BENCHMARKS = [
    {
        "name": "Diffusers (torch 2.1, SDPA)",
        "category": "SD1.5 (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
        },
    },
    {
        "name": r"Diffusers (torch 2.1, SDPA, [tiny VAE](https://github.com/madebyollin/taesd))\*",
        "category": "SD1.5 (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "tiny_vae": "madebyollin/taesd",
        },
    },
    {
        "name": "Diffusers (torch 2.1, xformers)",
        "category": "SD1.5 (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "enable_xformers": True,
        },
    },
    {
        "name": "Diffusers (torch 2.1, SDPA, compiled)",
        "category": "SD1.5 (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "use_compile": True,
        },
    },
    {
        "name": "Diffusers (torch 2.1, SDPA, compiled, NCHW channels last)",
        "category": "SD1.5 (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "use_compile": True,
            "use_nchw_channels": True,
        },
    },
    {
        "name": "Diffusers (torch 2.1, SDPA)",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        },
    },
    {
        "name": r"Diffusers (torch 2.1, SDPA, [tiny VAE](https://github.com/madebyollin/taesd))\*",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "tiny_vae": "madebyollin/taesdxl",
        },
    },
    {
        "name": "Diffusers (torch 2.1, xformers)",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "enable_xformers": True,
        },
    },
    {
        "name": "Diffusers (torch 2.1, SDPA, compiled)",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_compile": True,
        },
    },
    {
        "name": "Diffusers (torch 2.1, SDPA, compiled, NCHW channels last)",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_compile": True,
            "use_nchw_channels": True,
        },
    },
]
