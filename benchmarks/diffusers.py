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
) -> BenchmarkResults:
    # Some of these functionality might not be available in torch 2.1,
    # but setting just in case if in the future we upgrade to a newer
    # version of torch.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-cache"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

    import torch
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
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

    return benchmark_settings.apply(
        partial(
            pipeline,
            parameters.prompt,
            num_inference_steps=parameters.steps,
        )
    )


LOCAL_BENCHMARKS = [
    {
        "name": "Diffusers (fp16, SDPA)",
        "category": "SD1.5",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
        },
    },
    {
        "name": "Diffusers (fp16, xformers)",
        "category": "SD1.5",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "enable_xformers": True,
        },
    },
    {
        "name": "Diffusers (fp16, SDPA, compiled)",
        "category": "SD1.5",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "use_compile": True,
        },
    },
    {
        "name": "Diffusers (fp16, SDPA, compiled, NCHW channels last)",
        "category": "SD1.5",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "use_compile": True,
            "use_nchw_channels": True,
        },
    },
    {
        "name": "Diffusers (fp16, SDPA)",
        "category": "SDXL",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        },
    },
    {
        "name": "Diffusers (fp16, xformers)",
        "category": "SDXL",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "enable_xformers": True,
        },
    },
    {
        "name": "Diffusers (fp16, SDPA, compiled)",
        "category": "SDXL",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_compile": True,
        },
    },
    {
        "name": "Diffusers (fp16, SDPA, compiled, NCHW channels last)",
        "category": "SDXL",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_compile": True,
            "use_nchw_channels": True,
        },
    },
]
