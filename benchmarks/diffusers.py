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
) -> BenchmarkResults:
    import torch
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.to("cuda")
    if enable_xformers:
        pipeline.enable_xformers_memory_efficient_attention()

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
]
