from functools import partial

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "accelerate==0.24.1",
        "diffusers==0.21.4",
        "torch==2.1.0",
        "transformers==4.35.0",
    ],
    machine_type="GPU",
)
def diffusers_any(
    model_name: str,
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
) -> BenchmarkResults:
    import torch
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.to("cuda")
    return benchmark_settings.apply(
        partial(
            pipeline,
            parameters.prompt,
            num_inference_steps=parameters.steps,
        )
    )


LOCAL_BENCHMARKS = [
    {
        "name": "Diffusers SD1.5",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
        },
    },
    {
        "name": "Diffusers SDXL",
        "function": diffusers_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        },
    },
]
