from __future__ import annotations

from functools import partial

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "accelerate==0.24.1",
        "diffusers==0.24.0",
        "torch==2.1.1",
        "transformers>=4.35",
        "xformers>=0.0.22",
        "triton>=2.1.0",
        "https://github.com/chengzeyi/stable-fast/releases/download/v1.0.0/stable_fast-1.0.0+torch211cu121-cp311-cp311-manylinux2014_x86_64.whl",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu121",
    ],
    machine_type="GPU",
)
def stablefast_any(
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
    model_name: str,
) -> BenchmarkResults:
    import torch
    from diffusers import DiffusionPipeline
    from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig, compile

    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.to("cuda")

    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipeline = compile(pipeline, config)

    inference_func = partial(
        pipeline, parameters.prompt, num_inference_steps=parameters.steps
    )
    return benchmark_settings.apply(inference_func)


LOCAL_BENCHMARKS = [
    {
        "name": "Stable Fast (torch 2.1)",
        "category": "SD1.5 (End-to-end)",
        "function": stablefast_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
        },
    },
    {
        "name": "Stable Fast (torch 2.1)",
        "category": "SDXL (End-to-end)",
        "function": stablefast_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        },
    },
]
