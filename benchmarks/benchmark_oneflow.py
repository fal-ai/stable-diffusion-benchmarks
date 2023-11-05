from __future__ import annotations

import sys
from functools import partial

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "--pre",
        "torch==2.1.0",
        "transformers==4.27.1",
        "diffusers[torch]==0.19.3",
        "onediff",
        "oneflow",
        "-f",
        "https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu118",
    ],
    machine_type="GPU",
    python_version="3.10",
)
def oneflow_any(
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
    model_name: str,
) -> BenchmarkResults:
    import oneflow as flow
    import torch
    from diffusers import DiffusionPipeline
    from onediff.infer_compiler import oneflow_compile

    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.to("cuda")
    pipeline.unet = oneflow_compile(pipeline.unet)

    with flow.autocast("cuda"):
        infer_func = partial(
            pipeline, parameters.prompt, num_inference_steps=parameters.steps
        )
        return benchmark_settings.apply(infer_func)


# Since OneFlow doesn't support Python>3.10 (which we actively use), we are
# skipping these benchmarks in our continuous integration suite for now. If you
# make a change, be sure to run them manually on your computer and commit the
# results as artifacts.

LOCAL_BENCHMARKS = [
    {
        "name": "OneFlow",
        "category": "SD1.5 (End-to-end)",
        "function": oneflow_any,
        "kwargs": {
            "model_name": "runwayml/stable-diffusion-v1-5",
        },
        "skip_if": sys.version_info > (3, 10),
    },
    {
        "name": "OneFlow",
        "category": "SDXL (End-to-end)",
        "function": oneflow_any,
        "kwargs": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        },
        "skip_if": sys.version_info > (3, 10),
    },
]
