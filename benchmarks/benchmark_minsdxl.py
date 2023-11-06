from __future__ import annotations

import runpy
from functools import partial
from urllib.request import urlretrieve

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "accelerate==0.24.1",
        "diffusers==0.22.0",
        "torch==2.1.0",
        "transformers==4.35.0",
        "xformers==0.0.22.post7",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.3/flash_attn-2.3.3+cu122torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    ],
    machine_type="GPU",
)
def diffusers_any(
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
    model_url: str,
) -> BenchmarkResults:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.modeling_utils import ModelMixin

    file, _ = urlretrieve(model_url, "sdxl_rewrite.py")
    sdxl_rewrite = runpy.run_path(file)

    class UnetRewriteModel(sdxl_rewrite["UNet2DConditionModel"], ModelMixin):  # type: ignore
        pass

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline = pipeline.to("cuda")

    with torch.device("cuda"):
        with torch.cuda.amp.autocast():
            unet_new = UnetRewriteModel().half()
            unet_new.load_state_dict(pipeline.unet.state_dict())

    pipeline.unet = unet_new.eval()
    inference_func = partial(
        pipeline, parameters.prompt, num_inference_steps=parameters.steps
    )
    return benchmark_settings.apply(inference_func)


LOCAL_BENCHMARKS = [
    {
        "name": "[minSDXL](https://github.com/cloneofsimo/minSDXL) (torch 2.1)",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_url": "https://raw.githubusercontent.com/cloneofsimo/minSDXL/504838853cde2736d9d766ec55abe9b481ac7988/sdxl_rewrite.py",
        },
    },
    {
        "name": "[minSDXL+](https://github.com/isidentical/minSDXL) (torch 2.1, SDPA)",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_url": "https://raw.githubusercontent.com/isidentical/minSDXL/4e378780c75399823aa29404b9e1288d96c22943/sdxl_rewrite.py",
        },
    },
    {
        "name": "[minSDXL+](https://github.com/isidentical/minSDXL) (torch 2.1, flash-attention v2)",
        "category": "SDXL (End-to-end)",
        "function": diffusers_any,
        "kwargs": {
            "model_url": "https://raw.githubusercontent.com/isidentical/minSDXL/0fd7fe9c6f6544f7d16eb7a41cd7606cddb9527c/sdxl_rewrite.py",
        },
    },
]
