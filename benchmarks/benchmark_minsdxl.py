from __future__ import annotations

import runpy
from functools import partial
from urllib.request import urlretrieve

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "accelerate==0.24.1",
        "diffusers==0.21.4",
        "torch==2.1.0",
        "transformers==4.35.0",
        "xformers==0.0.22.post7",
        "https://github.com/tridao/flash-attention-wheels/releases/download/v2.0.6.post8/flash_attn_wheels_test-2.0.6.post8+cu121torch2.1cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    ],
    machine_type="GPU",
    _scheduler="nomad",
    _scheduler_options={"target_node": "65.21.219.34"},
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
]
