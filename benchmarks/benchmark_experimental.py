from __future__ import annotations

from functools import partial

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters


@fal.function(
    requirements=[
        "accelerate==0.24.1",
        "diffusers==0.22.0",
        "torch==2.1.0",
        "transformers==4.35.0",
        "xformers==0.0.22.post7",
        "git+https://github.com/openai/consistencydecoder.git@22a0449022f17a2d7bfc69535e8e8f3ff0585ecb",
    ],
    machine_type="GPU",
)
def diffusers_consistency_decoder(
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
) -> BenchmarkResults:
    import torch
    import torch.nn as nn
    from consistencydecoder import ConsistencyDecoder
    from diffusers import DiffusionPipeline

    class ConsistencyDecoderModule(nn.Module):
        def __init__(self, *args, **kwargs):
            self.decoder = ConsistencyDecoder(
                device="cuda",
                download_root="/data/other/consistencydecoder",
            )
            super().__init__(*args, **kwargs)

        def forward(self, *args, **kwargs):
            return self.decoder(*args, **kwargs)

    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.vae.decoder = ConsistencyDecoderModule()
    pipeline.to("cuda")

    inference_func = partial(
        pipeline,
        parameters.prompt,
        num_inference_steps=parameters.steps,
    )
    return benchmark_settings.apply(inference_func)


LOCAL_BENCHMARKS = [
    {
        "name": r"Diffusers (torch 2.1, SDPA) + OpenAI's [consistency decoder](https://github.com/openai/consistencydecoder)\*\*",
        "category": "SD1.5 (End-to-end)",
        "function": diffusers_consistency_decoder,
    },
]
