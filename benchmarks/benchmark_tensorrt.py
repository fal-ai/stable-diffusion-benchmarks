from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import fal

from benchmarks.settings import BenchmarkResults, BenchmarkSettings, InputParameters

DATA_DIR = Path("/data/tensorrt")
REPO_DIR = DATA_DIR / "repo"


def prepare_tensorrt() -> Path:
    DATA_DIR.mkdir(exist_ok=True)

    if not REPO_DIR.exists():
        try:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "https://github.com/rajeevsrao/TensorRT",
                    "--branch",
                    "release/9.0",
                    "--single-branch",
                    str(REPO_DIR),
                ]
            )
        except subprocess.CalledProcessError:
            print("Failed to clone TensorRT repo")
            shutil.rmtree(REPO_DIR)
            raise

    return REPO_DIR


@fal.function(
    # Copied from https://github.com/rajeevsrao/TensorRT/blob/release/9.0/demo/Diffusion/requirements.txt
    requirements=[
        "--pre",
        "accelerate==0.24.1",
        "colored",
        "controlnet_aux==0.0.6",
        "cuda-python",
        "diffusers==0.19.3",
        "ftfy",
        "matplotlib",
        "nvtx",
        "onnx-graphsurgeon",
        "onnx==1.14.0",
        "onnxruntime==1.15.1",
        "polygraphy==0.47.1",
        "scipy",
        "tensorrt==9.0.1.post12.dev4",
        "torch==2.1",
        "transformers==4.31.0",
        "--extra-index-url",
        "https://pypi.nvidia.com",
        "--extra-index-url",
        "https://pypi.ngc.nvidia.com",
    ],
    machine_type="GPU",
)
def tensorrt_any(
    benchmark_settings: BenchmarkSettings,
    parameters: InputParameters,
    model_version: str,
    image_height: int,
    image_width: int,
) -> BenchmarkResults:
    import torch

    trt_path = prepare_tensorrt()
    diffusion_dir = trt_path / "demo" / "Diffusion"
    if str(diffusion_dir) not in sys.path:
        sys.path.insert(0, str(diffusion_dir))

    with contextlib.chdir(diffusion_dir):
        from cuda import cudart
        from stable_diffusion_pipeline import StableDiffusionPipeline
        from utilities import PIPELINE_TYPE

        # Initialize demo
        options = {
            "version": model_version,
            "denoising_steps": parameters.steps,
            "use_cuda_graph": True,
            "max_batch_size": 4,
            "output_dir": "output",
        }

        if model_version == "1.5":
            options["pipeline_type"] = PIPELINE_TYPE.TXT2IMG
        elif model_version == "xl-1.0":
            options["pipeline_type"] = PIPELINE_TYPE.XL_BASE
            options["vae_scaling_factor"] = 0.13025
        else:
            raise ValueError(f"Unknown model version: {model_version}")

        pipeline = StableDiffusionPipeline(**options)
        pipeline.loadEngines(
            engine_dir=f"engine-{model_version}-{torch.cuda.get_device_name(0)}",
            framework_model_dir="pytorch_model",
            onnx_dir=f"onnx-{model_version}",
            onnx_opset=18,
            opt_batch_size=1,
            opt_image_height=image_height,
            opt_image_width=image_width,
            enable_all_tactics=False,
            enable_refit=False,
            force_build=False,
            force_export=False,
            force_optimize=False,
            static_batch=True,
            static_shape=True,
            timing_cache=f"cache-{model_version}-{torch.cuda.get_device_name(0)}",
        )

        # Load resources
        _, shared_device_memory = cudart.cudaMalloc(pipeline.calculateMaxDeviceMemory())
        pipeline.activateEngines(shared_device_memory)
        pipeline.loadResources(image_height, image_width, 1, seed=0)
        inference_func = partial(
            pipeline.infer,
            [parameters.prompt],
            [""],
            image_height=image_height,
            image_width=image_width,
            save_image=False,
        )
        results = benchmark_settings.apply(inference_func)
        pipeline.teardown()

    return results


LOCAL_BENCHMARKS = [
    {
        "name": "TensorRT 9.0 (cuda graphs, static shapes)",
        "category": "SD1.5 (End-to-end)",
        "function": tensorrt_any,
        "kwargs": {
            "model_version": "1.5",
            "image_height": 512,
            "image_width": 512,
        },
    },
    {
        "name": "TensorRT 9.0 (cuda graphs, static shapes)",
        "category": "SDXL (End-to-end)",
        "function": tensorrt_any,
        "kwargs": {
            "model_version": "xl-1.0",
            "image_height": 512,
            "image_width": 512,
        },
    },
]
