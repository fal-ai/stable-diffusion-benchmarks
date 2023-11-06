# Stable Diffusion Benchmarks

A set of benchmarks targeting different stable diffusion implementations to have a
better understanding of their performance and scalability.

## Benchmarks

Running on an A100 80G SXM hosted at [fal.ai](https://fal.ai).

> [!NOTE]
> Most of the implementations here are also based on Diffusers, which is an amazing library
> that pretty much the whole industry is using. However, when we use 'Diffusers' name in the
> benchmarks, it means the experience you might get with out-of-box Diffusers (w/applying
> necessary settings).

<!-- START TABLE -->
### SD1.5 (End-to-end) Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers (torch 2.1, SDPA) + OpenAI's [consistency decoder](https://github.com/openai/consistencydecoder)\*\* |   2.240s |     2.239s |  2.235s |  2.248s |   22.33 it/s |
| Diffusers (torch 2.1, xformers) |   1.729s |     1.728s |  1.720s |  1.747s |   28.94 it/s |
| Diffusers (torch 2.1, SDPA) |   1.604s |     1.603s |  1.589s |  1.618s |   31.19 it/s |
| Diffusers (torch 2.1, SDPA, [tiny VAE](https://github.com/madebyollin/taesd))\* |   1.567s |     1.562s |  1.547s |  1.602s |   32.02 it/s |
| Diffusers (torch 2.1, SDPA, compiled) |   1.354s |     1.354s |  1.351s |  1.356s |   36.93 it/s |
| Diffusers (torch 2.1, SDPA, compiled, NCHW channels last) |   1.058s |     1.057s |  1.056s |  1.060s |   47.29 it/s |
| OneFlow          |   0.951s |     0.953s |  0.941s |  0.957s |   52.48 it/s |
| TensorRT 9.0 (cuda graphs, static shapes) |   0.819s |     0.818s |  0.817s |  0.821s |   61.14 it/s |

### SDXL (End-to-end) Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| [minSDXL](https://github.com/cloneofsimo/minSDXL) (torch 2.1) |   8.146s |     8.146s |  8.137s |  8.155s |    6.14 it/s |
| Diffusers (torch 2.1, SDPA) |   5.932s |     5.932s |  5.924s |  5.940s |    8.43 it/s |
| [minSDXL+](https://github.com/isidentical/minSDXL) (torch 2.1, SDPA) |   5.887s |     5.887s |  5.872s |  5.897s |    8.49 it/s |
| Diffusers (torch 2.1, SDPA, [tiny VAE](https://github.com/madebyollin/taesd))\* |   5.739s |     5.738s |  5.722s |  5.767s |    8.71 it/s |
| Diffusers (torch 2.1, xformers) |   5.719s |     5.717s |  5.710s |  5.732s |    8.75 it/s |
| [minSDXL+](https://github.com/isidentical/minSDXL) (torch 2.1, flash-attention v2) |   5.323s |     5.322s |  5.313s |  5.340s |    9.39 it/s |
| Diffusers (torch 2.1, SDPA, compiled) |   5.217s |     5.216s |  5.213s |  5.220s |    9.59 it/s |
| Diffusers (torch 2.1, SDPA, compiled, NCHW channels last) |   5.136s |     5.137s |  5.125s |  5.147s |    9.73 it/s |
| OneFlow          |   4.605s |     4.607s |  4.581s |  4.625s |   10.85 it/s |
| TensorRT 9.0 (cuda graphs, static shapes) |   4.102s |     4.104s |  4.091s |  4.107s |   12.18 it/s |

<!-- END TABLE -->

Generation options:
- `prompt="A photo of a cat"`
- `num_inference_steps=50`
- For SD1.5, the width/height is 512x512 (the default); for SDXL, the width/height is 1024x1024.
- For all other options, the defaults from the generation systems are used.
- Weights are always half-precision (fp16) unless otherwise specified.
- Generation on benchmarks with a `*`/`**` means the used techniques might lead to quality degradation (or sometimes improvements) but the underlying diffusion model is still the same.

> [!NOTE]
> All the timings here are end to end, and reflects the time it takes to go from a single prompt
> to a decoded image. We are planning to make the benchmarking more granular and provide details
> and comparisons between each components (text encoder, VAE, and most importantly UNET) in the
> future, but for now, some of the results might not linearly scale with the number of inference
> steps since cost of certain components are one-time only.


Environments (like torch and other library versions) for each benchmark are defined
under [benchmarks/](benchmarks/) folder.
