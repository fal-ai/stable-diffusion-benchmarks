# Stable Diffusion Benchmarks

A set of benchmarks targeting different stable diffusion implementations to have a
better understanding of their performance and scalability.

## Benchmarks

Running on an A100 80G SXM hosted at [fal.ai](https://fal.ai).

<!-- START TABLE -->
### SD1.5 Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers (torch 2.1, xformers) |   1.758s |     1.759s |  1.746s |  1.772s |   28.43 it/s |
| Diffusers (torch 2.1, SDPA) |   1.591s |     1.590s |  1.581s |  1.601s |   31.44 it/s |
| Diffusers (torch 2.1, SDPA, compiled) |   1.352s |     1.351s |  1.348s |  1.356s |   37.01 it/s |
| Diffusers (torch 2.1, SDPA, compiled, NCHW channels last) |   1.066s |     1.065s |  1.062s |  1.076s |   46.95 it/s |
| OneFlow          |   0.951s |     0.953s |  0.941s |  0.957s |   52.48 it/s |
| TensorRT 9.0 (cuda graphs, static shapes) |   0.819s |     0.818s |  0.817s |  0.821s |   61.14 it/s |

### SDXL Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers (torch 2.1, SDPA) |   5.933s |     5.933s |  5.924s |  5.943s |    8.43 it/s |
| Diffusers (torch 2.1, xformers) |   5.724s |     5.724s |  5.714s |  5.731s |    8.74 it/s |
| Diffusers (torch 2.1, SDPA, compiled) |   5.246s |     5.247s |  5.233s |  5.259s |    9.53 it/s |
| Diffusers (torch 2.1, SDPA, compiled, NCHW channels last) |   5.132s |     5.132s |  5.121s |  5.142s |    9.74 it/s |
| OneFlow          |   4.605s |     4.607s |  4.581s |  4.625s |   10.85 it/s |
| TensorRT 9.0 (cuda graphs, static shapes) |   4.102s |     4.104s |  4.091s |  4.107s |   12.18 it/s |

<!-- END TABLE -->

Generation options:
- `prompt="A photo of a cat"`
- `num_inference_steps=50`
- For SD1.5, the width/height is 512x512 (the default); for SDXL, the width/height is 1024x1024.
- For all other options, the defaults from the generation systems are used.

Environments (like torch and other library versions) for each benchmark are defined
under [benchmarks/](benchmarks/) folder.
