# Stable Diffusion Benchmarks

A set of benchmarks targeting different stable diffusion implementations to have a
better understanding of their performance and scalability.

## Benchmarks

Running on an A100 80G SXM hosted at [fal.ai](https://fal.ai).

<!-- START TABLE -->
### SD1.5 Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers (fp16, SDPA) |   1.591s |     1.590s |  1.581s |  1.601s |   31.44 it/s |
| Diffusers (fp16, xformers) |   1.758s |     1.759s |  1.746s |  1.772s |   28.43 it/s |
| Diffusers (fp16, SDPA, compiled) |   1.352s |     1.351s |  1.348s |  1.356s |   37.01 it/s |
| Diffusers (fp16, SDPA, compiled, NCHW channels last) |   1.066s |     1.065s |  1.062s |  1.076s |   46.95 it/s |

### SDXL Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers (fp16, SDPA) |   5.933s |     5.933s |  5.924s |  5.943s |    8.43 it/s |
| Diffusers (fp16, xformers) |   5.724s |     5.724s |  5.714s |  5.731s |    8.74 it/s |
| Diffusers (fp16, SDPA, compiled) |   5.246s |     5.247s |  5.233s |  5.259s |    9.53 it/s |
| Diffusers (fp16, SDPA, compiled, NCHW channels last) |   5.132s |     5.132s |  5.121s |  5.142s |    9.74 it/s |

<!-- END TABLE -->

Generation options:
- `prompt="A photo of a cat"`
- `num_inference_steps=50`

Environments (like torch and other library versions) for each benchmark are defined
under [benchmarks/](benchmarks/) folder.
