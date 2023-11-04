# Stable Diffusion Benchmarks

A set of benchmarks targeting different stable diffusion implementations to have a
better understanding of their performance and scalability.

## Benchmarks

Running on an A100 80G SXM hosted at [fal.ai](https://fal.ai).

<!-- START TABLE -->
### SD1.5 Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers (fp16, SDPA) |   1.654s |     1.654s |  1.632s |  1.675s |   30.24 it/s |

### SDXL Benchmarks
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers (fp16, SDPA) |   5.912s |     5.912s |  5.906s |  5.918s |    8.46 it/s |

<!-- END TABLE -->

Generation options:
- `prompt="A photo of a cat"`
- `num_inference_steps=50`

Environments (like torch and other library versions) for each benchmark are defined
under [benchmarks/](benchmarks/) folder.
