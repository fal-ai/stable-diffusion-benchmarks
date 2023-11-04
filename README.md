# Stable Diffusion Benchmarks

A set of benchmarks targeting different stable diffusion implementations to have a
better understanding of their performance and scalability.

## Benchmarks

Running on an A100 80G SXM hosted at [fal.ai](https://fal.ai).

<!-- START TABLE -->
|                  | mean (s) | median (s) | min (s) | max (s) | speed (it/s) |
|------------------|----------|------------|---------|---------|--------------|
| Diffusers SD1.5  |   1.619s |     1.614s |  1.603s |  1.645s |   30.97 it/s |
| Diffusers SDXL   |   5.916s |     5.917s |  5.909s |  5.927s |    8.45 it/s |
<!-- END TABLE -->

Generation options:
- `prompt="A photo of a cat"`
- `num_inference_steps=50`
