# Stable Diffusion Benchmarks

A set of benchmarks targeting different stable diffusion implementations to have a
better understanding of their performance and scalability.

## Benchmarks

Running on an A100 80G SXM hosted at [fal.ai](https://fal.ai).

Generation options:
- `prompt="A photo of a cat"`
- `num_inference_steps=50`

<!-- START TABLE -->
|                  | mean (s) | median (s) | min (s) | max (s) |
|------------------|----------|------------|---------|---------|
| Diffusers SD1.5  |   1.619s |     1.614s |  1.603s |  1.645s |
| Diffusers SDXL   |   5.916s |     5.917s |  5.909s |  5.927s |
<!-- END TABLE -->
