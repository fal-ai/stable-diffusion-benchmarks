from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkSettings:
    warmup_iterations: int = 3
    benchmark_iterations: int = 10

    def apply(self, test_fn: Callable[[], Any]) -> BenchmarkResults:
        for _ in range(self.warmup_iterations):
            test_fn()

        timings = []
        for _ in range(self.benchmark_iterations):
            t0 = time.perf_counter()
            test_fn()
            timings.append(time.perf_counter() - t0)

        return BenchmarkResults(timings=timings)


@dataclass
class BenchmarkResults:
    timings: list[float]


@dataclass
class InputParameters:
    prompt: str = "A photo of a cat"
    steps: int = 50
