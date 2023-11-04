import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rich.progress import track

from benchmarks import diffusers
from benchmarks.settings import BenchmarkSettings, InputParameters

ALL_BENCHMARKS = [
    *diffusers.LOCAL_BENCHMARKS,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--session-id",
        type=str,
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    # For ensuring consistency among results, make sure to compare the numbers
    # within the same node. So the driver, cuda version, power supply, CPU compute
    # etc. are all the same.
    parser.add_argument("--target-node", type=str, default=None)

    options = parser.parse_args()

    settings = BenchmarkSettings(
        warmup_iterations=options.warmup_iterations,
        benchmark_iterations=options.iterations,
    )
    parameters = InputParameters(prompt="A photo of a cat", steps=50)

    timings = []
    for benchmark in track(ALL_BENCHMARKS, description="Running benchmarks..."):
        print(f"Running benchmark: {benchmark['name']}")
        function = benchmark["function"].on(_scheduler="nomad")
        if options.target_node:
            function = function.on(
                _scheduler_options={
                    "target_node": options.target_node,
                }
            )

        benchmark_results = function(
            benchmark_settings=settings,
            parameters=parameters,
            **benchmark["kwargs"],
        )
        timings.append(
            {
                "name": benchmark["name"],
                "timings": benchmark_results.timings,
            }
        )

    results = {
        "settings": asdict(settings),
        "parameters": asdict(parameters),
        "timings": timings,
    }

    with open(options.results_dir / f"{options.session_id}.json", "w") as results_file:
        json.dump(results, results_file)


if __name__ == "__main__":
    main()
