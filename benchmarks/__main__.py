import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rich.progress import track

from benchmarks import (
    benchmark_comfy,
    benchmark_diffusers,
    benchmark_experimental,
    benchmark_minsdxl,
    benchmark_oneflow,
    benchmark_tensorrt,
)
from benchmarks.settings import BenchmarkSettings, InputParameters

ALL_BENCHMARKS = [
    *benchmark_diffusers.LOCAL_BENCHMARKS,
    *benchmark_tensorrt.LOCAL_BENCHMARKS,
    *benchmark_oneflow.LOCAL_BENCHMARKS,
    *benchmark_minsdxl.LOCAL_BENCHMARKS,
    *benchmark_experimental.LOCAL_BENCHMARKS,
    *benchmark_comfy.LOCAL_BENCHMARKS,
]


def load_previous_timings(
    session_file: Path,
    settings: BenchmarkSettings,
    parameters: InputParameters,
) -> dict[tuple[str, str], list[float]]:
    if not session_file.exists():
        return {}

    with open(session_file) as stream:
        results = json.load(stream)

    if results["settings"] != asdict(settings):
        print(f"Settings mismatch: {results['settings']} != {asdict(settings)}")
        print(f"Skipping {session_file}")
        return {}

    if results["parameters"] != asdict(parameters):
        print(f"Parameters mismatch: {results['parameters']} != {asdict(parameters)}")
        print(f"Skipping {session_file}")
        return {}

    return {
        (timing["category"], timing["name"]): timing["timings"]
        for timing in results["timings"]
    }


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
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Force running all benchmarks, even if they have already been run.",
    )
    parser.add_argument(
        "--force-run-only",
        type=str.lower,
        help="Force running only the specified benchmarks, even if they have already been run.",
        choices=["diffusers", "tensorrt", "minsdxl", "oneflow", "consistency", "comfy"],
    )

    # For ensuring consistency among results, make sure to compare the numbers
    # within the same node. So the driver, cuda version, power supply, CPU compute
    # etc. are all the same.
    parser.add_argument("--target-node", type=str, default=None)

    options = parser.parse_args()
    session_file = options.results_dir / f"{options.session_id}.json"

    settings = BenchmarkSettings(
        warmup_iterations=options.warmup_iterations,
        benchmark_iterations=options.iterations,
    )
    parameters = InputParameters(prompt="A photo of a cat", steps=50)

    timings = []
    previous_timings = load_previous_timings(session_file, settings, parameters)
    for benchmark in track(ALL_BENCHMARKS, description="Running benchmarks..."):
        benchmark_key = (benchmark["category"], benchmark["name"])
        should_skip = benchmark.get("skip_if", False)
        should_force_run = options.force_run or (
            options.force_run_only
            and options.force_run_only in benchmark["name"].lower()
        )
        if benchmark_key in previous_timings and (not should_force_run or should_skip):
            print(f"Skipping {benchmark_key} (already run)")
            timings.append(
                {
                    "name": benchmark["name"],
                    "category": benchmark["category"],  # "SD1.5", "SDXL"
                    "timings": previous_timings[benchmark_key],
                }
            )
            continue

        print(f"Running benchmark: {benchmark_key}")
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
            **benchmark.get("kwargs", {}),
        )
        timings.append(
            {
                "name": benchmark["name"],
                "category": benchmark["category"],  # "SD1.5", "SDXL"
                "timings": benchmark_results.timings,
            }
        )

    results = {
        "settings": asdict(settings),
        "parameters": asdict(parameters),
        "timings": timings,
    }

    with open(session_file, "w") as stream:
        json.dump(results, stream)
        stream.write("\n")


if __name__ == "__main__":
    main()
