import json
import statistics
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

README_PATH = Path(__file__).parent.parent / "README.md"


def main():
    parser = ArgumentParser()
    parser.add_argument("results_files", type=Path, nargs="+")

    options = parser.parse_args()
    results = {
        result_file.stem: json.loads(result_file.read_text())
        for result_file in options.results_files
    }

    benchmarks = defaultdict(dict)
    for result_name, result_values in results.items():
        for timing in result_values["timings"]:
            benchmarks[(timing["category"], timing["name"])][result_name] = timing[
                "timings"
            ]

    with Console() as console:
        table = Table()
        table.add_column("Benchmark")
        for result_name in results.keys():
            table.add_column(" ".join(map(str.title, result_name.split("-"))))

        for benchmark_key, benchmark_results in sorted(
            benchmarks.items(),
            key=lambda kv: kv[0],
        ):
            if "CPU" in benchmark_key[0]:
                continue

            row = [f"{benchmark_key[0].split(' ')[0]:5} {benchmark_key[1]}"]
            raw_values = []
            for result_name in results.keys():
                if result_name in benchmark_results:
                    raw_values.append(
                        (
                            statistics.mean(benchmark_results[result_name]),
                            statistics.stdev(benchmark_results[result_name]),
                        )
                    )
                else:
                    raw_values.append((float("nan"), float("nan")))

            # Bold the best result
            best_index = raw_values.index(min(raw_values))
            for i, (mean, std) in enumerate(raw_values):
                if i == best_index:
                    row.append(f"[bold][green]{mean:.2f}s[/green][/bold] ± {std:.2f}s")
                elif mean is not float("nan"):
                    row.append(f"{mean:.2f}s ± {std:.2f}s")
                else:
                    row.append("N/A")

            table.add_row(*row)

        console.print(table)


if __name__ == "__main__":
    main()
