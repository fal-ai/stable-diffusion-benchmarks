import json
import statistics
from argparse import ArgumentParser
from pathlib import Path

README_PATH = Path(__file__).parent.parent / "README.md"
TABLE_HEADER = "|                  | mean (s) | median (s) | min (s) | max (s) |\n"
TABLE_DIVIDER = "|------------------|----------|------------|---------|---------|\n"
TABLE_ROW_FORMAT = (
    "| {name:16} | {mean:7.3f}s | {median:9.3f}s | {min:6.3f}s | {max:6.3f}s |\n"
)
START_MARKER = "<!-- START TABLE -->\n"
END_MARKER = "<!-- END TABLE -->\n"


def main():
    parser = ArgumentParser()
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--document-path", type=Path, default=README_PATH)

    options = parser.parse_args()
    with open(options.results_file) as f:
        results = json.load(f)

    with open(options.document_path) as f:
        lines = f.readlines()

    table_lines = [TABLE_HEADER, TABLE_DIVIDER]
    for timing in results["timings"]:
        benchmark_name = timing["name"]
        benchmark_timings = timing["timings"]
        table_lines.append(
            TABLE_ROW_FORMAT.format(
                name=benchmark_name,
                mean=statistics.mean(benchmark_timings),
                median=statistics.median(benchmark_timings),
                min=min(benchmark_timings),
                max=max(benchmark_timings),
            )
        )

    start_index = lines.index(START_MARKER) + 1
    end_index = lines.index(END_MARKER)
    lines[start_index:end_index] = table_lines

    with open(options.document_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
