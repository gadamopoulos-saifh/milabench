#!/usr/bin/env python
"""Export a milabench runs/ folder to a tidy CSV: one row per run.

Parses batch_size/packed out of each run's --run-name (expects the
"<test>-bs<N>-<packed|unpacked>" convention used in running_all_tests.sh),
and pulls the perf number out of the same .data files milabench itself
reads for `milabench compare`.
"""
import argparse
import csv
import os
import re

from milabench.common import _read_reports
from milabench.compare import fetch_runs
from milabench.summary import make_summary

RUN_NAME_RE = re.compile(r"^(?P<test>fsdp|tp|cp)-bs(?P<batch_size>\d+)-(?P<packed>packed|unpacked)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=None, help="runs/ folder, defaults to $MILABENCH_BASE/runs")
    parser.add_argument("--metric", default="train_rate")
    parser.add_argument("--stat", default="median")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    folder = args.folder
    if folder is None:
        base = os.environ["MILABENCH_BASE"]
        folder = os.path.join(base, "runs")

    runs = fetch_runs(folder, None)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "test", "batch_size", "packed", "bench", "perf"])

        for run in runs:
            summary = make_summary(_read_reports(run.path))
            m = RUN_NAME_RE.match(run.name)
            test, batch_size, packed = (m.group("test"), m.group("batch_size"), m.group("packed")) if m else ("", "", "")

            for bench, metrics in summary.items():
                perf = metrics.get(args.metric, {}).get(args.stat, "")
                writer.writerow([run.name, test, batch_size, packed, bench, perf])

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
