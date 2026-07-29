#!/usr/bin/env python
"""Export a milabench runs/ folder to a tidy CSV: one row per run.

Parses batch_size/packed out of each run's --run-name (expects the
"<test>-bs<N>-<packed|unpacked>" convention used in running_all_tests.sh),
and pulls the perf number out of the same .data files milabench itself
reads for `milabench compare`.

Deliberately doesn't reuse milabench.compare.fetch_runs: it tries to parse
a date out of every run folder name assuming a "name.date.microseconds"
shape, but milabench's own default auto-generated run names only have one
dot ("word.timestamp"), which makes fetch_runs crash on its own defaults.
We don't need date sorting here, so we just list the folders ourselves.
"""
import argparse
import csv
import os
import re

from milabench.common import _read_reports
from milabench.summary import make_summary

RUN_NAME_RE = re.compile(r"^(?P<test>fsdp|tp|cp)-bs(?P<batch_size>\d+)-(?P<packed>packed|unpacked)$")


def list_runs(folder):
    for name in sorted(os.listdir(folder)):
        if name.startswith("install") or name.startswith("prepare"):
            continue
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            yield name, path


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

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "test", "batch_size", "packed", "bench", "perf"])

        for run_name, path in list_runs(folder):
            m = RUN_NAME_RE.match(run_name)
            test, batch_size, packed = (m.group("test"), m.group("batch_size"), m.group("packed")) if m else ("", "", "")

            summary = make_summary(_read_reports(path))
            if not summary:
                # run produced no usable data (e.g. crashed/OOM'd)
                writer.writerow([run_name, test, batch_size, packed, "", ""])
                continue

            for bench, metrics in summary.items():
                # metrics.get(args.metric) can be present but None (e.g. the
                # underlying aggregate hit an internal error), not just absent
                perf = (metrics.get(args.metric) or {}).get(args.stat, "")
                writer.writerow([run_name, test, batch_size, packed, bench, perf])

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
