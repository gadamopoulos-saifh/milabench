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
import sys
import traceback

from milabench.common import _read_reports
from milabench.summary import make_summary

RUN_NAME_RE = re.compile(r"^(?P<test>fsdp|tp|cp)-bs(?P<batch_size>\d+)-(?P<packed>packed|unpacked)$")


def split_attempts(events):
    """Split one .data file's events into separate attempts.

    Reusing a fixed --run-name across reruns makes milabench append to the
    same .data file (DataReporter opens in "a" mode), so a single file can
    contain several concatenated config/start/.../end blocks. aggregate()
    assumes exactly one start/end pair per file and throws on the second
    one, so every re-run bench gets misreported as "did not finish
    successfully". Each attempt starts with a fresh "config" event, so we
    split there and let make_summary(..., latest_only=True) pick the most
    recent successful attempt per bench.
    """
    groups = []
    current = None
    for e in events:
        if e.get("event") == "config":
            if current:
                groups.append(current)
            current = []
        if current is not None:
            current.append(e)
    if current:
        groups.append(current)
    return groups


def read_summary(path):
    raw = _read_reports(path)
    split = {}
    for filepath, events in raw.items():
        for i, group in enumerate(split_attempts(events)):
            split[f"{filepath}#{i}"] = group
    return make_summary(split, latest_only=True)


def list_runs(folder):
    if not os.path.isdir(folder):
        return
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
    # mean (not median, which is what `milabench compare` defaults to) so the
    # numbers here line up with what `milabench report` prints
    parser.add_argument("--stat", default="mean")
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

            try:
                summary = read_summary(path) or {}
            except Exception:
                print(f"Skipping {run_name}: make_summary failed", file=sys.stderr)
                traceback.print_exc()
                summary = {}

            if not summary:
                # run produced no usable data (e.g. crashed/OOM'd)
                writer.writerow([run_name, test, batch_size, packed, "", ""])
                continue

            for bench, metrics in summary.items():
                try:
                    # metrics itself, or metrics[args.metric], can be present
                    # but None (e.g. the underlying aggregate hit an internal
                    # error), not just absent
                    perf = ((metrics or {}).get(args.metric) or {}).get(args.stat, "")
                except Exception:
                    print(f"Skipping {run_name}/{bench}: couldn't extract {args.metric}", file=sys.stderr)
                    perf = ""
                writer.writerow([run_name, test, batch_size, packed, bench, perf])

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
