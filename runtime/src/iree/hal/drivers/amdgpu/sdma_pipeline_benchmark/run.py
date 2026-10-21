# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare SDMA IB, inline SDMA ring, and shader pipelines."""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--modes",
    nargs="+",
    choices=["sdma-ib", "sdma-ring", "shader"],
    default=["sdma-ring", "shader"],
)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--rocm", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--allow-experimental-ib", action="store_true")
parser.add_argument(
    "--engine", type=int, default=0, help="SDMA engine ordinal (-1: runtime selects)"
)
args = parser.parse_args()
if len(set(args.modes)) != len(args.modes):
    parser.error("Duplicate modes")
if "sdma-ib" in args.modes and not args.allow_experimental_ib:
    parser.error("sdma-ib requires --allow-experimental-ib and an IB-enabled kernel")
data = args.data.resolve()
manifest = json.loads((data / "optimized-manifest.json").read_text())
for name, digest in manifest["sha256"].items():
    if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError("Prepared data checksum mismatch: " + name)
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = (
    str(args.rocm.resolve() / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
)

# Rotate order each round so each of the three modes occupies each position.
order = [
    mode
    for shift in range(3)
    for mode in args.modes[shift % len(args.modes) :]
    + args.modes[: shift % len(args.modes)]
]
rows, timeline = [], []
for run, mode in enumerate(order):
    path = out / f"{run}-{mode}.log"
    with path.open("w") as log:
        command = [
            str(args.binary.resolve()),
            "--sdma_data_directory=" + str(data),
            "--sdma_copy_mode=" + mode,
            "--sdma_stages=8",
            "--sdma_iterations=24",
            "--sdma_engine=" + str(args.engine),
        ]
        if mode == "sdma-ib":
            command.append("--sdma_allow_experimental_ib=true")
        subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=90,
        )
    text = path.read_text()
    assert len(re.findall(r"VALIDATE stage=\d+ max_abs_error=0", text)) == 8
    for line in text.splitlines():
        if not line.startswith(("RESULT ", "STAGE ")):
            continue
        values = dict(x.split("=", 1) for x in line.split()[1:])
        row = {
            k: (v if k == "mode" else float(v) if "." in v else int(v))
            for k, v in values.items()
        }
        row.update(run=run, mode=mode)
        (rows if line.startswith("RESULT ") else timeline).append(row)
    print("completed", path.name, flush=True)

steady = [r for r in rows if r["iter"] >= 4]
stats = {}
for mode in args.modes:
    samples = [r for r in steady if r["mode"] == mode]
    assert len(samples) == 60
    stats[mode] = {"samples": len(samples)}
    for field in ["gpu_us", "host_us", "submit_us", "copy_submit_us"]:
        stats[mode]["median_" + field] = statistics.median(r[field] for r in samples)
# Preserve run-to-run variation instead of hiding it behind one median.
stats["runs"] = [
    {
        "run": run,
        "mode": mode,
        "median_gpu_us": statistics.median(
            r["gpu_us"] for r in steady if r["run"] == run
        ),
    }
    for run, mode in enumerate(order)
]
for run in range(len(order)):
    for iteration in range(24):
        stages = sorted(
            [s for s in timeline if s["run"] == run and s["iter"] == iteration],
            key=lambda s: s["stage"],
        )
        assert len(stages) == 8
        for i, s in enumerate(stages):
            assert s["copy_start_us"] < s["copy_end_us"] <= s["compute_start_us"] + 1
            assert s["compute_start_us"] < s["compute_end_us"]
            if i > 1:
                assert s["copy_start_us"] + 1 >= stages[i - 2]["compute_end_us"]
            if i > 0:
                assert s["compute_start_us"] >= stages[i - 1]["compute_end_us"]
                overlap = max(
                    0,
                    min(s["copy_end_us"], stages[i - 1]["compute_end_us"])
                    - max(s["copy_start_us"], stages[i - 1]["compute_start_us"]),
                )
                s["overlap_previous_compute_us"] = overlap
                # Record overlap instead of requiring it: lack of overlap is
                # a valid performance outcome, not a correctness failure.
if "sdma-ib" in stats and "sdma-ring" in stats:
    stats["ratio_ring_over_ib_gpu"] = (
        stats["sdma-ring"]["median_gpu_us"] / stats["sdma-ib"]["median_gpu_us"]
    )
for name, value in [("summary", stats), ("timelines", timeline), ("samples", rows)]:
    (out / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(stats, indent=2))
