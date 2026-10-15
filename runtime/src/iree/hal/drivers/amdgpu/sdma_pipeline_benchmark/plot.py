# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plot recorded command-stream intervals without closing inter-stage gaps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
args = parser.parse_args()
base = args.results
samples = json.loads((base / "samples.json").read_text())
timeline = json.loads((base / "timelines.json").read_text())
summary = json.loads((base / "summary.json").read_text())
modes = list(dict.fromkeys(s["mode"] for s in samples))
fig, axes = plt.subplots(
    len(modes), 1, figsize=(12, 2.4 * len(modes)), sharex=True, squeeze=False
)
for ax, mode in zip(axes[:, 0], modes):
    median = summary[mode]["median_gpu_us"]
    chosen = min(
        (x for x in samples if x["mode"] == mode and x["iter"] >= 4),
        key=lambda x: abs(x["gpu_us"] - median),
    )
    stages = [
        s for s in timeline if s["run"] == chosen["run"] and s["iter"] == chosen["iter"]
    ]
    for s in stages:
        for y, kind, color in [(1, "copy", "#277da8"), (0, "compute", "#e29b39")]:
            start = s[kind + "_start_us"] / 1000
            duration = (s[kind + "_end_us"] - s[kind + "_start_us"]) / 1000
            ax.broken_barh(
                [(start, duration)],
                (y - 0.32, 0.64),
                facecolors=color,
                edgecolors="white",
                linewidth=0.5,
            )
            ax.text(
                start + duration / 2,
                y,
                str(s["stage"]),
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_yticks([0, 1], ["Compute interval", "Weight copy"])
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(
        f"{mode.upper()} — measured replay ({chosen['gpu_us'] / 1000:.3f} ms)",
        loc="left",
        fontsize=11,
    )
axes[-1, 0].set_xlabel("GPU time from first copy start (ms)")
fig.suptitle("HRX: eight weight-copy / optimized matmul stages", fontsize=13)
fig.tight_layout()
fig.savefig(base / "timeline.svg")
fig.savefig(base / "timeline.png", dpi=160)
