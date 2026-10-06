# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare the real BF16 helpers with 108 row/width/traversal/layout cases.

The check recipe's sample ordinal controls tensor extents; compiler config
controls kernel specialization. Bind both from the same row/width selection.
Each invocation releases its synthetic tensors before the next shape starts.
"""

import argparse
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checker", type=pathlib.Path, required=True)
parser.add_argument("--device", default="amdgpu")
parser.add_argument("--target", default="amdgpu:gfx1151")
args = parser.parse_args()

fixture = pathlib.Path(__file__).with_name("linear_tiled.loom")
motifs = fixture.parent.parent


def check_shape(inputs, outputs, sample, rows, group_size, *, column_major=False):
    print(
        json.dumps(
            dict(
                rows=rows,
                inputs=inputs,
                outputs=outputs,
                group_size=group_size,
                layout="column_major" if column_major else "row_major",
            )
        ),
        flush=True,
    )
    result = subprocess.run(
        [
            str(args.checker),
            str(fixture),
            "--library=" + str(motifs / "linear.loom"),
            "--library=" + str(motifs / "linear_tiled.loom"),
            "--device=" + args.device,
            "--target=" + args.target,
            f"--case=@linear_bf16_{inputs}_{outputs}",
            f"--sample={sample}",
            f"--config=projection_test.rows={rows}",
            f"--config=projection_test.inputs={inputs}",
            f"--config=projection_test.outputs={outputs}",
            f"--config=projection_test.row_tile_group_size={group_size}",
            f"--config=projection_test.column_major={int(column_major)}",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    report = json.loads(result.stdout)
    for field, expected in (
        ("sample_count", 1),
        ("failed_sample_count", 0),
        ("planning_issue_count", 0),
        ("skipped_case_count", 0),
    ):
        if report[field] != expected:
            raise RuntimeError(f"{field}: expected {expected}, got {report[field]}")


for inputs, outputs in ((6144, 6144), (6144, 1536), (6144, 16384), (16384, 6144)):
    for sample, rows in enumerate(
        (
            16,
            32,
            48,
            64,
            80,
            512,
            528,
            544,
            560,
            640,
            704,
            768,
            832,
            896,
            960,
            1088,
            2048,
            2560,
            3072,
            4608,
        )
    ):
        check_shape(inputs, outputs, sample, rows, 8)
        if rows in (16, 80):
            check_shape(inputs, outputs, sample, rows, 8, column_major=True)
        if rows >= 1088:
            check_shape(inputs, outputs, sample, rows, (rows + 63) // 64)
print("PASS: 100 row-major and eight column-major exact BF16 comparisons.", flush=True)
