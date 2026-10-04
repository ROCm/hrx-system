# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check exact F32 convolution at four spatial shapes with and without skips.

The recipe's sample ordinal controls tensor width; compiler configuration
binds the matching geometry and residual epilogue. Every invocation compares
both distinct-output and actual skip/output-alias launches with the original
helper, using full 96-channel reductions and original three-plane weights.
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

fixture = pathlib.Path(__file__).with_name("convolution_tiled.loom")
model = fixture.parent.parent

for case, height, widths in (
    ("single_row", 1, (1, 2, 33)),
    ("spatial", 17, (19,)),
):
    for sample, width in enumerate(widths):
        for add_residual in (0, 1):
            print(
                json.dumps(dict(height=height, width=width, add_residual=add_residual)),
                flush=True,
            )
            result = subprocess.run(
                [
                    str(args.checker),
                    str(fixture),
                    "--library=" + str(model / "kernels/convolution.loom"),
                    "--library=" + str(model / "kernels/convolution_tiled.loom"),
                    "--device=" + args.device,
                    "--target=" + args.target,
                    f"--case=@convolution_f32_{case}",
                    f"--sample={sample}",
                    f"--config=convolution_test.height={height}",
                    f"--config=convolution_test.width={width}",
                    f"--config=convolution_test.add_residual={add_residual}",
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
                    raise RuntimeError(
                        f"{field}: expected {expected}, got {report[field]}"
                    )
print("PASS: 8 cases, 16 exact distinct-output/alias comparisons.", flush=True)
