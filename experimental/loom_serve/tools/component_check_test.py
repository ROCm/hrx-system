#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise typed comparison through a real source-JIT copy and readback."""

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def encode(element_type, values):
    if element_type == "bf16":
        # Inputs are exactly BF16-representable, apart from deliberate NaNs.
        return b"".join(struct.pack("<f", value)[2:] for value in values)
    return struct.pack(
        "<" + {"f16": "e", "f32": "f", "f64": "d"}[element_type] * len(values), *values
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    arguments = parser.parse_args()
    model = Path(__file__).resolve().parent / "testdata" / "component"
    with tempfile.TemporaryDirectory(prefix="loom-component-check-") as directory:
        work = Path(directory)

        def check(
            name,
            element_type,
            values,
            expected,
            *,
            accepted=True,
            extra=(),
            diagnostic=None,
        ):
            (work / "input").write_bytes(values)
            (work / "expected").write_bytes(expected)
            result = subprocess.run(
                [
                    *arguments.checker,
                    f"--model={model}",
                    "--root=copy",
                    f"--config=copy.byte_count={len(values)}",
                    f"--input={work}/input",
                    f"--expected={work}/expected",
                    f"--actual={work}/actual",
                    f"--output_type={element_type}",
                    "--atol=0",
                    "--rtol=0",
                    *extra,
                ],
                capture_output=True,
                text=True,
            )
            if (result.returncode == 0) != accepted or (
                diagnostic is not None and diagnostic not in result.stderr
            ):
                raise AssertionError(
                    f"{name}: exit={result.returncode}\n{result.stdout}\n{result.stderr}"
                )
            records = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith('{"iteration"')
            ]
            if accepted:
                assert [record["iteration"] for record in records] == [0, 1], name
                assert (work / "actual").read_bytes() == values, name
                assert all(
                    record["output_type"] == element_type for record in records
                ), name
                assert all(record["nonfinite"] == 0 for record in records), name
            print(
                json.dumps(dict(check=name, accepted=accepted, records=records)),
                flush=True,
            )
            return records

        for element_type in ("f16", "bf16", "f32", "f64"):
            values = encode(element_type, [0, -0.0, 1, -1, 0.5, -2, 16, 0.125])
            exact = check(element_type + "_exact", element_type, values, values)
            assert all(
                record["elements"] == 8 and record["different"] == 0 for record in exact
            )
            changed = encode(element_type, [0, -0.0, 1.125, -1, 0.5, -2, 16, 0.125])
            check(
                element_type + "_reject",
                element_type,
                values,
                changed,
                accepted=False,
                diagnostic="exceed tolerance",
            )
            observed = check(
                element_type + "_report",
                element_type,
                values,
                changed,
                extra=("--report_only",),
            )
            assert all(
                record["different"] == 1 and record["outside_element_envelope"] == 1
                for record in observed
            )
            aggregate = check(
                element_type + "_aggregate",
                element_type,
                values,
                changed,
                extra=("--relative_l2_tolerance=0.1",),
            )
            assert all(record["comparison"] == "relative_l2" for record in aggregate)
            check(
                element_type + "_tight",
                element_type,
                values,
                changed,
                accepted=False,
                extra=("--relative_l2_tolerance=1e-12",),
                diagnostic="exceeds",
            )
            negative_zero = encode(
                element_type, [-0.0, -0.0, 1, -1, 0.5, -2, 16, 0.125]
            )
            signed = check(
                element_type + "_signed_zero", element_type, values, negative_zero
            )
            assert all(
                record["different"] == 1 and record["relative_l2"] == 0
                for record in signed
            )
            zeros = bytes(len(values))
            zero_reference = check(
                element_type + "_zero_reference",
                element_type,
                values,
                zeros,
                extra=("--report_only",),
            )
            assert all(record["relative_l2"] is None for record in zero_reference)
            nonfinite = encode(element_type, [math.nan, 0, 1, -1, 0.5, -2, 16, 0.125])
            for nonfinite_side in ("actual", "expected"):
                check(
                    element_type + "_nonfinite_" + nonfinite_side,
                    element_type,
                    nonfinite if nonfinite_side == "actual" else values,
                    nonfinite if nonfinite_side == "expected" else values,
                    accepted=False,
                    extra=("--report_only",),
                    diagnostic="nonfinite component pairs",
                )
            check(
                element_type + "_partial",
                element_type,
                values,
                values[:-1],
                accepted=False,
                diagnostic="nonempty, whole",
            )

        for magnitude in (1e-300, 1e300):
            records = check(
                f"f64_norm_{magnitude}",
                "f64",
                encode("f64", [magnitude, -magnitude]),
                encode("f64", [0.5 * magnitude, -0.5 * magnitude]),
                extra=("--report_only",),
            )
            assert all(
                math.isclose(record["relative_l2"], 1.0, rel_tol=1e-9)
                for record in records
            )
        overflow = check(
            "f64_difference_overflow",
            "f64",
            encode("f64", [1.7e308]),
            encode("f64", [-1.7e308]),
            extra=("--report_only",),
        )
        assert all(
            record["maximum_absolute_error"] is None and record["relative_l2"] == 2
            for record in overflow
        )
        check(
            "empty",
            "f32",
            encode("f32", [1]),
            b"",
            accepted=False,
            diagnostic="nonempty, whole",
        )
        check(
            "integer_rejected",
            "i32",
            encode("f32", [1]),
            encode("f32", [1]),
            accepted=False,
            diagnostic="output type must be",
        )
        for bound in ("-1", "nan", "inf"):
            check(
                "invalid_bound_" + bound,
                "f32",
                encode("f32", [1]),
                encode("f32", [1]),
                accepted=False,
                extra=("--relative_l2_tolerance=" + bound,),
                diagnostic="finite and nonnegative",
            )
    print(
        "PASS: typed observations preserve raw readback and comparison/error semantics.",
        flush=True,
    )


if __name__ == "__main__":
    main()
