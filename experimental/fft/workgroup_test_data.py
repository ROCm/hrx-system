# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small independent DFT fixtures for cooperative forward and power kernels."""

import argparse
import math
from pathlib import Path

from experimental.fft.test_data import load, round_storage, write_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", type=Path)
    parser.add_argument("length", type=int)
    parser.add_argument("frames", type=int)
    parser.add_argument("input_type", choices=("f32", "f16", "bf16"))
    parser.add_argument("output_type", choices=("f32", "f16", "bf16"))
    args = parser.parse_args()
    length, frames = args.length, args.frames
    inputs = [
        round_storage(((position * 17 + frame * 11) % 31 - 15) / 19, args.input_type)
        for frame in range(frames)
        for position in range(length)
    ]
    window = [
        round_storage(0.5 - 0.5 * math.cos(math.tau * p / length), "f32")
        for p in range(length)
    ]
    input_path = args.stem.with_name(args.stem.name + "-input.npy")
    window_path = args.stem.with_name(args.stem.name + "-window.npy")
    write_array(input_path, inputs + [-123.0] * 16, args.input_type)
    write_array(window_path, window, "f32")
    cases = []
    for mode in ("forward", "power"):
        expected = []
        for frame in range(frames):
            values = inputs[frame * length : (frame + 1) * length]
            if mode == "power":
                values = [value * weight for value, weight in zip(values, window)]
            for frequency in range(length // 2 + 1):
                phases = [math.tau * frequency * p / length for p in range(length)]
                real = math.fsum(v * math.cos(a) for v, a in zip(values, phases))
                imag = (
                    0.0
                    if frequency in (0, length // 2)
                    else -math.fsum(v * math.sin(a) for v, a in zip(values, phases))
                )
                expected.extend(
                    [real, imag] if mode == "forward" else [real * real + imag * imag]
                )
        expected = [round_storage(value, args.output_type) for value in expected]
        tolerance = {"f32": 5e-6, "f16": 2**-10 + 5e-6, "bf16": 2**-7 + 5e-6}[
            args.output_type
        ]
        bound = tolerance * max((abs(value) for value in expected), default=0.0) + 5e-6
        expected_path = args.stem.with_name(f"{args.stem.name}-{mode}-expected.npy")
        write_array(expected_path, expected + [-123.0] * 16, args.output_type)
        input_count, output_count = len(inputs) + 16, len(expected) + 16
        output_bytes = 4 if args.output_type == "f32" else 2
        body = f"check.case @{mode} {{\n"
        for name, path, count, kind in (
            ("input", input_path, input_count, args.input_type),
            ("original_input", input_path, input_count, args.input_type),
            ("expected", expected_path, output_count, args.output_type),
            ("window", window_path, length, "f32"),
            ("original_window", window_path, length, "f32"),
        ):
            body += load(name, path, count, kind)
        body += f"""  %output_storage = check.generate.fill value(-123.0) : tensor<{output_count + 16}x{args.output_type}>
  %output = check.tensor.view %output_storage offset({16 * output_bytes}) : tensor<{output_count + 16}x{args.output_type}> -> tensor<{output_count}x{args.output_type}>
  %prefix = check.tensor.view %output_storage offset(0) : tensor<{output_count + 16}x{args.output_type}> -> tensor<16x{args.output_type}>
  %suffix = check.tensor.view %output offset({len(expected) * output_bytes}) : tensor<{output_count}x{args.output_type}> -> tensor<16x{args.output_type}>
  %guard = check.generate.fill value(-123.0) : tensor<16x{args.output_type}>
  %frames = check.literal value({frames}) : i32
"""
        operands = (
            "%input, %output" + (", %window" if mode == "power" else "") + ", %frames"
        )
        types = (
            f"tensor<{input_count}x{args.input_type}>, tensor<{output_count}x{args.output_type}>"
            + (f", tensor<{length}xf32>" if mode == "power" else "")
            + ", i32"
        )
        body += f"""  kernel.launch @{mode}_test[%frames]({operands}) : [i32]({types})
  check.expect.close actual(%output) expected(%expected) atol({bound}) rtol(0.0) nan(different) : tensor<{output_count}x{args.output_type}>
  check.expect.bitwise actual(%input) expected(%original_input) : tensor<{input_count}x{args.input_type}>
  check.expect.bitwise actual(%window) expected(%original_window) : tensor<{length}xf32>
  check.expect.bitwise actual(%prefix) expected(%guard) : tensor<16x{args.output_type}>
  check.expect.bitwise actual(%suffix) expected(%guard) : tensor<16x{args.output_type}>
"""
        if mode == "forward" and frames:
            body += f"  %zero = check.generate.fill value(0.0) : tensor<1x{args.output_type}>\n"
            for frame in range(frames):
                for name, position in (("dc", 1), ("nyquist", length + 1)):
                    offset = (frame * (length + 2) + position) * output_bytes
                    body += f"""  %{name}{frame} = check.tensor.view %output offset({offset}) : tensor<{output_count}x{args.output_type}> -> tensor<1x{args.output_type}>
  check.expect.bitwise actual(%{name}{frame}) expected(%zero) : tensor<1x{args.output_type}>
"""
        cases.append(body + "  check.return\n}\n")
    args.stem.with_suffix(".loom").write_text(
        "kernel.decl @forward_test(%frames: i32) launch(%input: buffer, %output: buffer, %frames: i32)\n"
        "kernel.decl @power_test(%frames: i32) launch(%input: buffer, %output: buffer, %window: buffer, %frames: i32)\n\n"
        + "\n".join(cases)
    )


if __name__ == "__main__":
    main()
