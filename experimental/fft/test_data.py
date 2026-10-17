# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small independent direct-DFT fixtures for the real FFT contract."""

import argparse
import math
import struct
from pathlib import Path

from experimental.fft import tables


def round_storage(value, kind):
    if kind == "f32":
        return struct.unpack("<f", struct.pack("<f", value))[0]
    if kind == "f16":
        return struct.unpack("<e", struct.pack("<e", value))[0]
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def write_array(path, values, kind):
    # NPY v1 has a small documented header followed by the flat storage payload.
    descriptor = {"f32": "<f4", "f16": "<f2", "bf16": "<i2"}[kind]
    header = repr(dict(descr=descriptor, fortran_order=False, shape=(len(values),)))
    header += " " * ((-10 - len(header) - 1) % 64) + "\n"
    if kind == "bf16":
        words = [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in values]
        payload = struct.pack(f"<{len(words)}H", *words)
    else:
        payload = struct.pack(f"<{len(values)}{'f' if kind == 'f32' else 'e'}", *values)
    path.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header.encode()
        + payload
    )


def load(name, path, count, kind):
    if kind == "bf16":
        return f'''    %{name}_bits = check.file.read.npy path("{path.name}") : tensor<{count}xi16>
    %{name} = check.tensor.view %{name}_bits offset(0) : tensor<{count}xi16> -> tensor<{count}xbf16>
'''
    return f'    %{name} = check.file.read.npy path("{path.name}") : tensor<{count}x{kind}>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", type=Path)
    parser.add_argument("length", type=int)
    parser.add_argument("width", type=int)
    parser.add_argument("frames", type=int)
    parser.add_argument("input_type", choices=("f32", "f16", "bf16"))
    parser.add_argument("output_type", choices=("f32", "f16", "bf16"))
    parser.add_argument("--frequency-only", action="store_true")
    args = parser.parse_args()
    length, frames = args.length, args.frames
    table = (
        tables.frequency_table(length)
        if args.frequency_only
        else tables.transform_table(length)
    )
    table_path = args.stem.with_name(args.stem.name + "-table.npy")
    write_array(table_path, table, "f32")
    records = []
    for operation in ("forward", "inverse"):
        inputs, expected = [], []
        for frame in range(frames):
            count = length if operation == "forward" else length + 2
            values = [
                round_storage(((p * 17 + frame * 11) % 31 - 15) / 19, args.input_type)
                for p in range(count)
            ]
            if operation == "inverse":
                values[1] = math.nan
                values[-1] = math.nan
            inputs.extend(values)
            if operation == "forward":
                for frequency in range(length // 2 + 1):
                    phases = [math.tau * frequency * p / length for p in range(length)]
                    expected.append(
                        math.fsum(v * math.cos(a) for v, a in zip(values, phases))
                    )
                    expected.append(
                        0.0
                        if frequency in (0, length // 2)
                        else -math.fsum(v * math.sin(a) for v, a in zip(values, phases))
                    )
            else:
                for position in range(length):
                    terms = [values[0], values[length] * (-1 if position % 2 else 1)]
                    for frequency in range(1, length // 2):
                        phase = math.tau * frequency * position / length
                        terms.append(
                            2
                            * (
                                values[2 * frequency] * math.cos(phase)
                                - values[2 * frequency + 1] * math.sin(phase)
                            )
                        )
                    expected.append(math.fsum(terms) / length)
        inputs += [-123.0] * 16
        expected = [round_storage(value, args.output_type) for value in expected] + [
            -123.0
        ] * 16
        input_path = args.stem.with_name(f"{args.stem.name}-{operation}-input.npy")
        expected_path = args.stem.with_name(
            f"{args.stem.name}-{operation}-expected.npy"
        )
        write_array(input_path, inputs, args.input_type)
        write_array(expected_path, expected, args.output_type)
        input_count, output_count = len(inputs), len(expected)
        workspace_count = 2 * length * args.width
        output_bytes = 4 if args.output_type == "f32" else 2
        body = f"""check.scenario @{operation} {{
  check.trial[1](%trial: index, %entropy: check.entropy) {{
"""
        for name, path, count, kind in (
            ("input", input_path, input_count, args.input_type),
            ("original_input", input_path, input_count, args.input_type),
            ("expected", expected_path, output_count, args.output_type),
            ("table", table_path, len(table), "f32"),
            ("original_table", table_path, len(table), "f32"),
        ):
            body += load(name, path, count, kind)
        body += f"""    %output_storage = check.generate.fill value(-123.0) : tensor<{output_count + 16}x{args.output_type}>
    %output = check.tensor.view %output_storage offset({16 * output_bytes}) : tensor<{output_count + 16}x{args.output_type}> -> tensor<{output_count}x{args.output_type}>
    %output_prefix = check.tensor.view %output_storage offset(0) : tensor<{output_count + 16}x{args.output_type}> -> tensor<16x{args.output_type}>
    %output_suffix = check.tensor.view %output offset({(output_count - 16) * output_bytes}) : tensor<{output_count}x{args.output_type}> -> tensor<16x{args.output_type}>
    %output_guard = check.generate.fill value(-123.0) : tensor<16x{args.output_type}>
    %workspace_storage = check.generate.fill value(-123.0) : tensor<{workspace_count + 32}xf32>
    %workspace = check.tensor.view %workspace_storage offset(64) : tensor<{workspace_count + 32}xf32> -> tensor<{workspace_count}xf32>
    %workspace_prefix = check.tensor.view %workspace_storage offset(0) : tensor<{workspace_count + 32}xf32> -> tensor<16xf32>
    %workspace_suffix = check.tensor.view %workspace_storage offset({4 * (workspace_count + 16)}) : tensor<{workspace_count + 32}xf32> -> tensor<16xf32>
    %guard = check.generate.fill value(-123.0) : tensor<16xf32>
"""
        if frames and operation == "forward":
            body += f"""    %dc_imag = check.tensor.view %output offset({output_bytes}) : tensor<{output_count}x{args.output_type}> -> tensor<1x{args.output_type}>
    %nyquist_imag = check.tensor.view %output offset({(length + 1) * output_bytes}) : tensor<{output_count}x{args.output_type}> -> tensor<1x{args.output_type}>
    %zero = check.generate.fill value(0.0) : tensor<1x{args.output_type}>
"""
        if not frames:
            body += f"    %original_workspace = check.generate.fill value(-123.0) : tensor<{workspace_count + 32}xf32>\n"
        body += f"""
    %frames = check.literal value({frames}) : i32
    check.compare<@{operation}_test>(%input, %output, %table, %workspace, %frames) : (tensor<{input_count}x{args.input_type}>, tensor<{output_count}x{args.output_type}>, tensor<{len(table)}xf32>, tensor<{workspace_count}xf32>, i32) -> () {{
"""
        tolerance = {"f32": 2e-5, "f16": 2**-10 + 2e-5, "bf16": 2**-7 + 2e-5}[
            args.output_type
        ]
        for other in ("output", "expected"):
            body += f"      check.expect.close actual(%output) expected(%{other}) atol(0.00003) rtol({tolerance}) nan(different) : tensor<{output_count}x{args.output_type}>\n"
        if frames and operation == "forward":
            for endpoint in ("dc_imag", "nyquist_imag"):
                body += f"      check.expect.bitwise actual(%{endpoint}) expected(%zero) : tensor<1x{args.output_type}>\n"
        if not frames:
            body += f"      check.expect.bitwise actual(%workspace_storage) expected(%original_workspace) : tensor<{workspace_count + 32}xf32>\n"
        for actual, other, count, kind in (
            ("input", "original_input", input_count, args.input_type),
            ("table", "original_table", len(table), "f32"),
            ("output_prefix", "output_guard", 16, args.output_type),
            ("output_suffix", "output_guard", 16, args.output_type),
            ("workspace_prefix", "guard", 16, "f32"),
            ("workspace_suffix", "guard", 16, "f32"),
        ):
            body += f"      check.expect.bitwise actual(%{actual}) expected(%{other}) : tensor<{count}x{kind}>\n"
        body += "    }\n  }\n  check.return\n}\n"
        records.append(body)
    declarations = "\n".join(
        f"func.decl @{operation}_test(%input: buffer, %output: buffer, %table: buffer, %workspace: buffer, %frames: i32)"
        for operation in ("forward", "inverse")
    )
    args.stem.with_suffix(".loom").write_text(
        declarations + "\n\n" + "\n".join(records)
    )


if __name__ == "__main__":
    main()
