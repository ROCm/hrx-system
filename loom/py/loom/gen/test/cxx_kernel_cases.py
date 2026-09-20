# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C++ checks and kernel inputs with independent numerical references."""

import argparse
import math
import os
import random
import struct
from pathlib import Path

ELEMENTS = {"f16": ("e", "<f2", 2), "f32": ("f", "<f4", 4), "i8": ("b", "|i1", 1), "i16": ("h", "<i2", 2), "i32": ("i", "<i4", 4), "i64": ("q", "<i8", 8)}

BYTE_INPUTS = [0, 127, 128, 254, 255, 511]
WIDE_INPUTS = [0, 126, 254, 255, (1 << 32) - 1, 1 << 32, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]


def signed_bits(value, width):
    """Represent an integer bit pattern in signed literal/NumPy storage."""
    return (value + (1 << (width - 1))) % (1 << width) - (1 << (width - 1))


def rounded(value, element):
    code, _, _ = ELEMENTS[element]
    return struct.unpack("<" + code, struct.pack("<" + code, value))[0]


def write_npy(path, values, element):
    code, description, _ = ELEMENTS[element]
    header = repr({"descr": description, "fortran_order": False, "shape": (len(values),)})
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("ascii") + struct.pack(f"<{len(values)}{code}", *values))


class Arrays:
    """Owns fixture output files and their source-relative read paths."""

    def __init__(self, directory, source_directory):
        self.directory = directory
        self.source_directory = source_directory
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, filename, values, element):
        path = self.directory / filename
        write_npy(path, values, element)
        return Path(os.path.relpath(path, self.source_directory)).as_posix()


class Case:
    def __init__(self, arrays, name, element, count):
        self.arrays = arrays
        self.name = name
        self.element = element
        self.count = count
        self.guard = "-123" if element.startswith("i") else "-123.0"
        self.lines = [f"check.case public @{name} {{"]
        self.lines.append(f"  %storage = check.generate.fill value({self.guard}) : tensor<{count + 32}x{element}>")
        _, _, width = ELEMENTS[element]
        self.lines.append(f"  %output = check.tensor.view %storage offset({16 * width}) : tensor<{count + 32}x{element}> -> tensor<{count}x{element}>")

    def array(self, name, values):
        filename = f"{self.name}_{name}.npy"
        filename = self.arrays.write(filename, values, self.element)
        self.lines.append(f'  %{name} = check.file.read.npy path("{filename}") : tensor<{len(values)}x{self.element}>')

    def scalar(self, name, value, element):
        self.lines.append(f"  %{name} = check.literal value({value}) : {element}")

    def launch(self, kernel, arguments, types):
        self.lines.append(f"  kernel.launch @{kernel}({arguments}) : ({types})")

    def finish(self, expected, tolerance=None):
        self.array("expected", expected)
        if tolerance is None:
            self.lines.append(f"  check.expect.bitwise actual(%output) expected(%expected) : tensor<{self.count}x{self.element}>")
        else:
            self.lines.append(f"  check.expect.close actual(%output) expected(%expected) atol({tolerance}) rtol({tolerance}) nan(same) : tensor<{self.count}x{self.element}>")
        _, _, width = ELEMENTS[self.element]
        for name, offset in [("prefix", 0), ("suffix", (self.count + 16) * width)]:
            self.lines.append(f"  %{name} = check.tensor.view %storage offset({offset}) : tensor<{self.count + 32}x{self.element}> -> tensor<16x{self.element}>")
        self.lines.append(f"  %guard = check.generate.fill value({self.guard}) : tensor<16x{self.element}>")
        for name in ["prefix", "suffix"]:
            self.lines.append(f"  check.expect.bitwise actual(%{name}) expected(%guard) : tensor<16x{self.element}>")
        self.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        self.lines.extend(["  check.return", "}", ""])
        return "\n".join(self.lines)


def attention(arrays):
    cases = []
    for queries, keys, magnitude in [(1, 1, 1), (3, 17, 1), (5, 33, 16)]:
        rng = random.Random(730 + queries + keys)
        query = [rounded(rng.uniform(-magnitude, magnitude), "f32") for _ in range(queries * 64)]
        key = [rounded(rng.uniform(-magnitude, magnitude), "f32") for _ in range(keys * 64)]
        value = [rounded(rng.uniform(-2, 2), "f32") for _ in range(keys * 64)]
        expected = []
        for row in range(queries):
            scores = [math.fsum(query[row * 64 + channel] * key[item * 64 + channel] for channel in range(64)) * 0.125 for item in range(keys)]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = math.fsum(weights)
            expected.extend(math.fsum(weights[item] * value[item * 64 + channel] for item in range(keys)) / denominator for channel in range(64))
        case = Case(arrays, f"attention_{queries}_{keys}_{magnitude}", "f32", queries * 64)
        for name, values in [("query", query), ("key", key), ("value", value)]:
            case.array(name, values)
        case.scalar("queries", queries, "i32")
        case.scalar("keys", keys, "i32")
        case.launch(
            "flash_attention", "%query, %key, %value, %output, %queries, %keys", f"tensor<{len(query)}xf32>, tensor<{len(key)}xf32>, tensor<{len(value)}xf32>, tensor<{len(expected)}xf32>, i32, i32"
        )
        cases.append(case.finish(expected, 0.0003))
    return "kernel.decl @flash_attention() launch(%query: buffer, %key: buffer, %value: buffer, %output: buffer, %query_count: i32, %key_count: i32)\n\n" + "\n".join(cases)


def rms_norm(arrays):
    cases = []
    for columns in [1, 33, 129]:
        rng = random.Random(810 + columns)
        values = [rounded(rng.uniform(-2, 2), "f32") for _ in range(3 * columns)]
        epsilon = rounded(1e-5, "f32")
        expected = []
        for row in range(3):
            inputs = values[row * columns : (row + 1) * columns]
            scale = 1 / math.sqrt(math.fsum(value * value for value in inputs) / columns + epsilon)
            expected.extend(value * scale for value in inputs)
        case = Case(arrays, f"rms_norm_{columns}", "f32", len(values))
        case.array("input", values)
        case.scalar("columns", columns, "i32")
        case.scalar("epsilon", epsilon, "f32")
        case.launch("llama_rms_norm", "%input, %output, %columns, %epsilon", f"tensor<{len(values)}xf32>, tensor<{len(values)}xf32>, i32, f32")
        cases.append(case.finish(expected, 0.0002))
    return "kernel.decl @llama_rms_norm() launch(%input: buffer, %output: buffer, %columns: i32, %epsilon: f32)\n\n" + "\n".join(cases)


def swiglu(arrays):
    cases = []
    for columns in [1, 31, 65, 129]:
        rng = random.Random(920 + columns)
        values = [rounded(rng.uniform(-16, 16), "f16") for _ in range(6 * columns)]
        expected = []
        alpha = rounded(1.702, "f32")
        for row in range(3):
            for column in range(columns):
                gate = min(values[row * 2 * columns + column], 7)
                linear = min(max(values[row * 2 * columns + columns + column], -7), 7)
                expected.append(rounded(gate / (1 + math.exp(-alpha * gate)) * (linear + 1), "f16"))
        case = Case(arrays, f"swiglu_f16_{columns}", "f16", len(expected))
        case.array("input", values)
        case.scalar("columns", columns, "i32")
        case.launch("aiter_swiglu_f16", "%input, %output, %columns", f"tensor<{len(values)}xf16>, tensor<{len(expected)}xf16>, i32")
        cases.append(case.finish(expected, 0.002))
    return "kernel.decl @aiter_swiglu_f16() launch(%input: buffer, %output: buffer, %columns: i32)\n\n" + "\n".join(cases)


def control_flow(arrays):
    counts = [0, 1, 2, 7, 31, 64, 129]
    expected = []
    for count in counts:
        trips = max(1, count)
        inner = max(1, count & 3)
        nested = inner * count * (count - 1) // 2 + count * inner * (inner + 1) // 2
        expected.extend([count * (count + 1) // 2, count, trips * (trips + 1) // 2, trips, nested, count, trips, max(0, count - 1), trips, trips])
    case = Case(arrays, "loop_semantics", "i32", len(expected))
    case.array("counts", counts)
    case.scalar("length", len(counts), "i32")
    case.launch("control_flow", "%counts, %output, %length", f"tensor<{len(counts)}xi32>, tensor<{len(expected)}xi32>, i32")
    return "kernel.decl @control_flow() launch(%counts: buffer, %output: buffer, %length: i32)\n\n" + case.finish(expected)


def scheduled_sum_variant(arrays, kernel):
    cases = []
    rows = 7
    for columns in [0, 1, 2, 5, 17, 33]:
        rng = random.Random(1030 + columns)
        values = [rng.randrange(-100, 101) for _ in range(rows * max(1, columns))]
        expected = [sum(values[row * columns : (row + 1) * columns]) for row in range(rows)]
        case = Case(arrays, f"{kernel}_{columns}", "i32", rows)
        case.array("input", values)
        case.array("original", values)
        case.scalar("rows", rows, "i32")
        case.scalar("columns", columns, "i32")
        case.launch(kernel, "%input, %output, %rows, %columns", f"tensor<{len(values)}xi32>, tensor<{rows}xi32>, i32, i32")
        case.lines.append(f"  check.expect.bitwise actual(%input) expected(%original) : tensor<{len(values)}xi32>")
        cases.append(case.finish(expected))
    return f"kernel.decl @{kernel}() launch(%input: buffer, %output: buffer, %rows: i32, %columns: i32)\n\n" + "\n".join(cases)


def short_circuit(arrays):
    cases = []
    for length in [0, 1, 17, 33, 64]:
        values = [((index * 7) % 9) - 3 for index in range(max(1, length))]
        expected = []
        for lane in range(64):
            present = lane < length
            truth = present and values[lane] != 0
            first = (lane & 1) != 0
            second = (lane & 2) != 0
            expected.extend(
                [
                    int(present),
                    int(truth),
                    2 * int(present),
                    int(not present or truth),
                    12 if first else 1,
                    int(first and second),
                    1 if first else 12,
                    int(first or second),
                    4 * int((lane & 8) != 0),
                    int(lane != 31 and lane != 32),
                    next(trace for bound, trace in [(2, 123456), (4, 12345), (8, 1234), (16, 123), (32, 12), (64, 1)] if lane < bound),
                    int(lane == 0),
                ]
            )
        case = Case(arrays, f"short_circuit_{length}", "i32", len(expected))
        case.array("input", values)
        case.scalar("length", length, "i32")
        case.launch("short_circuit", "%input, %output, %length", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish(expected))
    return "kernel.decl @short_circuit() launch(%input: buffer, %output: buffer, %length: i32)\n\n" + "\n".join(cases)


def early_returns(arrays):
    cases = []
    for length in [0, 1, 17, 33, 64]:
        values = [((index * 7) % 9) - 3 for index in range(max(1, length))]
        expected = [-123] * (64 * 6)
        for lane in range(length):
            value = values[lane]
            next_value = values[lane + 1] if lane + 1 < length else None
            classified = -7 if next_value is None else abs(next_value) if next_value < 0 else 3 if next_value == 0 else next_value + 4
            trace = 1 if next_value is None else 3 if next_value == 0 else 2
            total = sum(values[lane : min(lane + 3, length)]) - 7 * max(0, lane + 3 - length)
            chosen = (-11 if lane & 1 else -13) if value < 0 else (17 if lane & 1 else 19)
            published = 11 if value < 0 else value
            state = value + 1 if value < 0 else 5 if value == 0 else value + 7
            expected[lane * 6 : (lane + 1) * 6] = [classified, trace, total, chosen, published, state]
        case = Case(arrays, f"early_returns_{length}", "i32", len(expected))
        case.array("input", values)
        case.scalar("length", length, "i32")
        case.launch("early_returns", "%input, %output, %length", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish(expected))
    return "kernel.decl @early_returns() launch(%input: buffer, %output: buffer, %length: i32)\n\n" + "\n".join(cases)


def integer_increment(arrays, width, inputs):
    cases = []
    argument_width = 32 if width == 8 else 64
    for input_value in inputs:
        expected = []
        for lane in range(64):
            expected.extend([signed_bits(input_value + 1, width), signed_bits(input_value + lane + 1, width)])
        case = Case(arrays, f"increment_u{width}_{input_value}", f"i{width}", len(expected))
        case.scalar("input", signed_bits(input_value, argument_width), f"i{argument_width}")
        case.launch(f"increment_u{width}", "%output, %input", f"tensor<{len(expected)}xi{width}>, i{argument_width}")
        cases.append(case.finish(expected))
    return f"kernel.decl @increment_u{width}() launch(%output: buffer, %input: i{argument_width})\n\n" + "\n".join(cases)


def function_cases(name, argument_widths, result_width, samples, *, argument_types=None):
    """Emit source calls with the oracle's exact argument and expected bits."""
    cases = []
    for ordinal, (arguments, expected) in enumerate(samples):
        # Negative enum inputs retain their numeric value before the enum cast.
        # The positive literal remains representable even for INT64_MIN.
        operands = [f"(-{-value - 1}LL - 1)" if value < 0 else f"0x{value % (1 << width):x}ULL" for value, width in zip(arguments, argument_widths, strict=True)]
        if argument_types is not None:
            operands = [f"static_cast<{kind}>({value})" for kind, value in zip(argument_types, operands, strict=True)]
        operands = ", ".join(operands)
        expected_bits = expected % (1 << result_width)
        cases.append(
            f"LOOM_CHECK_CASE({name}_{ordinal}) {{\n  const auto actual = {name}({operands});\n  loom::check::expect_equal(actual,\n      static_cast<decltype(actual)>(0x{expected_bits:x}ULL));\n}}"
        )
    return "\n\n".join(cases)


INCREMENT_INPUTS = [0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 254, 255, 256, 0x7FFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF]


def increment_byte_reference(value):
    first, second = value % 256, (value + 2) % 256
    return first | (second << 8) | (second << 16) | (first << 24)


def increment_select_reference(value, choose):
    first, second = value % 256, (value * 3 + 1) % 256
    if choose & 1:
        selected, first = first, (first + 1) % 256
    else:
        second = (second - 1) % 256
        selected = second
    if choose & 2:
        if choose & 4:
            first = (first + 1) % 256
            nested = first
        else:
            nested, second = second, (second - 1) % 256
    else:
        nested, first = first, (first - 1) % 256
    return selected | (nested << 8) | (first << 16) | (second << 24)


def increment_chain_reference(value):
    return sum(value < bound for bound in [256, 128, 64, 32, 16, 8]) + 256 * int(value < 4)


def increment_wide_reference(value):
    return (value ^ ((value + 2) * 17) ^ ((value + 2) * 3)) % (1 << 64)


def increment_condition_reference(value):
    selected, final = (value + 2, value + 2) if value & 1 else (value + 1, value)
    return (selected ^ (final * 17)) % (1 << 32)


def increment_functions():
    counts = [0, 1, 2, 3, 7, 8, 15, 16, 31, 32]
    functions = [
        ("increment_byte", [32], 32, [([value], increment_byte_reference(value)) for value in range(256)]),
        ("decrement_short", [32], 32, [([value], value * 65536 + value - 2 + 32768) for value in [-32766, -129, -1, 0, 1, 128, 32767]]),
        ("increment_wide", [64], 64, [([value], increment_wide_reference(value)) for value in WIDE_INPUTS]),
        ("increment_chain", [32], 32, [([value], increment_chain_reference(value)) for value in INCREMENT_INPUTS]),
        ("increment_or", [32, 32], 32, [([value, choose], (value + int(not choose)) * 2 + int(choose or value < 128)) for value in INCREMENT_INPUTS for choose in [0, 1]]),
        ("increment_select", [32, 32], 32, [([value, choose], increment_select_reference(value, choose)) for value in INCREMENT_INPUTS for choose in range(8)]),
        ("increment_condition", [32], 32, [([value], increment_condition_reference(value)) for value in INCREMENT_INPUTS]),
        ("increment_while", [32], 32, [([count], count * (count + 1) // 2 * 257 + count + 1) for count in counts]),
        ("increment_do", [32], 32, [([count], max(1, count) * (max(1, count) - 1) // 2 * 257 + max(1, count)) for count in counts]),
    ]
    return "\n\n".join(function_cases(name, widths, result_width, samples) for name, widths, result_width, samples in functions)


def increment_values(arrays):
    cases = []
    for input_value in [0, 1, 127, 255, 256, 0x7FFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF]:
        expected = []
        for lane in range(64):
            value = (input_value + lane) % (1 << 32)
            wide = increment_wide_reference((input_value * 4294967297 + lane) % (1 << 64))
            count = lane % 8
            expected.extend(
                [
                    increment_byte_reference(value),
                    (lane - 31) * 65536 + lane - 33 + 32768,
                    wide % (1 << 32),
                    wide >> 32,
                    increment_chain_reference(value),
                    (value + int(lane % 2 == 0)) * 2 + int(lane % 2 or value < 128),
                    increment_select_reference(value, lane % 8),
                    increment_condition_reference(value),
                    count * (count + 1) // 2 * 257 + count + 1,
                    max(1, count) * (max(1, count) - 1) // 2 * 257 + max(1, count),
                ]
            )
        case = Case(arrays, f"increment_values_{input_value}", "i32", len(expected))
        case.scalar("input", signed_bits(input_value, 32), "i32")
        case.launch("increment_values", "%output, %input", f"tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish([signed_bits(value, 32) for value in expected]))
    return "kernel.decl @increment_values() launch(%output: buffer, %input: i32)\n\n" + "\n".join(cases)


def increment_pointers(arrays):
    cases = []
    for length, choose in [(0, 0), (1, 0), (17, 1), (33, 2), *[(64, choose) for choose in range(3, 8)]]:
        values = [(index * 17) % 31 - 15 for index in range(max(1, length) * 8)]
        expected = [-123] * (64 * 12)
        for lane in range(length):
            inputs = values[lane * 8 : (lane + 1) * 8]
            cursor = 2 + int(not choose & 2)
            accepted = bool(choose & 2 or inputs[2])
            guarded = bool(choose & 4 and inputs[cursor])
            final = cursor + int(bool(choose & 4))
            expected[lane * 12 : (lane + 1) * 12] = [
                inputs[1],
                inputs[3],
                inputs[3],
                inputs[1],
                inputs[1 if choose & 1 else 2],
                inputs[2],
                int(accepted),
                inputs[cursor],
                int(guarded),
                inputs[final],
                inputs[final] + inputs[final + 1],
                inputs[final + 2],
            ]
        case = Case(arrays, f"increment_pointers_{length}_{choose}", "i32", len(expected))
        case.array("input", values)
        case.scalar("length", length, "i32")
        case.scalar("choose", choose, "i32")
        case.launch("increment_pointers", "%input, %output, %length, %choose", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32, i32")
        case.array("input_expected", values)
        case.lines.append(f"  check.expect.bitwise actual(%input) expected(%input_expected) : tensor<{len(values)}xi32>")
        cases.append(case.finish(expected))
    return "kernel.decl @increment_pointers() launch(%input: buffer, %output: buffer, %length: i32, %choose: i32)\n\n" + "\n".join(cases)


def continue_references(count, choose):
    indices = range(count)
    selected = [index for index in indices if not index & choose]
    post_selected = [index for index in range(1, count + 1) if not index & choose]
    branch_total = sum(1 + (2 if index & choose else 8) for index in indices)
    branch_total += sum((4 if index & choose else 16) + 32 + index for index in indices if not index & (2 if index & choose else 4))
    post_count = max(1, count)
    nested = sum(outer * 7 + inner for outer in indices for inner in range(5) if (outer + inner) % 2 == 0)
    scoped = [index + (11 if index & choose else 23) for index in indices if not (index + (11 if index & choose else 23)) & (1 if index & choose else 2)]
    return {
        "continue_branches": branch_total,
        "continue_for_step": sum(post_selected) + count * 257 + (count + 1) * 65537,
        "continue_while": sum(post_selected) + (count + 1) * 257,
        "continue_do": sum(index for index in range(post_count) if not index & choose) + post_count * (257 + 65537),
        "continue_nested": nested + 100 * len(selected),
        "continue_all": count * (count - 1) // 2 + (count + 2) * 257,
        "continue_scopes": sum(scoped) + len(scoped) * (len(scoped) + 1) // 2 + (len(scoped) + 1) * 257,
        "continue_byte": (250 + count + 7 * len(selected)) % 256,
    }


def continue_functions():
    counts = [0, 1, 2, 3, 7, 16, 31]
    cases = []
    for name in continue_references(0, 0):
        if name == "continue_all":
            samples = [([count], continue_references(count, 0)[name]) for count in counts]
            widths = [32]
        else:
            samples = [([count, choose], continue_references(count, choose)[name]) for count in counts for choose in [0, 1, 2, 3, 5, 7, 31]]
            widths = [32, 32]
        cases.append(function_cases(name, widths, 32, samples))
    return "\n".join(cases)


def continue_values(arrays):
    cases = []
    for count in [0, 1, 7, 31]:
        for choose in [0, 3, 7]:
            expected = []
            for lane in range(64):
                length, mask = count + lane % 4, choose ^ (lane % 8)
                expected.extend(continue_references(length, mask).values())
            case = Case(arrays, f"continue_values_{count}_{choose}", "i32", len(expected))
            case.scalar("count", count, "i32")
            case.scalar("choose", choose, "i32")
            case.launch("continue_values", "%output, %count, %choose", f"tensor<{len(expected)}xi32>, i32, i32")
            cases.append(case.finish(expected))
    return "kernel.decl @continue_values() launch(%output: buffer, %count: i32, %choose: i32)\n\n" + "\n".join(cases)


def continue_scheduled(arrays):
    cases = []
    for count in [0, 1, 2, 5, 17, 33]:
        for choose in [0, 3]:
            values = [(index * 17 + 7) % 251 for index in range(max(1, count * 64))]
            expected = []
            for lane in range(64):
                selected = sum(values[lane * count + index] for index in range(count) if not index & (choose ^ (lane % 8)))
                expected.extend([selected] * 4)
            case = Case(arrays, f"continue_scheduled_{count}_{choose}", "i32", len(expected))
            case.array("input", values)
            case.array("original", values)
            case.scalar("count", count, "i32")
            case.scalar("choose", choose, "i32")
            case.launch("continue_scheduled", "%input, %output, %count, %choose", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32, i32")
            case.lines.append(f"  check.expect.bitwise actual(%input) expected(%original) : tensor<{len(values)}xi32>")
            cases.append(case.finish(expected))
    return "kernel.decl @continue_scheduled() launch(%input: buffer, %output: buffer, %count: i32, %choose: i32)\n\n" + "\n".join(cases)


def continue_copy(arrays):
    values = [index * 7 + 3 for index in range(64 * 16)]
    expected = [value if index % 2 else -123 for index, value in enumerate(values)]
    case = Case(arrays, "copy_odd_indices", "i32", len(expected))
    case.array("input", values)
    case.array("original", values)
    case.launch("continue_copy", "%input, %output", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>")
    case.lines.append(f"  check.expect.bitwise actual(%input) expected(%original) : tensor<{len(values)}xi32>")
    return "kernel.decl @continue_copy() launch(%input: buffer, %output: buffer)\n\n" + case.finish(expected)


def continue_pointers(arrays):
    cases = []
    for length in [0, 1, 17, 32, 33]:
        for choose in [0, 1, 7]:
            values = [(index * 7) % 37 for index in range(max(1, length * 64))]
            expected = [-123] * (34 * 64)
            for lane in range(64):
                selected = [value for value in values[lane * length : (lane + 1) * length] if value & choose]
                expected[lane * 34 : lane * 34 + len(selected)] = selected
                expected[lane * 34 + 33] = len(selected)
            case = Case(arrays, f"continue_pointers_{length}_{choose}", "i32", len(expected))
            case.array("input", values)
            case.array("original", values)
            case.scalar("length", length, "i32")
            case.scalar("choose", choose, "i32")
            case.launch("continue_pointers", "%input, %output, %length, %choose", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32, i32")
            case.lines.append(f"  check.expect.bitwise actual(%input) expected(%original) : tensor<{len(values)}xi32>")
            cases.append(case.finish(expected))
    return "kernel.decl @continue_pointers() launch(%input: buffer, %output: buffer, %length: i32, %choose: i32)\n\n" + "\n".join(cases)


def continue_vectors(arrays):
    cases = []
    for count in [0, 1, 2, 7, 31]:
        for choose in [0, 1, 7]:
            expected = [1, 2, 3, 4]
            for index in range(count):
                expected = [value + index * (lane + 1) for lane, value in enumerate(expected)]
                if not index & choose:
                    expected = [value ^ mask for value, mask in zip(expected, [17, 31, 63, 127], strict=True)]
            case = Case(arrays, f"continue_vectors_{count}_{choose}", "i32", len(expected))
            case.scalar("count", count, "i32")
            case.scalar("choose", choose, "i32")
            case.launch("continue_vectors", "%output, %count, %choose", "tensor<4xi32>, i32, i32")
            cases.append(case.finish(expected))
    return "kernel.decl @continue_vectors() launch(%output: buffer, %count: i32, %choose: i32)\n\n" + "\n".join(cases)


CONSTANT_LOOP_STARTS = [0, 1, 2, 3, 7, 16, 17, 18, 19, 20, 21, 0x80000000, 0xFFFFFFFF]


def schedule_functions():
    cases = []
    for name in ["call", "snapshot", "wide", "narrow", "signed", "unevaluated", "initializer", "serial"]:
        samples = []
        for count in [0, 1, 2, 3, 4, 5, 7, 16, 17, 33]:
            expected = sum(range(count))
            if name == "snapshot":
                expected += count + 3
            elif name == "unevaluated":
                expected += count + 8 + ord("A")
            samples.append(([count], expected))
        cases.append(function_cases(f"schedule_{name}", [32], 32, samples))
    return "\n".join(cases)


def constant_loop_functions():
    cases = [
        ("counted_stride", [([start], sum(range(start, 20, 4))) for start in CONSTANT_LOOP_STARTS]),
        ("counted_empty", [([start], 7) for start in CONSTANT_LOOP_STARTS]),
        ("counted_maximum_step", [([start], 7 if start == 0 else 0) for start in CONSTANT_LOOP_STARTS]),
        ("counted_edge", [([start], sum(range(start, 0xFFFFFFFC, 4))) for start in range(0xFFFFFFF0, 0x100000000)]),
    ]
    return "\n".join(function_cases(name, [32], 32, samples) for name, samples in cases)


def constant_loops(arrays):
    values = [(index * 17 + 7) % 251 for index in range(64 * 20)]
    cases = []
    for start in CONSTANT_LOOP_STARTS:
        expected = []
        for lane in range(64):
            row = values[lane * 20 : (lane + 1) * 20]
            expected.extend([sum(row[start:20:2])] * 2)
            expected.extend([sum(row[0:20:2])] * 2)
            expected.append(sum(row[0:20:4]))
            expected.append(sum(row))
            iterations = max(0, (20 - start) // 2)
            expected.append(sum(row[start : start + iterations]) + 19 - iterations)
            expected.append(signed_bits(sum(range(0xFFFFFFF0 + lane % 8, 0xFFFFFFFC, 4)), 32))
            expected.extend([sum(row[0:bound:2]) for bound in [0, 1, 2, 5]])
        case = Case(arrays, f"constant_loops_{start}", "i32", len(expected))
        case.array("input", values)
        case.array("original", values)
        case.scalar("start", signed_bits(start, 32), "i32")
        case.launch("constant_loops", "%input, %output, %start", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32")
        case.lines.append(f"  check.expect.bitwise actual(%input) expected(%original) : tensor<{len(values)}xi32>")
        cases.append(case.finish(expected))
    return "kernel.decl @constant_loops() launch(%input: buffer, %output: buffer, %start: i32)\n\n" + "\n".join(cases)


def assumption_functions():
    values = [0, 1, 127, 128, 254, 255]
    seven = [[0] * 7, [255] * 7] + [[255 if lane == active else 0 for lane in range(7)] for active in range(7)]
    samples = [
        ("bound_pair", [32, 32], [([a, b], a * 257 + b) for a in values for b in values]),
        ("bound_seven", [32] * 7, [(args, sum(a * b for a, b in zip(args, [1, 2, 3, 5, 7, 11, 13], strict=True))) for args in seven]),
        ("bound_repeated", [32], [([value], value * 17) for value in [0, 1, 15, 16, 31]]),
        ("bound_capacity", [32], [([value], value * 16 + 336) for value in [0, 1, 255, 256, 426, 427]]),
        ("bound_cast", [32], [([value], value + 5) for value in [0, 1, 7, 15]]),
        ("bound_byte", [8], [([value], value + (1024 if value >= 128 else 0)) for value in range(256)]),
        ("bound_wide", [64], [([value], value * 3) for value in values]),
        ("bound_size", [32], [([value], value) for value in [0, 1, 7, 15]]),
        ("bound_scoped", [32], [([value], value + (1 if value < 256 else 3)) for value in [0, 1, 127, 128, 255, 256, 427, 0x7FFFFFFF, 0xFFFFFFFF]]),
    ]
    return "\n".join(function_cases(name, widths, 32, cases) for name, widths, cases in samples)


def assumption_kernel(arrays):
    cases = []
    for input_value in [0, 1, 127, 128, 255, 256, 427, 0xFFFFFFC0, 0xFFFFFFFF]:
        expected = []
        for lane in range(64):
            value = (input_value + lane) % (1 << 32)
            byte = value % 256
            expected.extend(
                [
                    byte * 257 + value // 256 % 256,
                    byte + 278,
                    value % 32 * 17,
                    value % 428 * 16 + 336,
                    value % 16 + 5,
                    byte + (1024 if byte >= 128 else 0),
                    byte * 3,
                    value % 16,
                    signed_bits(value + (1 if value < 256 else 3), 32),
                ]
            )
        case = Case(arrays, f"assumptions_{input_value}", "i32", len(expected))
        case.scalar("input", signed_bits(input_value, 32), "i32")
        case.launch("assumption_kernel", "%output, %input", f"tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish(expected))
    return "kernel.decl @assumption_kernel() launch(%output: buffer, %input: i32)\n\n" + "\n".join(cases)


def integer_functions():
    cases = []

    def function(name, argument_widths, result_width, samples):
        cases.append(function_cases(name, argument_widths, result_width, samples))

    products = [(0, -1), (65536, 65536), (-65537, 98304), (65537, -98304), (-(1 << 31), 65536), ((1 << 31) - 1, 65536), (12345, 6789)]
    function("fixed_multiply", [32, 32], 32, [(pair, pair[0] * pair[1] // 65536) for pair in products])
    function("byte_increment", [32], 32, [([value], (value + 1) % 256) for value in BYTE_INPUTS])
    function("byte_decrement", [32], 32, [([value], (value - 1) % 256) for value in BYTE_INPUTS])
    function("short_decrement", [32], 32, [([value], value - 1) for value in [-32767, -129, -1, 0, 1, 32767]])
    function("wide_increment", [64], 64, [([value], (value + 1) % (1 << 64)) for value in WIDE_INPUTS])
    narrow_values = [0, 1, 0x12345678, (1 << 31), (1 << 32) - 1]
    function("shift_left_narrow", [32, 64], 32, [([value, count], value * (1 << count) % (1 << 32)) for value in narrow_values for count in [0, 1, 16, 31]])
    wide_values = [0, 1, -1, -65537, 0x123456789ABCDEF, -(1 << 63)]
    counts = [0, 1, 16, 31, 32, 63]
    function("shift_left_wide", [64, 32], 64, [([value, count], value * (1 << count) % (1 << 64)) for value in wide_values for count in counts])
    function("shift_right_signed", [64, 32], 64, [([value, count], value // (1 << count)) for value in wide_values for count in counts])
    function("shift_right_unsigned", [64, 32], 64, [([value, count], (value % (1 << 64)) // (1 << count)) for value in wide_values for count in counts])
    return "\n\n".join(cases) + "\n"


def enum_functions():
    cases = []
    commands = [0, 1, 2, 3, 4, 5, 6, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    cases.append(function_cases("enum_dispatch", [32], 32, [([value], 128 if value == 1 else 255 if value >= 5 else value + 7) for value in commands], argument_types=["Command"]))
    cases.append(function_cases("enum_byte", [8], 32, [([value], (value + 1) % 256) for value in range(256)], argument_types=["Byte"]))
    cases.append(function_cases("enum_signed", [8], 64, [([value], value * 65537) for value in [-128, -127, -1, 0, 1, 126, 127]], argument_types=["SignedByte"]))
    cases.append(function_cases("enum_unsigned", [32], 64, [([value], value + 1) for value in commands], argument_types=["Word"]))
    wide = [0, 1, (1 << 32) - 1, 1 << 32, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    cases.append(function_cases("enum_compare64", [64, 64], 32, [([left, right], int(left < right)) for left in wide for right in wide], argument_types=["Long", "Long"]))
    cases.append(function_cases("enum_inferred", [32], 64, [([value], (1 << 40) if value else -1) for value in commands]))
    cases.append(function_cases("enum_inferred_unsigned", [64], 32, [([value], int(value < (1 << 64) - 1)) for value in wide]))
    cases.append(function_cases("enum_specialization", [32], 64, [([value], (1 << 40) + 0xFFFFFFFF + (4 if value else 0)) for value in commands]))
    cases.append(function_cases("enum_packed_unsigned", [8], 32, [([value], value + 1) for value in range(256)], argument_types=["PackedByte"]))
    cases.append(function_cases("enum_packed_signed", [8], 32, [([value], value - 1) for value in range(-128, 128)], argument_types=["PackedSignedByte"]))
    cases.append(function_cases("enum_bool", [32], 32, [([value], int(value != 0)) for value in commands]))
    return "\n\n".join(cases) + "\n"


def enum_storage(arrays, width):
    mask = (1 << width) - 1
    values = [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), mask - 1, mask]
    values += [(index * 0x123456789ABCDEF) & mask for index in range(64 - len(values))]
    cases = []
    for delta in [1, 1 << (width - 1), mask]:
        element = f"i{width}"
        case = Case(arrays, f"enum_storage_u{width}_{delta}", element, len(values))
        case.array("input", [signed_bits(value, width) for value in values])
        case.array("original", [signed_bits(value, width) for value in values])
        case.scalar("delta", signed_bits(delta, width), element)
        case.launch(f"enum_storage_u{width}", "%input, %output, %delta", f"tensor<64x{element}>, tensor<64x{element}>, {element}")
        case.lines.append(f"  check.expect.bitwise actual(%input) expected(%original) : tensor<64x{element}>")
        expected = [signed_bits((value + delta) & mask, width) for value in values]
        cases.append(case.finish(expected))
    return f"kernel.decl @enum_storage_u{width}() launch(%input: buffer, %output: buffer, %delta: i{width})\n\n" + "\n".join(cases)


def comparison_functions():
    samples = []
    for mask in range(128):
        arguments = [256 if mask & (1 << index) else 255 for index in range(7)]
        samples.append((arguments, int(mask == 0)))
    samples.append(([0] * 7, 1))
    for index in range(7):
        arguments = [0] * 7
        arguments[index] = (1 << 32) - 1
        samples.append((arguments, 0))
    return function_cases("comparison_chain", [32] * 7, 32, samples) + "\n"


def pointer_walk(arrays):
    cases = []
    counts = [0, 1, 2, 5, 17, 33, 47]
    for start, displacement in [(1, -1), (7, -2), (31, 3)]:
        values = [(index * 97 + 13) % 257 - 128 for index in range(7 * 64 + 64)]
        expected = []
        for lane, count in enumerate(counts):
            base = start + lane * 64
            selected = base + (2 if lane & 1 else 4)
            trips = max(1, count)
            total = sum(values[base : base + count])
            expected.extend(
                [
                    values[selected - 1],
                    values[selected + displacement],
                    values[base + 2],
                    total,
                    values[base + count],
                    total,
                    values[base],
                    values[base + trips],
                    values[base + trips - 1],
                    values[base + trips - 2],
                ]
            )
        case = Case(arrays, f"pointer_walk_{start}_{abs(displacement)}", "i32", len(expected))
        case.array("input", values)
        case.array("counts", counts)
        case.scalar("length", len(counts), "i32")
        case.scalar("start", start, "i64")
        case.scalar("displacement", displacement, "i64")
        case.launch("pointer_walk", "%input, %counts, %output, %length, %start, %displacement", f"tensor<{len(values)}xi32>, tensor<{len(counts)}xi32>, tensor<{len(expected)}xi32>, i32, i64, i64")
        for name, original in [("input", values), ("counts", counts)]:
            case.array(name + "_expected", original)
            case.lines.append(f"  check.expect.bitwise actual(%{name}) expected(%{name}_expected) : tensor<{len(original)}xi32>")
        cases.append(case.finish(expected))
    return "kernel.decl @pointer_walk() launch(%input: buffer, %counts: buffer, %output: buffer, %length: i32, %start: i64, %displacement: i64)\n\n" + "\n".join(cases)


def vector_depth(arrays):
    cases = []
    for blocks in [1, 3, 7]:
        rng = random.Random(843 + blocks)
        previous = [rng.getrandbits(32) for _ in range(blocks * 16)]
        depth = [rng.getrandbits(32) for _ in previous]
        previous[:4] = [0, 0xFFFFFFFF, 0x12345678, 0xFFFF0000]
        depth[:4] = [0xFFFFFFFF, 0, 0x87654321, 0xFFFF]
        expected = [signed_bits((a & 0xFFFF) | (b & 0xFFFF0000), 32) for a, b in zip(previous, depth, strict=True)]
        case = Case(arrays, f"vector_depth_{blocks}", "i32", len(previous))
        case.array("previous", [signed_bits(value, 32) for value in previous])
        case.array("depth", [signed_bits(value, 32) for value in depth])
        case.scalar("count", blocks, "i32")
        case.launch("vector_depth", "%previous, %depth, %output, %count", f"tensor<{len(previous)}xi32>, tensor<{len(depth)}xi32>, tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish(expected))
    return "kernel.decl @vector_depth() launch(%previous: buffer, %depth: buffer, %output: buffer, %count: i32)\n\n" + "\n".join(cases)


def vector_depth_span(arrays):
    cases = []
    for count in [0, 1, 15, 16, 17, 31, 32, 129]:
        rng = random.Random(281 + count)
        previous = [rng.getrandbits(32) for _ in range(max(1, count))]
        depth, step = 0xFFFF1234, 0x137CF
        expected = [signed_bits((value & 0xFFFF) | ((depth + index * step) & 0xFFFF0000), 32) for index, value in enumerate(previous[:count])]
        case = Case(arrays, f"vector_depth_span_{count}", "i32", count)
        case.array("previous", [signed_bits(value, 32) for value in previous])
        for name, value in [("count", count), ("depth", depth), ("step", step)]:
            case.scalar(name, signed_bits(value, 32), "i32")
        case.launch("vector_depth_span", "%previous, %output, %count, %depth, %step", f"tensor<{len(previous)}xi32>, tensor<{count}xi32>, i32, i32, i32")
        cases.append(case.finish(expected))
    return "kernel.decl @vector_depth_span() launch(%previous: buffer, %output: buffer, %count: i32, %depth: i32, %step: i32)\n\n" + "\n".join(cases)


def vector_values():
    cases = []

    def function(name, samples):
        cases.append(function_cases(name, [32] * len(samples[0][0]), 32, samples))

    function(
        "vector_unsigned", [([value, lane], ((([value, 0x80000000, 0xFFFFFFFF, 7][lane] + 17) & 0xFFFFFFFF) >> 3) // 3) for value in [0, 1, 0x7FFFFFFF, 0xFFFFFFF0, 0xFFFFFFFF] for lane in range(4)]
    )
    pairs = [(0, 0), (0, 0xFFFFFFFF), (0xFFFFFFFF, 0), (0x80000000, 0x7FFFFFFF)]
    function("vector_mask", [([a, b, lane], -int([a, 0xFFFFFFFF, 0, 0x80000000][lane] > [b, 0, 0, 0x7FFFFFFF][lane])) for a, b in pairs for lane in range(4)])
    function("vector_narrow", [([value, lane], (([value & 255, 255, 128, 0] + [0] * 12)[lane] + 200 & 255) >> 1) for value in [0, 55, 56, 127, 255, 511] for lane in [0, 1, 2, 3, 8, 15]])

    def signed_result(value, divisor):
        quotient = abs(value) // abs(divisor) * (-1 if (value < 0) != (divisor < 0) else 1)
        return (quotient >> 1) + value - quotient * divisor

    for width, name in [(8, "vector_signed_byte"), (16, "vector_signed_short")]:
        inputs = [-(1 << (width - 1)), -127, -1, 0, 1, (1 << (width - 1)) - 1]
        function(name, [([value, divisor, lane], signed_result([value, -127, -1, 127][lane], [divisor, 3, -3, 3][lane])) for value in inputs for divisor in [-3, 3] for lane in range(4)])
    function(
        "vector_unsigned_short",
        [([value, lane], ((([value & 65535, 65535, 32768, 0] + [0] * 4)[lane] * 257 + 60000) & 65535) >> 5) for value in [0, 1, 32767, 32768, 65535, 65536] for lane in [0, 1, 2, 3, 7]],
    )
    function(
        "vector_bitcast", [([a, b, lane], [a, b, 0x3F800000, 0x80000000][lane] ^ 0x80000000) for a, b in [(0, 0x80000000), (0x3F000000, 0xBF800000), (0x7F800000, 0xFF800000)] for lane in range(4)]
    )
    float_pairs = [(0, 0x80000000), (0x7FC00000, 0x7FC00000), (0x3F800000, 0x40000000), (0x7F800000, 0x7F800000)]
    function("vector_float_ne", [([a, b, lane], -int(f32_bits([a, 0x7FC00000, 0, 0x80000000][lane]) != f32_bits([b, 0x3F800000, 0x80000000, 0][lane]))) for a, b in float_pairs for lane in range(4)])
    function("vector_ext_splat", [([value, lane], value) for value in [0, 1, 0xFFFFFFFF, 0x80000000, 17] for lane in range(4)])
    return "\n\n".join(cases) + "\n"


def vector_control(arrays):
    cases = []
    for value in [0, 0xFFFFFFFF, 0x7FFFFFFF]:
        for count in [0, 1, 3, 7]:
            expected = [(initial + 4 * count) & 0xFFFFFFFF for initial in [value, 1, 0, 0]]
            if count > 2:
                expected = [~element for element in expected]
            case = Case(arrays, f"vector_control_{value}_{count}", "i32", 4)
            case.scalar("input", signed_bits(value, 32), "i32")
            case.scalar("count", count, "i32")
            case.launch("vector_control_kernel", "%output, %input, %count", "tensor<4xi32>, i32, i32")
            cases.append(case.finish([signed_bits(element, 32) for element in expected]))
    return "kernel.decl @vector_control_kernel() launch(%output: buffer, %input: i32, %count: i32)\n\n" + "\n".join(cases)


CONSTRUCTOR_INPUTS = [0, 1, 127, 255, 256, 0xFFFFFF80, 0x7FFFFFFC, 0xFFFFFFFF]


def constructor_references(value):
    signed = signed_bits(value, 32)
    floating = struct.unpack("<I", struct.pack("<f", float(signed)))[0]
    return {
        "constructor_return": [signed + lane for lane in range(4)],
        "constructor_argument": [value, 7, 0, 0],
        "constructor_single": [value, 0, 0, 0],
        "constructor_narrow": [value & 255, 255, 128] + [0] * 13,
        "constructor_float": [floating, 0x80000000, 0x3FC00000, 0],
        "constructor_nested": [value + 9, 9, 9, 9],
    }


def vector_constructor_values():
    samples = {}
    for value in CONSTRUCTOR_INPUTS:
        for name, lanes in constructor_references(value).items():
            samples.setdefault(name, []).extend(([value, lane], expected) for lane, expected in enumerate(lanes))
    return "\n\n".join(function_cases(name, [32, 32], 32, values) for name, values in samples.items()) + "\n"


def vector_initializers(arrays):
    cases = []
    for value in CONSTRUCTOR_INPUTS:
        expected = []
        for name, lanes in constructor_references(value).items():
            expected.extend(struct.unpack("<4I", bytes(lanes)) if name == "constructor_narrow" else lanes)
        expected.extend(value + lane for lane in range(4))
        expected.extend([1234, 0, 0, 0])
        case = Case(arrays, f"vector_initializers_{value}", "i32", len(expected))
        case.scalar("input", signed_bits(value, 32), "i32")
        case.launch("vector_initializers", "%output, %input", f"tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish([signed_bits(element, 32) for element in expected]))
    return "kernel.decl @vector_initializers() launch(%output: buffer, %input: i32)\n\n" + "\n".join(cases)


def f32_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def vector_masks(arrays):
    cases = []
    for ordinal, (left, right) in enumerate(
        [
            ([0, 0xFFFFFFFF, 0x80000000, 7], [0, 0, 0x7FFFFFFF, 7]),
            ([0, 0x80000000, 0x7FC00000, 0x7F800000], [0x80000000, 0, 0x3F800000, 0x7F800000]),
            ([0x3F800000, 0xFF800000, 0x7FC00000, 0xBF800000], [0x40000000, 0x7F800000, 0x7FC00000, 0xC0000000]),
        ]
    ):
        expected = [-int(a < b) for a, b in zip(left, right, strict=True)]
        expected += [-int(a == b) for a, b in zip(left, right, strict=True)]
        expected += [-int(a == 0) for a in left]
        expected += [-int(signed_bits(a, 32) < signed_bits(b, 32)) for a, b in zip(left, right, strict=True)]
        expected += [-int(f32_bits(a) != f32_bits(b)) for a, b in zip(left, right, strict=True)]
        case = Case(arrays, f"vector_masks_{ordinal}", "i32", len(expected))
        case.array("input", [signed_bits(value, 32) for value in left + right])
        case.launch("vector_masks", "%input, %output", "tensor<8xi32>, tensor<20xi32>")
        cases.append(case.finish(expected))
    return "kernel.decl @vector_masks() launch(%input: buffer, %output: buffer)\n\n" + "\n".join(cases)


def shaped_intrinsic_values():
    lookups = []
    for value in [0, 0x80000000, 0xFFFFFFFF]:
        table = [value, 1, 0xFFFFFFFF, 0x80000000, *range(4, 16)]
        for index in range(16):
            lookups.extend(([value, index, lane], table[[index, 15 - index, 0, 15][lane]]) for lane in range(4))
    dots = []
    rhs = [-128, 127, -1, 1, 127, -128, 1, -1, -3, 5, -7, 11, 127, 127, 127, 127]
    for value in [0, 1, 127, 128, 255, 511]:
        lhs = [value & 255, 255, 128, 127, 0, 1, 255, 128, 23, 45, 67, 89, 255, 255, 255, 255]
        for initial in [0, 0x7FFFFFFF, -0x80000000]:
            accumulators = [initial, -1, 0x7FFFFFFF, -0x80000000]
            for lane in range(4):
                product = sum(lhs[index] * rhs[index] for index in range(4 * lane, 4 * lane + 4))
                dots.append(([value, initial, lane], accumulators[lane] + product))
    return function_cases("lookup_lane", [32, 32, 32], 32, lookups) + "\n" + function_cases("dot_lane", [32, 32, 32], 32, dots)


def register_lookup(arrays, *, floating=False):
    name = "register_lookup_float" if floating else "register_lookup"
    table = (
        [0, 0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000, 1, 0x80000001, 0x3F800000, 0xBF800000, 0x3F000000, 0x40000000, 0xC0000000, 0x7F7FFFFF, 0x00800000, 0x007FFFFF, 0xFFC12345]
        if floating
        else [signed_bits(0x9E3779B9 * index + 0x80000000, 32) for index in range(16)]
    )
    table = [signed_bits(value, 32) for value in table]
    cases = []
    for count in [0, 1, 4, 7]:
        indices = [(index * 11 + 15) % 16 for index in range(max(4, count * 4))]
        expected = [table[index] for index in indices[: count * 4]]
        case = Case(arrays, f"{name}_{count}", "i32", len(expected))
        case.array("table", table)
        case.array("indices", indices)
        case.scalar("count", count, "i32")
        case.launch(name, "%table, %indices, %output, %count", f"tensor<16xi32>, tensor<{len(indices)}xi32>, tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish(expected))
    return f"kernel.decl @{name}() launch(%table: buffer, %indices: buffer, %output: buffer, %count: i32)\n\n" + "\n".join(cases)


def mixed_dot(arrays):
    cases = []
    for count in [0, 1, 3, 9]:
        lhs = [([0, 127, 128, 255][index % 4] + index // 4) & 255 for index in range(max(16, count * 16))]
        rhs = [(255 - index * 17) & 255 for index in range(len(lhs))]
        accumulators = [[0, 0x7FFFFFFF, -0x80000000, -1][index % 4] for index in range(max(4, count * 4))]
        expected = []
        for group in range(count):
            for lhs_signed, rhs_signed in [(True, True), (False, True), (True, False), (False, False)]:
                for lane in range(4):
                    products = []
                    for index in range(16 * group + 4 * lane, 16 * group + 4 * lane + 4):
                        a = signed_bits(lhs[index], 8) if lhs_signed else lhs[index]
                        b = signed_bits(rhs[index], 8) if rhs_signed else rhs[index]
                        products.append(a * b)
                    expected.append(signed_bits(accumulators[4 * group + lane] + sum(products), 32))
        case = Case(arrays, f"mixed_dot_{count}", "i32", len(expected))
        for name, values in [("lhs", lhs), ("rhs", rhs)]:
            packed = struct.unpack(f"<{len(values) // 4}i", bytes(values))
            case.array(name, packed)
        case.array("acc", accumulators)
        case.scalar("count", count, "i32")
        case.launch("mixed_dot", "%lhs, %rhs, %acc, %output, %count", f"tensor<{len(lhs) // 4}xi32>, tensor<{len(rhs) // 4}xi32>, tensor<{len(accumulators)}xi32>, tensor<{len(expected)}xi32>, i32")
        cases.append(case.finish(expected))
    return "kernel.decl @mixed_dot() launch(%lhs: buffer, %rhs: buffer, %acc: buffer, %output: buffer, %count: i32)\n\n" + "\n".join(cases)


def launch_grid(kernel, x, y=1, z=1):
    return "".join(f"config.def @{kernel}.workgroup_count.{axis} = {count} : index\n" for axis, count in zip("xyz", (x, y, z), strict=True)) + "\n"


def scheduled_sum(arrays):
    return "\n".join(scheduled_sum_variant(arrays, f"scheduled_sum_unroll_{unroll}_depth_{depth}") for unroll in (1, 3) for depth in (1, 2))


def symbol_exports(arrays):
    samples = [0, 1, 2, 3, 255, 256, 65535, 65536, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
    rng = random.Random(83728)
    samples += [rng.randrange(1 << 32) for _ in range(20)]
    cases = []
    for index, value in enumerate(samples):
        case = Case(arrays, f"symbol_export_{index}", "i32", 1)
        original = signed_bits(value, 32)
        case.array("input", [original])
        case.launch("library.dispatch", "%output, %input", "tensor<1xi32>, tensor<1xi32>")
        case.array("original", [original])
        case.lines.append("  check.expect.bitwise actual(%input) expected(%original) : tensor<1xi32>")
        cases.append(case.finish([signed_bits(value * value + 7, 32)]))
    return "kernel.decl @library.dispatch() launch(%output: buffer, %input: buffer)\n\n" + "\n".join(cases)


KERNEL_GROUPS = {
    "aiter_swiglu_f16": lambda arrays: launch_grid("aiter_swiglu_f16", 3) + swiglu(arrays),
    "assumptions": assumption_kernel,
    "constant_loops": constant_loops,
    "control_flow": control_flow,
    "early_returns": early_returns,
    "enum_values": lambda arrays: "\n".join(enum_storage(arrays, width) for width in (8, 16, 32, 64)),
    "flash_attention": lambda arrays: launch_grid("flash_attention", 3) + attention(arrays),
    "increment_values": lambda arrays: increment_values(arrays) + "\n" + increment_pointers(arrays),
    "integer_increment": lambda arrays: integer_increment(arrays, 8, BYTE_INPUTS) + "\n" + integer_increment(arrays, 64, WIDE_INPUTS),
    "llama_rms_norm": lambda arrays: launch_grid("llama_rms_norm", 3) + rms_norm(arrays),
    "pointer_walk": pointer_walk,
    "scheduled_sum": scheduled_sum,
    "shaped_intrinsics": lambda arrays: register_lookup(arrays) + "\n" + register_lookup(arrays, floating=True) + "\n" + mixed_dot(arrays),
    "short_circuit": short_circuit,
    "structured_continue": lambda arrays: "\n".join(reference(arrays) for reference in (continue_values, continue_scheduled, continue_copy, continue_pointers, continue_vectors)),
    "symbol_exports": symbol_exports,
    "vector_depth": lambda arrays: vector_depth(arrays) + "\n" + vector_depth_span(arrays),
    "vector_initializers": vector_initializers,
    "vector_values": lambda arrays: vector_control(arrays) + "\n" + vector_masks(arrays),
}


HOST_REFERENCES = {
    "assumptions.cpp": assumption_functions,
    "comparison_functions.cpp": comparison_functions,
    "constant_loops.cpp": constant_loop_functions,
    "enum_values.cpp": enum_functions,
    "increment_values.cpp": increment_functions,
    "integer_functions.cpp": integer_functions,
    "schedule_values.cpp": schedule_functions,
    "shaped_intrinsics.cpp": shaped_intrinsic_values,
    "structured_continue.cpp": continue_functions,
    "vector_initializers.cpp": vector_constructor_values,
    "vector_values.cpp": vector_values,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    host = modes.add_parser("host", help="emit one source-authored scalar check group")
    host.add_argument("--source", type=Path, required=True)
    host.add_argument("--output", type=Path, required=True)
    kernel = modes.add_parser("kernel", help="emit one kernel check group and its arrays")
    kernel.add_argument("--group", choices=KERNEL_GROUPS, required=True)
    kernel.add_argument("--output", type=Path, required=True)
    kernel.add_argument("--arrays", type=Path, required=True)
    options = parser.parse_args()
    if options.mode == "kernel":
        arrays = Arrays(options.arrays, options.output.parent)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(KERNEL_GROUPS[options.group](arrays))
        return

    reference = HOST_REFERENCES.get(options.source.name)
    if reference is None:
        parser.error(f"no host reference for source '{options.source.name}'")

    # The source and output are declared build inputs/outputs. A relative include
    # preserves their relationship without embedding a sandbox or checkout path.
    include = Path(os.path.relpath(options.source, options.output.parent)).as_posix()
    contents = f'// Generated from independent Python numerical references.\n#include <loomcxx/check.h>\n#include "{include}"\n\n' + reference() + "\n"
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if not options.output.exists() or options.output.read_text() != contents:
        options.output.write_text(contents)


if __name__ == "__main__":
    main()
