# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C++ kernel inputs with independent floating-point and exact integer references."""

import math
import random
import struct
import sys
from pathlib import Path

ELEMENTS = {"f16": ("e", "<f2", 2), "f32": ("f", "<f4", 4), "i8": ("b", "|i1", 1), "i32": ("i", "<i4", 4), "i64": ("q", "<i8", 8)}

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


class Case:
    def __init__(self, directory, name, element, count):
        self.directory = directory
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
        write_npy(self.directory / filename, values, self.element)
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
        self.lines.extend(["  check.return", "}", ""])
        return "\n".join(self.lines)


def attention(directory):
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
        case = Case(directory, f"attention_{queries}_{keys}_{magnitude}", "f32", queries * 64)
        for name, values in [("query", query), ("key", key), ("value", value)]:
            case.array(name, values)
        case.scalar("queries", queries, "i32")
        case.scalar("keys", keys, "i32")
        case.launch(
            "flash_attention", "%query, %key, %value, %output, %queries, %keys", f"tensor<{len(query)}xf32>, tensor<{len(key)}xf32>, tensor<{len(value)}xf32>, tensor<{len(expected)}xf32>, i32, i32"
        )
        cases.append(case.finish(expected, 0.0003))
    return "kernel.decl @flash_attention() launch(%query: buffer, %key: buffer, %value: buffer, %output: buffer, %query_count: i32, %key_count: i32)\n\n" + "\n".join(cases)


def rms_norm(directory):
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
        case = Case(directory, f"rms_norm_{columns}", "f32", len(values))
        case.array("input", values)
        case.scalar("columns", columns, "i32")
        case.scalar("epsilon", epsilon, "f32")
        case.launch("llama_rms_norm", "%input, %output, %columns, %epsilon", f"tensor<{len(values)}xf32>, tensor<{len(values)}xf32>, i32, f32")
        cases.append(case.finish(expected, 0.0002))
    return "kernel.decl @llama_rms_norm() launch(%input: buffer, %output: buffer, %columns: i32, %epsilon: f32)\n\n" + "\n".join(cases)


def swiglu(directory):
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
        case = Case(directory, f"swiglu_f16_{columns}", "f16", len(expected))
        case.array("input", values)
        case.scalar("columns", columns, "i32")
        case.launch("aiter_swiglu_f16", "%input, %output, %columns", f"tensor<{len(values)}xf16>, tensor<{len(expected)}xf16>, i32")
        cases.append(case.finish(expected, 0.002))
    return "kernel.decl @aiter_swiglu_f16() launch(%input: buffer, %output: buffer, %columns: i32)\n\n" + "\n".join(cases)


def control_flow(directory):
    counts = [0, 1, 2, 7, 31, 64, 129]
    expected = []
    for count in counts:
        trips = max(1, count)
        inner = max(1, count & 3)
        nested = inner * count * (count - 1) // 2 + count * inner * (inner + 1) // 2
        expected.extend([count * (count + 1) // 2, count, trips * (trips + 1) // 2, trips, nested, count, trips, max(0, count - 1), trips, trips])
    case = Case(directory, "loop_semantics", "i32", len(expected))
    case.array("counts", counts)
    case.scalar("length", len(counts), "i32")
    case.launch("control_flow", "%counts, %output, %length", f"tensor<{len(counts)}xi32>, tensor<{len(expected)}xi32>, i32")
    return "kernel.decl @control_flow() launch(%counts: buffer, %output: buffer, %length: i32)\n\n" + case.finish(expected)


def scheduled_sum(directory):
    cases = []
    rows = 7
    for columns in [0, 1, 2, 5, 17, 33]:
        rng = random.Random(1030 + columns)
        values = [rng.randrange(-100, 101) for _ in range(rows * max(1, columns))]
        expected = [sum(values[row * columns : (row + 1) * columns]) for row in range(rows)]
        case = Case(directory, f"scheduled_sum_{columns}", "i32", rows)
        case.array("input", values)
        case.scalar("rows", rows, "i32")
        case.scalar("columns", columns, "i32")
        case.launch("scheduled_sum", "%input, %output, %rows, %columns", f"tensor<{len(values)}xi32>, tensor<{rows}xi32>, i32, i32")
        cases.append(case.finish(expected))
    return "kernel.decl @scheduled_sum() launch(%input: buffer, %output: buffer, %rows: i32, %columns: i32)\n\n" + "\n".join(cases)


def short_circuit(directory):
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
        case = Case(directory, f"short_circuit_{length}", "i32", len(expected))
        case.array("input", values)
        case.scalar("length", length, "i32")
        case.launch("short_circuit", "%input, %output, %length", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        cases.append(case.finish(expected))
    return "kernel.decl @short_circuit() launch(%input: buffer, %output: buffer, %length: i32)\n\n" + "\n".join(cases)


def early_returns(directory):
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
        case = Case(directory, f"early_returns_{length}", "i32", len(expected))
        case.array("input", values)
        case.scalar("length", length, "i32")
        case.launch("early_returns", "%input, %output, %length", f"tensor<{len(values)}xi32>, tensor<{len(expected)}xi32>, i32")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        cases.append(case.finish(expected))
    return "kernel.decl @early_returns() launch(%input: buffer, %output: buffer, %length: i32)\n\n" + "\n".join(cases)


def integer_increment(directory, width, inputs):
    cases = []
    argument_width = 32 if width == 8 else 64
    for input_value in inputs:
        expected = []
        for lane in range(64):
            expected.extend([signed_bits(input_value + 1, width), signed_bits(input_value + lane + 1, width)])
        case = Case(directory, f"increment_u{width}_{input_value}", f"i{width}", len(expected))
        case.scalar("input", signed_bits(input_value, argument_width), f"i{argument_width}")
        case.launch(f"increment_u{width}", "%output, %input", f"tensor<{len(expected)}xi{width}>, i{argument_width}")
        cases.append(case.finish(expected))
    return f"kernel.decl @increment_u{width}() launch(%output: buffer, %input: i{argument_width})\n\n" + "\n".join(cases)


def function_cases(name, argument_widths, result_width, samples):
    types = ", ".join(f"i{width}" for width in argument_widths)
    parameters = ", ".join(f"%arg{index}: i{width}" for index, width in enumerate(argument_widths))
    cases = [f"func.decl @{name}({parameters}) -> (i{result_width})"]
    for ordinal, (arguments, expected) in enumerate(samples):
        lines = [f"check.case public @{name}_{ordinal} {{"]
        for index, (value, width) in enumerate(zip(arguments, argument_widths, strict=True)):
            lines.append(f"  %arg{index} = check.literal value({signed_bits(value, width)}) : i{width}")
        operands = ", ".join(f"%arg{index}" for index in range(len(arguments)))
        lines.append(f"  %actual = func.call @{name}({operands}) : ({types}) -> (i{result_width})")
        lines.append(f"  %expected = check.literal value({signed_bits(expected, result_width)}) : i{result_width}")
        lines.append(f"  check.expect.equal actual(%actual) expected(%expected) : i{result_width}")
        lines.extend(["  check.return", "}"])
        cases.append("\n".join(lines))
    return "\n\n".join(cases)


def integer_functions(directory):
    del directory
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


def comparison_functions(directory):
    del directory
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


def pointer_walk(directory):
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
        case = Case(directory, f"pointer_walk_{start}_{abs(displacement)}", "i32", len(expected))
        case.array("input", values)
        case.array("counts", counts)
        case.scalar("length", len(counts), "i32")
        case.scalar("start", start, "i64")
        case.scalar("displacement", displacement, "i64")
        case.launch("pointer_walk", "%input, %counts, %output, %length, %start, %displacement", f"tensor<{len(values)}xi32>, tensor<{len(counts)}xi32>, tensor<{len(expected)}xi32>, i32, i64, i64")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        for name, original in [("input", values), ("counts", counts)]:
            case.array(name + "_expected", original)
            case.lines.append(f"  check.expect.bitwise actual(%{name}) expected(%{name}_expected) : tensor<{len(original)}xi32>")
        cases.append(case.finish(expected))
    return "kernel.decl @pointer_walk() launch(%input: buffer, %counts: buffer, %output: buffer, %length: i32, %start: i64, %displacement: i64)\n\n" + "\n".join(cases)


def vector_depth(directory):
    cases = []
    for blocks in [1, 3, 7]:
        rng = random.Random(843 + blocks)
        previous = [rng.getrandbits(32) for _ in range(blocks * 16)]
        depth = [rng.getrandbits(32) for _ in previous]
        previous[:4] = [0, 0xFFFFFFFF, 0x12345678, 0xFFFF0000]
        depth[:4] = [0xFFFFFFFF, 0, 0x87654321, 0xFFFF]
        expected = [signed_bits((a & 0xFFFF) | (b & 0xFFFF0000), 32) for a, b in zip(previous, depth, strict=True)]
        case = Case(directory, f"vector_depth_{blocks}", "i32", len(previous))
        case.array("previous", [signed_bits(value, 32) for value in previous])
        case.array("depth", [signed_bits(value, 32) for value in depth])
        case.scalar("count", blocks, "i32")
        case.launch("vector_depth", "%previous, %depth, %output, %count", f"tensor<{len(previous)}xi32>, tensor<{len(depth)}xi32>, tensor<{len(expected)}xi32>, i32")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        cases.append(case.finish(expected))
    return "kernel.decl @vector_depth() launch(%previous: buffer, %depth: buffer, %output: buffer, %count: i32)\n\n" + "\n".join(cases)


def vector_depth_span(directory):
    cases = []
    for count in [0, 1, 15, 16, 17, 31, 32, 129]:
        rng = random.Random(281 + count)
        previous = [rng.getrandbits(32) for _ in range(max(1, count))]
        depth, step = 0xFFFF1234, 0x137CF
        expected = [signed_bits((value & 0xFFFF) | ((depth + index * step) & 0xFFFF0000), 32) for index, value in enumerate(previous[:count])]
        case = Case(directory, f"vector_depth_span_{count}", "i32", count)
        case.array("previous", [signed_bits(value, 32) for value in previous])
        for name, value in [("count", count), ("depth", depth), ("step", step)]:
            case.scalar(name, signed_bits(value, 32), "i32")
        case.launch("vector_depth_span", "%previous, %output, %count, %depth, %step", f"tensor<{len(previous)}xi32>, tensor<{count}xi32>, i32, i32, i32")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        cases.append(case.finish(expected))
    return "kernel.decl @vector_depth_span() launch(%previous: buffer, %output: buffer, %count: i32, %depth: i32, %step: i32)\n\n" + "\n".join(cases)


def vector_values(directory):
    del directory
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


def vector_control(directory):
    cases = []
    for value in [0, 0xFFFFFFFF, 0x7FFFFFFF]:
        for count in [0, 1, 3, 7]:
            expected = [(initial + 4 * count) & 0xFFFFFFFF for initial in [value, 1, 0, 0]]
            if count > 2:
                expected = [~element for element in expected]
            case = Case(directory, f"vector_control_{value}_{count}", "i32", 4)
            case.scalar("input", signed_bits(value, 32), "i32")
            case.scalar("count", count, "i32")
            case.launch("vector_control_kernel", "%output, %input, %count", "tensor<4xi32>, i32, i32")
            case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
            cases.append(case.finish([signed_bits(element, 32) for element in expected]))
    return "kernel.decl @vector_control_kernel() launch(%output: buffer, %input: i32, %count: i32)\n\n" + "\n".join(cases)


def f32_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def vector_masks(directory):
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
        case = Case(directory, f"vector_masks_{ordinal}", "i32", len(expected))
        case.array("input", [signed_bits(value, 32) for value in left + right])
        case.launch("vector_masks", "%input, %output", "tensor<8xi32>, tensor<20xi32>")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        cases.append(case.finish(expected))
    return "kernel.decl @vector_masks() launch(%input: buffer, %output: buffer)\n\n" + "\n".join(cases)


def shaped_intrinsic_values(directory):
    del directory
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


def register_lookup(directory, *, floating=False):
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
        case = Case(directory, f"{name}_{count}", "i32", len(expected))
        case.array("table", table)
        case.array("indices", indices)
        case.scalar("count", count, "i32")
        case.launch(name, "%table, %indices, %output, %count", f"tensor<16xi32>, tensor<{len(indices)}xi32>, tensor<{len(expected)}xi32>, i32")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        cases.append(case.finish(expected))
    return f"kernel.decl @{name}() launch(%table: buffer, %indices: buffer, %output: buffer, %count: i32)\n\n" + "\n".join(cases)


def mixed_dot(directory):
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
        case = Case(directory, f"mixed_dot_{count}", "i32", len(expected))
        for name, values in [("lhs", lhs), ("rhs", rhs)]:
            packed = struct.unpack(f"<{len(values) // 4}i", bytes(values))
            case.array(name, packed)
        case.array("acc", accumulators)
        case.scalar("count", count, "i32")
        case.launch("mixed_dot", "%lhs, %rhs, %acc, %output, %count", f"tensor<{len(lhs) // 4}xi32>, tensor<{len(rhs) // 4}xi32>, tensor<{len(accumulators)}xi32>, tensor<{len(expected)}xi32>, i32")
        case.lines.append('  check.expect.event<device> {type = "asan_report", count = 0}')
        cases.append(case.finish(expected))
    return "kernel.decl @mixed_dot() launch(%lhs: buffer, %rhs: buffer, %acc: buffer, %output: buffer, %count: i32)\n\n" + "\n".join(cases)


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    for name, generator in [
        ("flash_attention", attention),
        ("llama_rms_norm", rms_norm),
        ("aiter_swiglu_f16", swiglu),
        ("control_flow", control_flow),
        ("scheduled_sum", scheduled_sum),
        ("short_circuit", short_circuit),
        ("early_returns", early_returns),
        ("increment_u8", lambda directory: integer_increment(directory, 8, BYTE_INPUTS)),
        ("increment_u64", lambda directory: integer_increment(directory, 64, WIDE_INPUTS)),
        ("integer_functions", integer_functions),
        ("comparison_functions", comparison_functions),
        ("pointer_walk", pointer_walk),
        ("vector_depth", vector_depth),
        ("vector_depth_span", vector_depth_span),
        ("vector_values", vector_values),
        ("vector_control", vector_control),
        ("vector_masks", vector_masks),
        ("shaped_intrinsic_values", shaped_intrinsic_values),
        ("register_lookup", register_lookup),
        ("register_lookup_float", lambda directory: register_lookup(directory, floating=True)),
        ("mixed_dot", mixed_dot),
    ]:
        (directory / f"{name}.loom").write_text(generator(directory))


if __name__ == "__main__":
    main()
