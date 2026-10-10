# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C++ kernel inputs with independent numerical references."""

import argparse
import math
import random
import struct
from pathlib import Path

from loom.gen.test.kernel_fixture import Arrays, Case, rounded, signed_bits


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


def launch_grid(kernel, x, y=1, z=1):
    return "".join(f"config.def @{kernel}.workgroup_count.{axis} = {count} : index\n" for axis, count in zip("xyz", (x, y, z), strict=True)) + "\n"


def volatile_memory(arrays):
    cases = []
    inputs = [signed_bits(index * 0x10203041 + 0x7FFFFF00, 32) for index in range(64)]
    parameter_values = [0, 37, -128, (1 << 31) - 1]
    for ordinal, value in enumerate(parameter_values):
        case = Case(arrays, f"volatile_scalar_parameter_{ordinal}", "i32", 1)
        case.array("input", [value])
        case.array("original", [value])
        case.launch("volatile_scalar_parameter", "%input, %output", "tensor<1xi32>, tensor<1xi32>")
        case.lines.append("  check.expect.bitwise actual(%input) expected(%original) : tensor<1xi32>")
        cases.append(case.finish([value]))
    case = Case(arrays, "volatile_vector_parameter_values", "i32", 4)
    case.array("input", parameter_values)
    case.array("original", parameter_values)
    case.launch("volatile_vector_parameter_kernel", "%input, %output", "tensor<4xi32>, tensor<4xi32>")
    case.lines.append("  check.expect.bitwise actual(%input) expected(%original) : tensor<4xi32>")
    cases.append(case.finish(parameter_values))
    for kernel in ("volatile_memory", "ordinary_memory"):
        for count in (0, 1, 17, 64):
            expected = [-123] * 128
            for index in range(count):
                doubled = (inputs[index] * 2) % (1 << 32)
                expected[index] = signed_bits(doubled, 32)
                expected[64 + index] = signed_bits(doubled ^ 0xA5A55A5A, 32)
            case = Case(arrays, f"{kernel}_{count}", "i32", len(expected))
            case.array("input", inputs)
            case.array("original", inputs)
            case.scalar("count", count, "i32")
            case.launch(kernel, "%input, %output, %count", "tensor<64xi32>, tensor<128xi32>, i32")
            case.lines.append("  check.expect.bitwise actual(%input) expected(%original) : tensor<64xi32>")
            cases.append(case.finish(expected))
    for ordinal, values in enumerate(([0, 1, -1, -(1 << 31)], [37, -128, 65535, (1 << 31) - 1], inputs[:4])):
        case = Case(arrays, f"volatile_vectors_{ordinal}", "i32", 4)
        case.array("input", values)
        case.array("original", values)
        case.launch("volatile_vectors", "%input, %output", "tensor<4xi32>, tensor<4xi32>")
        case.lines.append("  check.expect.bitwise actual(%input) expected(%original) : tensor<4xi32>")
        cases.append(case.finish(values))
    case = Case(arrays, "volatile_shared_values", "i32", 64)
    case.array("input", inputs)
    case.array("original", inputs)
    case.launch("volatile_shared", "%input, %output", "tensor<64xi32>, tensor<64xi32>")
    case.lines.append("  check.expect.bitwise actual(%input) expected(%original) : tensor<64xi32>")
    cases.append(case.finish([signed_bits(value + 1, 32) for value in inputs]))
    for origin, stride in ((0, 4), (3, 7), (15, 15)):
        case = Case(arrays, f"volatile_views_{origin}_{stride}", "i32", 4)
        case.array("input", inputs)
        case.array("original", inputs)
        case.scalar("origin", origin, "i32")
        case.scalar("stride", stride, "i32")
        case.launch("volatile_views", "%input, %output, %origin, %stride", "tensor<64xi32>, tensor<4xi32>, i32, i32")
        case.lines.append("  check.expect.bitwise actual(%input) expected(%original) : tensor<64xi32>")
        cases.append(case.finish([-123, signed_bits(inputs[origin + stride + 1] * 2, 32), -123, -123]))
    declarations = "".join(f"kernel.decl @{kernel}() launch(%input: buffer, %output: buffer, %count: i32)\n\n" for kernel in ("volatile_memory", "ordinary_memory"))
    declarations += "".join(
        f"kernel.decl @{kernel}() launch(%input: buffer, %output: buffer)\n\n" for kernel in ("volatile_scalar_parameter", "volatile_vector_parameter_kernel", "volatile_vectors", "volatile_shared")
    )
    declarations += "kernel.decl @volatile_views() launch(%input: buffer, %output: buffer, %origin: i32, %stride: i32)\n\n"
    return declarations + "\n".join(cases)


def packed_byte_shifts(arrays):
    # Every byte position sees all 256 values, with different adjacent bytes.
    values = [(packet + lane * 73) % 256 for packet in range(256) for lane in range(8)]
    expected = []
    for packet in range(256):
        lanes = values[packet * 8 : (packet + 1) * 8]
        for amount in range(8):
            expected.extend((value * 2**amount) % 256 for value in lanes)
            expected.extend(value // 2**amount for value in lanes)
        expected.extend((value * 2 ** (value % 8)) % 256 for value in lanes)
        expected.extend(value // 2 ** (value % 8) for value in lanes)
        expected.extend(255 if value % 2 else 0 for value in lanes)
    case = Case(arrays, "packed_byte_shifts_values", "i8", len(expected))
    inputs = [signed_bits(value, 8) for value in values]
    case.array("input", inputs)
    case.array("original", inputs)
    case.launch("packed_byte_shifts", "%input, %output", f"tensor<{len(values)}xi8>, tensor<{len(expected)}xi8>")
    case.lines.append(f"  check.expect.bitwise actual(%input) expected(%original) : tensor<{len(values)}xi8>")
    return "kernel.decl @packed_byte_shifts() launch(%input: buffer, %output: buffer)\n\n" + case.finish([signed_bits(value, 8) for value in expected])


IQ4NL_CODEBOOK = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def rounded_bfloat16(value):
    """Round one finite value to bfloat16 and return its exact float value."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def pack_iq4xs(scale, group_scales, codes):
    """Pack logical scales/codes into the 136-byte IQ4_XS storage layout."""
    scale_codes = [value + 32 for value in group_scales]
    high = sum((value >> 4) << (2 * group) for group, value in enumerate(scale_codes))
    low = bytes((scale_codes[group] & 15) | ((scale_codes[group + 1] & 15) << 4) for group in range(0, 8, 2))
    quants = bytes(codes[group * 32 + lane] | (codes[group * 32 + lane + 16] << 4) for group in range(8) for lane in range(16))
    return struct.pack("<eH", scale, high) + low + quants


def iq4xs_blocks(arrays):
    # Expected values come from logical scales and code indices before packing.
    # Eight blocks cover every signed six-bit scale and all 256 packed byte
    # values, including distinct adjacent record contents. Low codes permute
    # all sixteen entries; group-dependent high codes span every pairing.
    packed = bytearray([0xA5] * 32)
    mutated = bytearray(packed)
    expected = []
    for block, scale in enumerate((0.5, -0.25, 2.0, -4.0, 0.0625, -0.125, 1.0, -2.0)):
        group_scales = [block * 8 + group - 32 for group in range(8)]
        codes = [(3 * lane + 5 * group + 7 * block + (lane // 16) * (group + 8 * block)) % 16 for group in range(8) for lane in range(32)]
        expected.extend(scale * group_scales[index // 32] * IQ4NL_CODEBOOK[code] for index, code in enumerate(codes))
        packed.extend(pack_iq4xs(scale, group_scales, codes))
        mutated.extend(pack_iq4xs(scale, [value ^ 1 for value in group_scales], [code ^ 1 for code in codes]))
    packed.extend([0xA5] * 32)
    mutated.extend([0xA5] * 32)

    cases = ""
    for kernel in ("decode_iq4xs", "decode_iq4xs_packed"):
        case = Case(arrays, kernel + "_values", "f32", len(expected))
        case.array("input_storage", [signed_bits(value, 8) for value in packed], "i8")
        case.array("original", [signed_bits(value, 8) for value in packed], "i8")
        case.array("codebook", IQ4NL_CODEBOOK, "i8")
        case.lines.append("  %input = check.tensor.view %input_storage offset(32) : tensor<1152xi8> -> tensor<1088xi8>")
        case.launch(kernel, "%input, %codebook, %output", "tensor<1088xi8>, tensor<16xi8>, tensor<2048xf32>")
        case.lines.append("  check.expect.bitwise actual(%input_storage) expected(%original) : tensor<1152xi8>")
        cases += case.finish(expected)
    input_path = arrays.write("iq4xs_update_input.npy", [signed_bits(value, 8) for value in packed], "i8")
    expected_path = arrays.write("iq4xs_update_expected.npy", [signed_bits(value, 8) for value in mutated], "i8")
    cases += f'''check.case public @iq4xs_update {{
  %storage = check.file.read.npy path("{input_path}") : tensor<1152xi8>
  %input = check.tensor.view %storage offset(32) : tensor<1152xi8> -> tensor<1088xi8>
  kernel.launch @update_iq4xs(%input) : (tensor<1088xi8>)
  %expected = check.file.read.npy path("{expected_path}") : tensor<1152xi8>
  check.expect.bitwise actual(%storage) expected(%expected) : tensor<1152xi8>
  check.return
}}
'''
    declarations = "kernel.decl @decode_iq4xs() launch(%blocks: buffer, %codebook: buffer, %output: buffer)\n\n"
    declarations += "kernel.decl @decode_iq4xs_packed() launch(%blocks: buffer, %codebook: buffer, %output: buffer)\n\n"
    declarations += "kernel.decl @update_iq4xs() launch(%blocks: buffer)\n\n"
    return declarations + cases


def iq4xs_gate_up(arrays):
    # Preserve the production 2,560 x 640 x top-10 execution geometry while
    # storing only the two experts selected by this fixture. Expected values
    # are evaluated from the logical scale/code records before packing.
    input_size = 2560
    output_size = 640
    route_ids = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    expert_count = 2
    block_count = input_size // 256
    input_values = [rounded_bfloat16(-1.0 + (index % 257) / 128.0) for index in range(input_size)]

    projections = {}
    packed_weights = {}
    for name, kind in (("gate", 0), ("up", 1)):
        records = bytearray()
        experts = []
        for expert in range(expert_count):
            rows = []
            for channel in range(output_size):
                products = []
                for block in range(block_count):
                    scale = 2.0 ** -(11 + ((expert + channel + block + kind) % 3))
                    group_scales = [((expert * 19 + channel * 7 + block * 13 + group * 9 + kind * 23) % 64) - 32 for group in range(8)]
                    codes = [(expert * 11 + channel * 5 + block * 7 + group * 3 + lane * 13 + lane // 5 + kind * 9) % 16 for group in range(8) for lane in range(32)]
                    records.extend(pack_iq4xs(scale, group_scales, codes))
                    for group, group_scale in enumerate(group_scales):
                        decoded_scale = rounded_bfloat16(scale * group_scale)
                        for lane in range(32):
                            code = codes[group * 32 + lane]
                            weight = rounded_bfloat16(decoded_scale * IQ4NL_CODEBOOK[code])
                            input_index = block * 256 + group * 32 + lane
                            products.append(weight * input_values[input_index])
                rows.append(math.fsum(products))
            experts.append(rows)
        projections[name] = experts
        packed_weights[name] = records

    expected = []
    for expert in route_ids:
        for channel in range(output_size):
            gate = projections["gate"][expert][channel]
            up = projections["up"][expert][channel]
            expected.append(gate / (1.0 + math.exp(-gate)) * up)

    case = Case(arrays, "iq4xs_gate_up_values", "f32", len(expected))
    case.lines.append(f"  %input = check.generate.iota offset(-1.0) step(0.0078125) period(257) : tensor<{input_size}xbf16>")
    case.array("route_ids", route_ids, "i32")
    for name in ("gate", "up"):
        case.array(f"{name}_weight", [signed_bits(value, 8) for value in packed_weights[name]], "i8")
    case.launch(
        "qwen38_iq4xs_gate_up_swiglu",
        "%input, %route_ids, %gate_weight, %up_weight, %output",
        f"tensor<{input_size}xbf16>, tensor<{len(route_ids)}xi32>, tensor<{len(packed_weights['gate'])}xi8>, tensor<{len(packed_weights['up'])}xi8>, tensor<{len(expected)}xf32>",
    )
    declaration = "kernel.decl @qwen38_iq4xs_gate_up_swiglu() launch(%input: buffer, %route_ids: buffer, %gate_weight: buffer, %up_weight: buffer, %output: buffer)\n\n"
    benchmark = "check.benchmark<@iq4xs_gate_up_values> @iq4xs_gate_up_latency\n"
    return declaration + case.finish(expected, 0.00001) + benchmark


def pack_q4k(scale, minimum, group_scales, group_minimums, codes):
    """Pack logical Q4_K fields into its 144-byte storage layout."""
    scale0 = bytes((group_scales[group] & 15) | ((group_scales[group] >> 4) << 4) | ((group_scales[group + 4] >> 4) << 6) for group in range(4))
    scale1 = bytes((group_minimums[group] & 15) | ((group_minimums[group] >> 4) << 4) | ((group_minimums[group + 4] >> 4) << 6) for group in range(4))
    scale2 = bytes((group_scales[group + 4] & 15) | ((group_minimums[group + 4] & 15) << 4) for group in range(4))
    quants = bytes(codes[(pair * 2) * 32 + half * 16 + lane] | (codes[(pair * 2 + 1) * 32 + half * 16 + lane] << 4) for pair in range(4) for half in range(2) for lane in range(16))
    return struct.pack("<ee", scale, minimum) + scale0 + scale1 + scale2 + quants


def pack_q8_1(scales, values):
    """Pack four logical 32-element Q8_1 groups into one 144-byte record."""
    sums = [scales[group] * sum(values[group * 32 : (group + 1) * 32]) for group in range(4)]
    header = b"".join(struct.pack("<ee", scales[group], sums[group]) for group in range(4))
    quants = bytes(value & 0xFF for group in range(4) for half in range(2) for value in values[group * 32 + half * 16 : group * 32 + (half + 1) * 16])
    return header + quants


def q4k_q8_swiglu(arrays):
    # This is the smallest supported input specialization: two Q4_K blocks and
    # four Q8_1 records. Logical fields are retained separately from their
    # packed bytes so the reference does not repeat the kernel's decode or
    # addressing implementation.
    token_count = 2
    route_count = 8
    route_stride = 11
    expert_count = 128
    tested_expert_count = 2
    output_size = 768
    q4_block_count = 2

    q8_input = bytearray()
    logical_q8 = []
    for token in range(token_count):
        token_groups = []
        for record in range(q4_block_count * 2):
            scales = [2.0 ** -(4 + ((token + record + group) % 3)) for group in range(4)]
            values = [((token * 11 + record * 7 + group * 5 + lane * 3 + lane // 5) % 15) - 7 for group in range(4) for lane in range(32)]
            q8_input.extend(pack_q8_1(scales, values))
            token_groups.append((scales, values))
        logical_q8.append(token_groups)

    logical_weights = {}
    packed_weights = {}
    for name, kind in (("gate", 0), ("up", 1)):
        records = bytearray()
        experts = []
        for expert in range(tested_expert_count):
            rows = []
            for channel in range(output_size):
                blocks = []
                for block in range(q4_block_count):
                    scale = 2.0 ** -(9 + ((expert + channel + block + kind) % 3))
                    minimum = 2.0 ** -(11 + ((expert + channel + 2 * block + kind) % 2))
                    group_scales = [(expert * 17 + channel * 3 + block * 11 + group * 7 + kind * 13) % 64 for group in range(8)]
                    group_minimums = [(expert * 13 + channel * 5 + block * 3 + group * 9 + kind * 7) % 64 for group in range(8)]
                    codes = [(expert * 13 + channel * 7 + block * 5 + group * 3 + lane * 11 + lane // 7 + kind * 9) % 16 for group in range(8) for lane in range(32)]
                    records.extend(pack_q4k(scale, minimum, group_scales, group_minimums, codes))
                    blocks.append((scale, minimum, group_scales, group_minimums, codes))
                rows.append(blocks)
            experts.append(rows)
        logical_weights[name] = experts
        packed_weights[name] = records

    route_ids = []
    for token in range(token_count):
        route_ids.extend((token + route) % tested_expert_count for route in range(route_count))
        route_ids.extend([expert_count - 1] * (route_stride - route_count))
    expected = []
    for token in range(token_count):
        for route in range(route_count):
            expert = route_ids[token * route_stride + route]
            for channel in range(output_size):
                accumulators = {}
                for name in ("gate", "up"):
                    accumulator = 0.0
                    for block in range(q4_block_count):
                        scale, minimum, group_scales, group_minimums, codes = logical_weights[name][expert][channel][block]
                        for group in range(8):
                            q8_scales, q8_values = logical_q8[token][block * 2 + group // 4]
                            q8_scale = q8_scales[group % 4]
                            q8_group = q8_values[(group % 4) * 32 : (group % 4 + 1) * 32]
                            q4_group = codes[group * 32 : (group + 1) * 32]
                            dot = sum(q4 * q8 for q4, q8 in zip(q4_group, q8_group, strict=True))
                            accumulator += q8_scale * scale * group_scales[group] * dot
                            accumulator -= minimum * group_minimums[group] * q8_scale * sum(q8_group)
                    accumulators[name] = accumulator
                gate = accumulators["gate"]
                expected.append(gate / (1.0 + math.exp(-gate)) * accumulators["up"])

    case = Case(arrays, "q4k_q8_swiglu_values", "f32", len(expected))
    case.array("q8_input", [signed_bits(value, 8) for value in q8_input], "i8")
    case.array("route_ids", route_ids, "i32")
    for name in ("gate", "up"):
        case.array(f"{name}_weight", [signed_bits(value, 8) for value in packed_weights[name]], "i8")
    case.scalar("token_count", token_count, "i32")
    case.scalar("route_count", route_count, "i32")
    case.scalar("route_stride", route_stride, "i32")
    case.scalar("expert_count", expert_count, "i32")
    case.scalar("output_size", output_size, "i32")
    case.launch(
        "ffn_routed_gate_up_swiglu_q4k_q8",
        "%token_count, %route_count, %route_stride, %expert_count, %output_size, %q8_input, %route_ids, %gate_weight, %up_weight, %output",
        f"i32, i32, i32, i32, i32, tensor<{len(q8_input)}xi8>, tensor<{token_count * route_stride}xi32>, tensor<{len(packed_weights['gate'])}xi8>, tensor<{len(packed_weights['up'])}xi8>, tensor<{len(expected)}xf32>",
        "%token_count, %route_count, %route_stride, %expert_count, %output_size",
        "i32, i32, i32, i32, i32",
    )
    declaration = "kernel.decl @ffn_routed_gate_up_swiglu_q4k_q8(%workload_token_count: i32, %workload_route_count: i32, %workload_route_stride: i32, %workload_expert_count: i32, %workload_output_size: i32) launch(%token_count: i32, %route_count: i32, %route_stride: i32, %expert_count: i32, %output_size: i32, %q8_input: buffer, %route_ids: buffer, %gate_weight: buffer, %up_weight: buffer, %output: buffer)\n\n"
    return declaration + case.finish(expected, 0.002)


KERNEL_GROUPS = {
    "aiter_swiglu_f16": lambda arrays: launch_grid("aiter_swiglu_f16", 3) + swiglu(arrays),
    "control_flow": control_flow,
    "early_returns": early_returns,
    "flash_attention": lambda arrays: launch_grid("flash_attention", 3) + attention(arrays),
    "iq4xs_blocks": iq4xs_blocks,
    "iq4xs_gate_up": iq4xs_gate_up,
    "llama_rms_norm": lambda arrays: launch_grid("llama_rms_norm", 3) + rms_norm(arrays),
    "packed_byte_shifts": packed_byte_shifts,
    "pointer_walk": pointer_walk,
    "q4k_q8_swiglu": q4k_q8_swiglu,
    "short_circuit": short_circuit,
    "vector_depth": lambda arrays: vector_depth(arrays) + "\n" + vector_depth_span(arrays),
    "volatile_memory": volatile_memory,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    kernel = modes.add_parser("kernel", help="emit one kernel check group and its arrays")
    kernel.add_argument("--group", choices=KERNEL_GROUPS, required=True)
    kernel.add_argument("--output", type=Path, required=True)
    kernel.add_argument("--arrays", type=Path, required=True)
    options = parser.parse_args()
    arrays = Arrays(options.arrays, options.output.parent)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(KERNEL_GROUPS[options.group](arrays))


if __name__ == "__main__":
    main()
