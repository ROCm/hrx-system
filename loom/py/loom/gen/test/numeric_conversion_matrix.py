# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates exact target-independent FP8 conversion witnesses."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from loom.dialect.encoding.numeric_formats import (
    FP8_FORMATS,
    Fp8Format,
    Fp8SpecialPolicy,
)
from loom.gen import bootstrap as _bootstrap
from loom.gen.support.generated_file import (
    GeneratedFileMaintenanceMode,
    GeneratedFileMaintenanceResult,
    GeneratedFileSet,
    line_comment_header,
    maintain_generated_file_set,
)
from loom.ir import ScalarTypeKind

_GENERATOR = "loom.gen.test.numeric_conversion_matrix"
DESCRIPTION = "FP8 numeric conversion witnesses"
REGENERATE_COMMAND = "python3 loom/py/loom/gen/run.py numeric_conversion_matrix --in-place"
_OUTPUT_ROOT = Path("loom/src/loom/test/corpus/encoding")


@dataclass(frozen=True)
class _BinaryFormat:
    scalar_type: ScalarTypeKind
    type_spelling: str
    integer_type_spelling: str
    bit_count: int
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int


@dataclass(frozen=True)
class _BinaryValue:
    kind: Literal["finite", "infinity", "nan"]
    sign: int = 0
    magnitude: Fraction = Fraction(0)


@dataclass(frozen=True)
class _ExactValue:
    name: str
    raw_bits: int
    expected_bits: int


_BINARY_FORMATS = (
    _BinaryFormat(
        ScalarTypeKind.F32,
        "f32",
        "i32",
        bit_count=32,
        exponent_bits=8,
        mantissa_bits=23,
        exponent_bias=127,
    ),
    _BinaryFormat(
        ScalarTypeKind.F16,
        "f16",
        "i16",
        bit_count=16,
        exponent_bits=5,
        mantissa_bits=10,
        exponent_bias=15,
    ),
    _BinaryFormat(
        ScalarTypeKind.BF16,
        "bf16",
        "i16",
        bit_count=16,
        exponent_bits=8,
        mantissa_bits=7,
        exponent_bias=127,
    ),
)


def _pow2(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << -exponent)


def _round_nearest_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled_remainder = remainder * 2
    if doubled_remainder > value.denominator:
        return quotient + 1
    if doubled_remainder < value.denominator:
        return quotient
    return quotient + (quotient & 1)


def _floor_log2(value: Fraction) -> int:
    assert value > 0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _pow2(exponent):
        exponent -= 1
    return exponent


def _binary_decode_bits(raw_bits: int, binary_format: _BinaryFormat) -> _BinaryValue:
    sign = (raw_bits >> (binary_format.bit_count - 1)) & 1
    exponent_mask = (1 << binary_format.exponent_bits) - 1
    mantissa_mask = (1 << binary_format.mantissa_bits) - 1
    exponent = (raw_bits >> binary_format.mantissa_bits) & exponent_mask
    mantissa = raw_bits & mantissa_mask
    if exponent == exponent_mask:
        return _BinaryValue("infinity" if mantissa == 0 else "nan", sign)
    if exponent == 0:
        magnitude = Fraction(mantissa) * _pow2(1 - binary_format.exponent_bias - binary_format.mantissa_bits)
    else:
        magnitude = Fraction((1 << binary_format.mantissa_bits) | mantissa) * _pow2(exponent - binary_format.exponent_bias - binary_format.mantissa_bits)
    return _BinaryValue("finite", sign, magnitude)


def _binary_encode_bits(value: _BinaryValue, binary_format: _BinaryFormat) -> int:
    sign_bits = value.sign << (binary_format.bit_count - 1)
    exponent_mask = (1 << binary_format.exponent_bits) - 1
    if value.kind == "nan":
        quiet_nan = 1 << (binary_format.mantissa_bits - 1)
        return sign_bits | (exponent_mask << binary_format.mantissa_bits) | quiet_nan
    if value.kind == "infinity":
        return sign_bits | (exponent_mask << binary_format.mantissa_bits)
    if value.magnitude == 0:
        return sign_bits

    exponent = _floor_log2(value.magnitude)
    minimum_normal_exponent = 1 - binary_format.exponent_bias
    if exponent < minimum_normal_exponent:
        quantum = _pow2(minimum_normal_exponent - binary_format.mantissa_bits)
        mantissa = _round_nearest_even(value.magnitude / quantum)
        if mantissa == 0:
            return sign_bits
        if mantissa < (1 << binary_format.mantissa_bits):
            return sign_bits | mantissa
        exponent = minimum_normal_exponent
        mantissa = 0
    else:
        scaled_significand = value.magnitude / _pow2(exponent)
        rounded_significand = _round_nearest_even(scaled_significand * (1 << binary_format.mantissa_bits))
        if rounded_significand == (1 << (binary_format.mantissa_bits + 1)):
            exponent += 1
            rounded_significand >>= 1
        mantissa = rounded_significand - (1 << binary_format.mantissa_bits)

    encoded_exponent = exponent + binary_format.exponent_bias
    if encoded_exponent >= exponent_mask:
        return sign_bits | (exponent_mask << binary_format.mantissa_bits)
    return sign_bits | (encoded_exponent << binary_format.mantissa_bits) | mantissa


def _fp8_decode_bits(row: Fp8Format, raw_bits: int) -> _BinaryValue:
    raw_bits &= 0xFF
    sign = raw_bits >> 7
    magnitude_bits = raw_bits & 0x7F
    exponent_mask = (1 << row.exponent_bits) - 1
    mantissa_mask = (1 << row.mantissa_bits) - 1
    exponent = magnitude_bits >> row.mantissa_bits
    mantissa = magnitude_bits & mantissa_mask
    special_policy = row.special_policy
    if special_policy == Fp8SpecialPolicy.FINITE_NAN_UNSIGNED_ZERO and raw_bits == 0x80:
        return _BinaryValue("nan")
    if special_policy == Fp8SpecialPolicy.FINITE_NAN and magnitude_bits == 0x7F:
        return _BinaryValue("nan")
    if special_policy == Fp8SpecialPolicy.IEEE and exponent == exponent_mask:
        if mantissa == 0:
            return _BinaryValue("infinity", sign)
        return _BinaryValue("nan")
    if exponent == 0:
        magnitude = Fraction(mantissa) * _pow2(1 - row.exponent_bias - row.mantissa_bits)
    else:
        magnitude = Fraction((1 << row.mantissa_bits) | mantissa) * _pow2(exponent - row.exponent_bias - row.mantissa_bits)
    return _BinaryValue("finite", sign, magnitude)


def _fp8_positive_finite_payloads(row: Fp8Format) -> tuple[int, ...]:
    return tuple(payload for payload in range(0x80) if _fp8_decode_bits(row, payload).kind == "finite")


def _fp8_maximum_finite_payload(row: Fp8Format) -> int:
    return _fp8_positive_finite_payloads(row)[-1]


def _fp8_nan_payload(row: Fp8Format, sign: int) -> int:
    if row.special_policy == Fp8SpecialPolicy.FINITE_NAN_UNSIGNED_ZERO:
        return 0x80
    return (sign << 7) | 0x7F


def _fp8_encode_bits(row: Fp8Format, value: _BinaryValue) -> int:
    special_policy = row.special_policy
    if value.kind == "nan":
        return _fp8_nan_payload(row, value.sign)

    maximum_payload = _fp8_maximum_finite_payload(row)
    sign_bits = value.sign << 7
    if value.kind == "infinity":
        if special_policy == Fp8SpecialPolicy.IEEE:
            exponent_mask = (1 << row.exponent_bits) - 1
            return sign_bits | (exponent_mask << row.mantissa_bits)
        return sign_bits | maximum_payload
    if value.magnitude == 0:
        return 0 if special_policy == Fp8SpecialPolicy.FINITE_NAN_UNSIGNED_ZERO else sign_bits

    finite_payloads = _fp8_positive_finite_payloads(row)
    maximum_value = _fp8_decode_bits(row, maximum_payload).magnitude
    if special_policy == Fp8SpecialPolicy.IEEE:
        previous_value = _fp8_decode_bits(row, finite_payloads[-2]).magnitude
        overflow_threshold = maximum_value + (maximum_value - previous_value) / 2
        if value.magnitude >= overflow_threshold:
            exponent_mask = (1 << row.exponent_bits) - 1
            return sign_bits | (exponent_mask << row.mantissa_bits)
    elif value.magnitude >= maximum_value:
        return sign_bits | maximum_payload

    best_payload = min(
        finite_payloads,
        key=lambda payload: (
            abs(_fp8_decode_bits(row, payload).magnitude - value.magnitude),
            payload & 1,
            payload,
        ),
    )
    return sign_bits | best_payload


def _fp8_boundary_values(row: Fp8Format) -> tuple[tuple[str, _BinaryValue], ...]:
    mantissa_mask = (1 << row.mantissa_bits) - 1
    minimum_subnormal = _fp8_decode_bits(row, 1).magnitude
    maximum_subnormal = _fp8_decode_bits(row, mantissa_mask).magnitude
    minimum_normal = _fp8_decode_bits(row, 1 << row.mantissa_bits).magnitude
    one_payload = _fp8_encode_bits(row, _BinaryValue("finite", magnitude=Fraction(1)))
    next_one = _fp8_decode_bits(row, one_payload + 1).magnitude
    maximum_finite = _fp8_decode_bits(row, _fp8_maximum_finite_payload(row)).magnitude
    return (
        ("positive_zero", _BinaryValue("finite")),
        ("negative_zero", _BinaryValue("finite", sign=1)),
        ("minimum_subnormal", _BinaryValue("finite", magnitude=minimum_subnormal)),
        ("negative_minimum_subnormal", _BinaryValue("finite", sign=1, magnitude=minimum_subnormal)),
        ("zero_tie", _BinaryValue("finite", magnitude=minimum_subnormal / 2)),
        ("even_subnormal_tie", _BinaryValue("finite", magnitude=minimum_subnormal * Fraction(3, 2))),
        ("maximum_subnormal", _BinaryValue("finite", magnitude=maximum_subnormal)),
        ("minimum_normal", _BinaryValue("finite", magnitude=minimum_normal)),
        ("one", _BinaryValue("finite", magnitude=Fraction(1))),
        ("negative_one", _BinaryValue("finite", sign=1, magnitude=Fraction(1))),
        ("one_next_tie", _BinaryValue("finite", magnitude=(Fraction(1) + next_one) / 2)),
        ("maximum_finite", _BinaryValue("finite", magnitude=maximum_finite)),
        ("positive_infinity", _BinaryValue("infinity")),
        ("negative_infinity", _BinaryValue("infinity", sign=1)),
        ("quiet_nan", _BinaryValue("nan")),
    )


def _fp8_decode_payloads(row: Fp8Format) -> tuple[tuple[str, int], ...]:
    mantissa_mask = (1 << row.mantissa_bits) - 1
    one_payload = _fp8_encode_bits(row, _BinaryValue("finite", magnitude=Fraction(1)))
    maximum_finite = _fp8_maximum_finite_payload(row)
    candidates = [
        ("positive_zero", 0x00),
        ("negative_zero_or_nan", 0x80),
        ("minimum_subnormal", 0x01),
        ("negative_minimum_subnormal", 0x81),
        ("maximum_subnormal", mantissa_mask),
        ("negative_maximum_subnormal", 0x80 | mantissa_mask),
        ("minimum_normal", 1 << row.mantissa_bits),
        ("negative_minimum_normal", 0x80 | (1 << row.mantissa_bits)),
        ("one", one_payload),
        ("negative_one", 0x80 | one_payload),
        ("maximum_finite", maximum_finite),
        ("negative_maximum_finite", 0x80 | maximum_finite),
    ]
    special_policy = row.special_policy
    if special_policy == Fp8SpecialPolicy.IEEE:
        infinity = ((1 << row.exponent_bits) - 1) << row.mantissa_bits
        candidates.extend(
            (
                ("positive_infinity", infinity),
                ("positive_nan", 0x7F),
                ("negative_nan", 0xFF),
            )
        )
    elif special_policy == Fp8SpecialPolicy.FINITE_NAN:
        candidates.extend((("positive_nan", 0x7F), ("negative_nan", 0xFF)))

    seen = {payload for _, payload in candidates}
    for payload in _fp8_positive_finite_payloads(row):
        if len(candidates) >= 15:
            break
        if payload not in seen:
            candidates.append((f"finite_payload_{payload}", payload))
            seen.add(payload)
    return tuple(candidates[:15])


def _encode_exact_values(row: Fp8Format, source_format: _BinaryFormat) -> tuple[_ExactValue, ...]:
    values = []
    for name, intended_value in _fp8_boundary_values(row):
        source_bits = _binary_encode_bits(intended_value, source_format)
        source_value = _binary_decode_bits(source_bits, source_format)
        values.append(_ExactValue(name, source_bits, _fp8_encode_bits(row, source_value)))
    return tuple(values)


def _decode_exact_values(row: Fp8Format, result_format: _BinaryFormat) -> tuple[_ExactValue, ...]:
    return tuple(
        _ExactValue(
            name,
            payload,
            _binary_encode_bits(_fp8_decode_bits(row, payload), result_format),
        )
        for name, payload in _fp8_decode_payloads(row)
    )


def _decode_comparison_mask(row: Fp8Format, raw_bits: int, bit_count: int) -> int:
    all_bits = (1 << bit_count) - 1
    if _fp8_decode_bits(row, raw_bits).kind == "nan":
        return all_bits >> 1
    return all_bits


def _scalar_type_spelling(scalar_type: ScalarTypeKind) -> str:
    spellings = {
        ScalarTypeKind.F8E4M3: "f8E4M3",
        ScalarTypeKind.F8E5M2: "f8E5M2",
    }
    return spellings[scalar_type]


def _signed_literal(raw_bits: int, bit_count: int) -> int:
    sign_bit = 1 << (bit_count - 1)
    return raw_bits - (1 << bit_count) if raw_bits & sign_bit else raw_bits


def _generated_header(direction: str) -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        *line_comment_header(
            "//",
            generator=_GENERATOR,
            regenerate=REGENERATE_COMMAND,
        ),
        "//",
        f"// Exact schema-driven FP8 {direction} witnesses generated from the",
        "// target-independent numeric-format definitions. Heterogeneous",
        "// expected payloads are reduced to zero difference tensors on the device",
        "// so the complete goldens remain readable here without binary fixtures.",
        f"config.decl @fp8_bidirectional_{direction}.row_capacity : %value: index where [range(%value, 1, 4294967295)]",
        "",
    ]


def _emit_exact_constants(
    values: tuple[_ExactValue, ...],
    source_integer_type: str,
    source_bit_count: int,
    expected_integer_type: str,
    expected_bit_count: int,
    suffix: str,
) -> list[str]:
    lines: list[str] = []
    for value in values:
        lines.append(f"  %source_{value.name}_{suffix} = scalar.constant {_signed_literal(value.raw_bits, source_bit_count)} : {source_integer_type}")
        lines.append(f"  %expected_{value.name}_{suffix} = scalar.constant {_signed_literal(value.expected_bits, expected_bit_count)} : {expected_integer_type}")
    return lines


def _emit_exact_vector(
    values: tuple[_ExactValue, ...],
    source_integer_type: str,
    source_bit_count: int,
    expected_integer_type: str,
    expected_bit_count: int,
    vector_name: str,
    suffix: str,
) -> list[str]:
    names = ", ".join(f"%source_{value.name}_{suffix}" for value in values)
    expected_names = ", ".join(f"%expected_{value.name}_{suffix}" for value in values)
    width = len(values)
    return [
        f"  %source_bits_{vector_name} = vector.from_elements {names} : vector<{width}x{source_integer_type}>",
        f"  %expected_bits_{vector_name} = vector.from_elements {expected_names} : vector<{width}x{expected_integer_type}>",
    ]


def _emit_decode_comparison_vector(
    row: Fp8Format,
    values: tuple[_ExactValue, ...],
    integer_type: str,
    bit_count: int,
    vector_name: str,
) -> str:
    all_bits = (1 << bit_count) - 1
    masks = ", ".join("%comparison_mask_all_bits" if _decode_comparison_mask(row, value.raw_bits, bit_count) == all_bits else "%comparison_mask_without_sign" for value in values)
    return f"  %comparison_mask_{vector_name} = vector.from_elements {masks} : vector<{len(values)}x{integer_type}>"


def _emit_encode_pair(row: Fp8Format, source_format: _BinaryFormat) -> str:
    format_name = row.keyword
    schema_spelling = f"#encoding.{format_name}<payload_elements=8>"
    source_name = source_format.type_spelling
    carrier_type = _scalar_type_spelling(row.carrier_type)
    kernel_name = f"fp8_encode_{format_name}_{source_name}_exact"
    row_kernel_name = f"fp8_encode_{format_name}_{source_name}_rows"
    benchmark_name = f"{row_kernel_name}_benchmark"
    exact_case_name = f"{kernel_name}_case"
    row_case_name = f"{row_kernel_name}_case"
    values = _encode_exact_values(row, source_format)
    head = values[:8]
    tail = values[8:] + values[:1]
    exact_width = len(head) + len(tail)
    one_code = _fp8_encode_bits(row, _BinaryValue("finite", magnitude=Fraction(1)))
    lines = [
        f"// {source_name} to {format_name}: exact special values, RNE boundaries,",
        "// and two packed eight-lane conversions. The final zero lane pads the",
        "// fifteen semantic witnesses to two complete eight-lane conversions.",
        f"kernel.def @{kernel_name}() {{",
        "  %unit = index.constant 1 : index",
        "  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index",
        "} launch(%output: buffer) {",
        "  %base = index.constant 0 : offset",
        "  %head_offset = index.constant 0 : index",
        "  %tail_offset = index.constant 8 : index",
        f"  %output_view = buffer.view %output[%base] : buffer -> view<{exact_width}xi8>",
        f"  %head_schema = encoding.define {schema_spelling} : encoding<schema>",
        f"  %tail_schema = encoding.define {schema_spelling} : encoding<schema>",
        *_emit_exact_constants(
            values,
            source_format.integer_type_spelling,
            source_format.bit_count,
            "i8",
            8,
            "encode",
        ),
        *_emit_exact_vector(
            head,
            source_format.integer_type_spelling,
            source_format.bit_count,
            "i8",
            8,
            "head",
            "encode",
        ),
        f"  %source_head = vector.bitcast %source_bits_head : vector<8x{source_format.integer_type_spelling}> to vector<8x{source_name}>",
        f"  %encoded_head = vector.encode %source_head using %head_schema : vector<8x{source_name}>, encoding<schema> -> vector<8x{carrier_type}>",
        f"  %encoded_bits_head = vector.bitcast %encoded_head : vector<8x{carrier_type}> to vector<8xi8>",
        "  %difference_head = vector.xori %encoded_bits_head, %expected_bits_head : vector<8xi8>",
        f"  vector.store %difference_head, %output_view[%head_offset] : vector<8xi8>, view<{exact_width}xi8>",
        *_emit_exact_vector(
            tail,
            source_format.integer_type_spelling,
            source_format.bit_count,
            "i8",
            8,
            "tail",
            "encode",
        ),
        f"  %source_tail = vector.bitcast %source_bits_tail : vector<{len(tail)}x{source_format.integer_type_spelling}> to vector<{len(tail)}x{source_name}>",
        f"  %encoded_tail = vector.encode %source_tail using %tail_schema : vector<{len(tail)}x{source_name}>, encoding<schema> -> vector<{len(tail)}x{carrier_type}>",
        f"  %encoded_bits_tail = vector.bitcast %encoded_tail : vector<{len(tail)}x{carrier_type}> to vector<{len(tail)}xi8>",
        f"  %difference_tail = vector.xori %encoded_bits_tail, %expected_bits_tail : vector<{len(tail)}xi8>",
        f"  vector.store %difference_tail, %output_view[%tail_offset] : vector<{len(tail)}xi8>, view<{exact_width}xi8>",
        "  kernel.return",
        "}",
        "",
        f"kernel.def @{row_kernel_name}(%row_count: index) {{",
        "  %unit = index.constant 1 : index",
        "  %workgroup_size = index.constant 64 : index",
        "  %rounding = index.constant 63 : index",
        "  %row_capacity = config.get @fp8_bidirectional_encode.row_capacity : index",
        "  %rounded_rows = index.add %row_capacity, %rounding : index",
        "  %workgroups = index.div %rounded_rows, %workgroup_size : index",
        "  kernel.launch.config workgroups(%workgroups, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index",
        "} launch(%row_count: index, %input: buffer, %output: buffer) {",
        "  %base = index.constant 0 : offset",
        "  %row_capacity = config.get @fp8_bidirectional_encode.row_capacity : index",
        "  %bounded_row_count = index.assume %row_count [le(%row_count, %row_capacity)] : index",
        "  %workgroup = kernel.workgroup.id<x> : index",
        "  %lane = kernel.workitem.id<x> : index",
        "  %workgroup_size = index.constant 64 : index",
        "  %row = index.madd %workgroup, %workgroup_size, %lane : index",
        "  %column = index.constant 0 : index",
        "  %in_bounds = index.cmp ult, %row, %bounded_row_count : index",
        "  %input_noalias, %output_noalias = buffer.assume.noalias %input, %output : buffer, buffer",
        f"  %input_view = buffer.view %input_noalias[%base] : buffer -> view<[%bounded_row_count]x8x{source_name}>",
        "  %output_view = buffer.view %output_noalias[%base] : buffer -> view<[%bounded_row_count]x8xi8>",
        f"  %schema = encoding.define {schema_spelling} : encoding<schema>",
        "  scf.if %in_bounds {",
        f"    %source = vector.load %input_view[%row, %column] : view<[%bounded_row_count]x8x{source_name}> -> vector<8x{source_name}>",
        f"    %encoded = vector.encode %source using %schema : vector<8x{source_name}>, encoding<schema> -> vector<8x{carrier_type}>",
        f"    %encoded_bits = vector.bitcast %encoded : vector<8x{carrier_type}> to vector<8xi8>",
        "    vector.store %encoded_bits, %output_view[%row, %column] : vector<8xi8>, view<[%bounded_row_count]x8xi8>",
        "  }",
        "  kernel.return",
        "}",
        "",
        f"check.case public @{exact_case_name} {{",
        f"  %output = check.generate.fill value(0) : tensor<{exact_width}xi8>",
        f"  %expected = check.generate.fill value(0) : tensor<{exact_width}xi8>",
        f"  kernel.launch @{kernel_name}(%output) : (tensor<{exact_width}xi8>)",
        f"  check.expect.equal actual(%output) expected(%expected) : tensor<{exact_width}xi8>",
        "  check.return",
        "}",
        "",
        f"check.case public @{row_case_name} {{",
        '  %row_count = check.param.choice values([257, 65536]) name("row_count") : index',
        f"  %input = check.generate.fill value(1.0) : tensor<[%row_count]x8x{source_name}>",
        "  %output = check.generate.fill value(0) : tensor<[%row_count]x8xi8>",
        f"  %expected = check.generate.fill value({_signed_literal(one_code, 8)}) : tensor<[%row_count]x8xi8>",
        f"  kernel.launch @{row_kernel_name}[%row_count](%row_count, %input, %output) : [index](index, tensor<[%row_count]x8x{source_name}>, tensor<[%row_count]x8xi8>)",
        "  check.expect.equal actual(%output) expected(%expected) : tensor<[%row_count]x8xi8>",
        "  check.return",
        "}",
        "",
        f"check.benchmark<@{row_case_name}> @{benchmark_name} {{row_count = 65536}}",
        "",
    ]
    return "\n".join(lines)


def _emit_decode_pair(row: Fp8Format, result_format: _BinaryFormat) -> str:
    format_name = row.keyword
    schema_spelling = f"#encoding.{format_name}<payload_elements=8>"
    result_name = result_format.type_spelling
    carrier_type = _scalar_type_spelling(row.carrier_type)
    kernel_name = f"fp8_decode_{format_name}_{result_name}_exact"
    row_kernel_name = f"fp8_decode_{format_name}_{result_name}_rows"
    benchmark_name = f"{row_kernel_name}_benchmark"
    exact_case_name = f"{kernel_name}_case"
    row_case_name = f"{row_kernel_name}_case"
    values = _decode_exact_values(row, result_format)
    head = values[:8]
    tail = values[8:] + values[:1]
    exact_width = len(head) + len(tail)
    one_code = _fp8_encode_bits(row, _BinaryValue("finite", magnitude=Fraction(1)))
    output_integer_type = result_format.integer_type_spelling
    lines = [
        f"// {format_name} to {result_name}: exact special values and finite",
        "// boundaries through two packed eight-lane conversions. The final zero",
        "// lane pads the fifteen semantic witnesses to a complete register.",
        "// NaN sign is outside the numeric contract and is the only result bit",
        "// excluded from the exact comparison.",
        f"kernel.def @{kernel_name}() {{",
        "  %unit = index.constant 1 : index",
        "  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index",
        "} launch(%output: buffer) {",
        "  %base = index.constant 0 : offset",
        "  %head_offset = index.constant 0 : index",
        "  %tail_offset = index.constant 8 : index",
        f"  %output_view = buffer.view %output[%base] : buffer -> view<{exact_width}x{output_integer_type}>",
        f"  %head_schema = encoding.define {schema_spelling} : encoding<schema>",
        f"  %tail_schema = encoding.define {schema_spelling} : encoding<schema>",
        f"  %comparison_mask_all_bits = scalar.constant -1 : {output_integer_type}",
        f"  %comparison_mask_without_sign = scalar.constant {(1 << (result_format.bit_count - 1)) - 1} : {output_integer_type}",
        *_emit_exact_constants(
            values,
            "i8",
            8,
            output_integer_type,
            result_format.bit_count,
            "decode",
        ),
        *_emit_exact_vector(
            head,
            "i8",
            8,
            output_integer_type,
            result_format.bit_count,
            "head",
            "decode",
        ),
        _emit_decode_comparison_vector(
            row,
            head,
            output_integer_type,
            result_format.bit_count,
            "head",
        ),
        f"  %source_head = vector.bitcast %source_bits_head : vector<8xi8> to vector<8x{carrier_type}>",
        f"  %decoded_head = vector.decode %source_head using %head_schema : vector<8x{carrier_type}>, encoding<schema> -> vector<8x{result_name}>",
        f"  %decoded_bits_head = vector.bitcast %decoded_head : vector<8x{result_name}> to vector<8x{output_integer_type}>",
        f"  %raw_difference_head = vector.xori %decoded_bits_head, %expected_bits_head : vector<8x{output_integer_type}>",
        f"  %difference_head = vector.andi %raw_difference_head, %comparison_mask_head : vector<8x{output_integer_type}>",
        f"  vector.store %difference_head, %output_view[%head_offset] : vector<8x{output_integer_type}>, view<{exact_width}x{output_integer_type}>",
        *_emit_exact_vector(
            tail,
            "i8",
            8,
            output_integer_type,
            result_format.bit_count,
            "tail",
            "decode",
        ),
        _emit_decode_comparison_vector(
            row,
            tail,
            output_integer_type,
            result_format.bit_count,
            "tail",
        ),
        f"  %source_tail = vector.bitcast %source_bits_tail : vector<{len(tail)}xi8> to vector<{len(tail)}x{carrier_type}>",
        f"  %decoded_tail = vector.decode %source_tail using %tail_schema : vector<{len(tail)}x{carrier_type}>, encoding<schema> -> vector<{len(tail)}x{result_name}>",
        f"  %decoded_bits_tail = vector.bitcast %decoded_tail : vector<{len(tail)}x{result_name}> to vector<{len(tail)}x{output_integer_type}>",
        f"  %raw_difference_tail = vector.xori %decoded_bits_tail, %expected_bits_tail : vector<{len(tail)}x{output_integer_type}>",
        f"  %difference_tail = vector.andi %raw_difference_tail, %comparison_mask_tail : vector<{len(tail)}x{output_integer_type}>",
        f"  vector.store %difference_tail, %output_view[%tail_offset] : vector<{len(tail)}x{output_integer_type}>, view<{exact_width}x{output_integer_type}>",
        "  kernel.return",
        "}",
        "",
        f"kernel.def @{row_kernel_name}(%row_count: index) {{",
        "  %unit = index.constant 1 : index",
        "  %workgroup_size = index.constant 64 : index",
        "  %rounding = index.constant 63 : index",
        "  %row_capacity = config.get @fp8_bidirectional_decode.row_capacity : index",
        "  %rounded_rows = index.add %row_capacity, %rounding : index",
        "  %workgroups = index.div %rounded_rows, %workgroup_size : index",
        "  kernel.launch.config workgroups(%workgroups, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index",
        "} launch(%row_count: index, %input: buffer, %output: buffer) {",
        "  %base = index.constant 0 : offset",
        "  %row_capacity = config.get @fp8_bidirectional_decode.row_capacity : index",
        "  %bounded_row_count = index.assume %row_count [le(%row_count, %row_capacity)] : index",
        "  %workgroup = kernel.workgroup.id<x> : index",
        "  %lane = kernel.workitem.id<x> : index",
        "  %workgroup_size = index.constant 64 : index",
        "  %row = index.madd %workgroup, %workgroup_size, %lane : index",
        "  %column = index.constant 0 : index",
        "  %in_bounds = index.cmp ult, %row, %bounded_row_count : index",
        "  %input_noalias, %output_noalias = buffer.assume.noalias %input, %output : buffer, buffer",
        "  %input_view = buffer.view %input_noalias[%base] : buffer -> view<[%bounded_row_count]x8xi8>",
        f"  %output_view = buffer.view %output_noalias[%base] : buffer -> view<[%bounded_row_count]x8x{result_name}>",
        f"  %schema = encoding.define {schema_spelling} : encoding<schema>",
        "  scf.if %in_bounds {",
        "    %source_bits = vector.load %input_view[%row, %column] : view<[%bounded_row_count]x8xi8> -> vector<8xi8>",
        f"    %source = vector.bitcast %source_bits : vector<8xi8> to vector<8x{carrier_type}>",
        f"    %decoded = vector.decode %source using %schema : vector<8x{carrier_type}>, encoding<schema> -> vector<8x{result_name}>",
        f"    vector.store %decoded, %output_view[%row, %column] : vector<8x{result_name}>, view<[%bounded_row_count]x8x{result_name}>",
        "  }",
        "  kernel.return",
        "}",
        "",
        f"check.case public @{exact_case_name} {{",
        f"  %output = check.generate.fill value(0) : tensor<{exact_width}x{output_integer_type}>",
        f"  %expected = check.generate.fill value(0) : tensor<{exact_width}x{output_integer_type}>",
        f"  kernel.launch @{kernel_name}(%output) : (tensor<{exact_width}x{output_integer_type}>)",
        f"  check.expect.equal actual(%output) expected(%expected) : tensor<{exact_width}x{output_integer_type}>",
        "  check.return",
        "}",
        "",
        f"check.case public @{row_case_name} {{",
        '  %row_count = check.param.choice values([257, 65536]) name("row_count") : index',
        f"  %input = check.generate.fill value({_signed_literal(one_code, 8)}) : tensor<[%row_count]x8xi8>",
        f"  %output = check.generate.fill value(0.0) : tensor<[%row_count]x8x{result_name}>",
        f"  %expected = check.generate.fill value(1.0) : tensor<[%row_count]x8x{result_name}>",
        f"  kernel.launch @{row_kernel_name}[%row_count](%row_count, %input, %output) : [index](index, tensor<[%row_count]x8xi8>, tensor<[%row_count]x8x{result_name}>)",
        f"  check.expect.bitwise actual(%output) expected(%expected) : tensor<[%row_count]x8x{result_name}>",
        "  check.return",
        "}",
        "",
        f"check.benchmark<@{row_case_name}> @{benchmark_name} {{row_count = 65536}}",
        "",
    ]
    return "\n".join(lines)


def _fp8_format_family(row: Fp8Format) -> Literal["e4m3", "e5m2"]:
    if row.carrier_type == ScalarTypeKind.F8E4M3:
        return "e4m3"
    if row.carrier_type == ScalarTypeKind.F8E5M2:
        return "e5m2"
    raise ValueError(f"unsupported FP8 carrier type: {row.carrier_type.name}")


def emit_fp8_encode_matrix(family: Literal["e4m3", "e5m2"]) -> str:
    return "\n".join(
        [
            *_generated_header("encode"),
            *(_emit_encode_pair(row, source_format) for row in FP8_FORMATS if _fp8_format_family(row) == family for source_format in _BINARY_FORMATS),
        ]
    )


def emit_fp8_decode_matrix(family: Literal["e4m3", "e5m2"]) -> str:
    return "\n".join(
        [
            *_generated_header("decode"),
            *(_emit_decode_pair(row, result_format) for row in FP8_FORMATS if _fp8_format_family(row) == family for result_format in _BINARY_FORMATS),
        ]
    )


def checked_in_file_set() -> GeneratedFileSet:
    """Returns the checked-in FP8 conversion witness ownership set."""
    return GeneratedFileSet.from_mapping(
        {
            (_OUTPUT_ROOT / "fp8_bidirectional_decode_e4m3.loom").as_posix(): emit_fp8_decode_matrix("e4m3"),
            (_OUTPUT_ROOT / "fp8_bidirectional_decode_e5m2.loom").as_posix(): emit_fp8_decode_matrix("e5m2"),
            (_OUTPUT_ROOT / "fp8_bidirectional_encode_e4m3.loom").as_posix(): emit_fp8_encode_matrix("e4m3"),
            (_OUTPUT_ROOT / "fp8_bidirectional_encode_e5m2.loom").as_posix(): emit_fp8_encode_matrix("e5m2"),
        }
    )


def maintain_checked_in_files(
    mode: GeneratedFileMaintenanceMode,
) -> GeneratedFileMaintenanceResult:
    """Checks or updates all checked-in FP8 conversion witnesses."""
    return maintain_generated_file_set(
        _bootstrap.find_repo_root(),
        checked_in_file_set(),
        mode=mode,
        description=DESCRIPTION,
        regenerate_command=REGENERATE_COMMAND,
    )


def _write_output(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    maintenance_mode = parser.add_mutually_exclusive_group()
    maintenance_mode.add_argument(
        "--check",
        action="store_true",
        help="Check the checked-in generated witnesses.",
    )
    maintenance_mode.add_argument(
        "--in-place",
        action="store_true",
        help="Regenerate the checked-in generated witnesses.",
    )
    parser.add_argument("--encode-e4m3-output", type=Path)
    parser.add_argument("--encode-e5m2-output", type=Path)
    parser.add_argument("--decode-e4m3-output", type=Path)
    parser.add_argument("--decode-e5m2-output", type=Path)
    args = parser.parse_args(argv)
    explicit_outputs = (
        args.encode_e4m3_output,
        args.encode_e5m2_output,
        args.decode_e4m3_output,
        args.decode_e5m2_output,
    )
    if args.check or args.in_place:
        if any(output is not None for output in explicit_outputs):
            parser.error("checked-in maintenance modes cannot be combined with explicit outputs")
        result = maintain_checked_in_files("update" if args.in_place else "check")
        return 0 if result.ok else 1
    if all(output is None for output in explicit_outputs):
        parser.error("expected --check, --in-place, or at least one explicit output")
    if args.encode_e4m3_output is not None:
        _write_output(args.encode_e4m3_output, emit_fp8_encode_matrix("e4m3"))
    if args.encode_e5m2_output is not None:
        _write_output(args.encode_e5m2_output, emit_fp8_encode_matrix("e5m2"))
    if args.decode_e4m3_output is not None:
        _write_output(args.decode_e4m3_output, emit_fp8_decode_matrix("e4m3"))
    if args.decode_e5m2_output is not None:
        _write_output(args.decode_e5m2_output, emit_fp8_decode_matrix("e5m2"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
