# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent exact arithmetic oracle for conversion test vectors."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
from fractions import Fraction
from typing import Literal

from model.isa.selectors import SELECTOR_VALUES, parse_integer_conversion_name


@dataclasses.dataclass(frozen=True, slots=True)
class _BinaryFormat:
    bit_count: int
    exponent_bit_count: int
    mantissa_bit_count: int
    exponent_bias: int
    finite_nan: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _BinaryValue:
    kind: Literal["finite", "infinity", "nan"]
    sign: int = 0
    magnitude: Fraction = Fraction(0)


_FORMATS = {
    "f8e4m3": _BinaryFormat(8, 4, 3, 7, finite_nan=True),
    "f8e5m2": _BinaryFormat(8, 5, 2, 15),
    "f16": _BinaryFormat(16, 5, 10, 15),
    "bf16": _BinaryFormat(16, 8, 7, 127),
    "f32": _BinaryFormat(32, 8, 23, 127),
    "f64": _BinaryFormat(64, 11, 52, 1023),
}


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _round_nearest_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled_remainder = remainder * 2
    if doubled_remainder < value.denominator:
        return quotient
    if doubled_remainder > value.denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def _floor_log2(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent - 1 if value < _pow2(exponent) else exponent


def _decode(bits: int, binary_format: _BinaryFormat) -> _BinaryValue:
    sign = (bits >> (binary_format.bit_count - 1)) & 1
    exponent_mask = (1 << binary_format.exponent_bit_count) - 1
    mantissa_mask = (1 << binary_format.mantissa_bit_count) - 1
    exponent = (bits >> binary_format.mantissa_bit_count) & exponent_mask
    mantissa = bits & mantissa_mask
    if exponent == exponent_mask:
        if binary_format.finite_nan:
            if mantissa == mantissa_mask:
                return _BinaryValue("nan", sign)
        else:
            return _BinaryValue("infinity" if mantissa == 0 else "nan", sign)
    if exponent == 0:
        magnitude = Fraction(mantissa) * _pow2(
            1 - binary_format.exponent_bias - binary_format.mantissa_bit_count
        )
    else:
        magnitude = Fraction(
            (1 << binary_format.mantissa_bit_count) | mantissa
        ) * _pow2(
            exponent - binary_format.exponent_bias - binary_format.mantissa_bit_count
        )
    return _BinaryValue("finite", sign, magnitude)


def _encode(value: _BinaryValue, binary_format: _BinaryFormat) -> int:
    sign_bits = value.sign << (binary_format.bit_count - 1)
    exponent_mask = (1 << binary_format.exponent_bit_count) - 1
    mantissa_mask = (1 << binary_format.mantissa_bit_count) - 1
    if value.kind == "nan":
        quiet_mantissa = (
            mantissa_mask
            if binary_format.finite_nan
            else 1 << (binary_format.mantissa_bit_count - 1)
        )
        return (exponent_mask << binary_format.mantissa_bit_count) | quiet_mantissa
    if value.kind == "infinity":
        if binary_format.finite_nan:
            return (
                sign_bits
                | (exponent_mask << binary_format.mantissa_bit_count)
                | (mantissa_mask - 1)
            )
        return sign_bits | (exponent_mask << binary_format.mantissa_bit_count)
    if value.magnitude == 0:
        return sign_bits

    exponent = _floor_log2(value.magnitude)
    minimum_normal_exponent = 1 - binary_format.exponent_bias
    maximum_normal_exponent = (
        exponent_mask
        - binary_format.exponent_bias
        - (0 if binary_format.finite_nan else 1)
    )
    if exponent < minimum_normal_exponent:
        quantum = _pow2(minimum_normal_exponent - binary_format.mantissa_bit_count)
        mantissa = _round_nearest_even(value.magnitude / quantum)
        if mantissa == 0:
            return sign_bits
        if mantissa < (1 << binary_format.mantissa_bit_count):
            return sign_bits | mantissa
        exponent = minimum_normal_exponent
        mantissa = 0
    else:
        rounded_significand = _round_nearest_even(
            value.magnitude / _pow2(exponent) * (1 << binary_format.mantissa_bit_count)
        )
        if rounded_significand == 1 << (binary_format.mantissa_bit_count + 1):
            rounded_significand >>= 1
            exponent += 1
        if exponent > maximum_normal_exponent:
            if binary_format.finite_nan:
                return (
                    sign_bits
                    | (exponent_mask << binary_format.mantissa_bit_count)
                    | (mantissa_mask - 1)
                )
            return sign_bits | (exponent_mask << binary_format.mantissa_bit_count)
        mantissa = rounded_significand - (1 << binary_format.mantissa_bit_count)
    encoded_exponent = exponent + binary_format.exponent_bias
    if (
        binary_format.finite_nan
        and encoded_exponent == exponent_mask
        and mantissa == mantissa_mask
    ):
        mantissa -= 1
    return sign_bits | (encoded_exponent << binary_format.mantissa_bit_count) | mantissa


def _selector_rows(table_id: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (selector.value, selector.name)
        for selector in SELECTOR_VALUES
        if selector.table_id == table_id
    )


def _integer_conversion_rows() -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    dirty_high_bits = 0xA55AA55AA55AA55A
    for selector, name in _selector_rows("core.selector.integer.convert"):
        operation, source_bit_count, destination_bit_count = (
            parse_integer_conversion_name(name)
        )
        source_mask = (1 << source_bit_count) - 1
        destination_mask = (1 << destination_bit_count) - 1
        samples = {
            0,
            1,
            source_mask,
            1 << (source_bit_count - 1),
            0xFEDCBA9876543210 & source_mask,
        }
        for low_source_bits in sorted(samples):
            source_bits = low_source_bits | (dirty_high_bits & ~source_mask)
            result = low_source_bits
            if operation == "sign_extend" and (result & (1 << (source_bit_count - 1))):
                result |= ~source_mask
            rows.append((selector, source_bits, result & destination_mask))
    return tuple(rows)


def _neighbor_bits(bits: int, binary_format: _BinaryFormat) -> tuple[int, ...]:
    sign_mask = 1 << (binary_format.bit_count - 1)
    magnitude = bits & (sign_mask - 1)
    return tuple(
        (bits & sign_mask) | neighbor
        for neighbor in range(max(0, magnitude - 1), magnitude + 2)
        if neighbor < sign_mask
    )


_TARGET_BOUNDARIES = {
    "f8e4m3": (0x00, 0x07, 0x37, 0x38, 0x7D),
    "f8e5m2": (0x00, 0x03, 0x3B, 0x3C, 0x7A),
    "f16": (0x0000, 0x03FF, 0x3BFF, 0x3C00, 0x7BFE),
    "bf16": (0x0000, 0x007F, 0x3F7F, 0x3F80, 0x7F7E),
    "f32": (0x00000000, 0x007FFFFF, 0x3F7FFFFF, 0x3F800000, 0x7F7FFFFE),
    "f64": (
        0x0000000000000000,
        0x000FFFFFFFFFFFFF,
        0x3FEFFFFFFFFFFFFF,
        0x3FF0000000000000,
        0x7FEFFFFFFFFFFFFE,
    ),
}


def _float_source_samples(
    source_format: _BinaryFormat, target_format_name: str
) -> tuple[int, ...]:
    exponent_mask = (1 << source_format.exponent_bit_count) - 1
    mantissa_mask = (1 << source_format.mantissa_bit_count) - 1
    sign_mask = 1 << (source_format.bit_count - 1)
    infinity = exponent_mask << source_format.mantissa_bit_count
    samples = {
        0,
        sign_mask,
        1,
        sign_mask | 1,
        mantissa_mask,
        1 << source_format.mantissa_bit_count,
        source_format.exponent_bias << source_format.mantissa_bit_count,
        (source_format.exponent_bias << source_format.mantissa_bit_count) - 1,
        (source_format.exponent_bias << source_format.mantissa_bit_count) + 1,
        infinity - 1,
        sign_mask | (infinity - 1),
        infinity,
        sign_mask | infinity,
        infinity | 1,
        sign_mask | infinity | 1,
    }
    target_format = _FORMATS[target_format_name]
    for lower_bits in _TARGET_BOUNDARIES[target_format_name]:
        lower = _decode(lower_bits, target_format)
        upper = _decode(lower_bits + 1, target_format)
        midpoint = (lower.magnitude + upper.magnitude) / 2
        for sign in (0, 1):
            midpoint_bits = _encode(
                _BinaryValue("finite", sign, midpoint), source_format
            )
            samples.update(_neighbor_bits(midpoint_bits, source_format))
    target_exponent_mask = (1 << target_format.exponent_bit_count) - 1
    target_mantissa_mask = (1 << target_format.mantissa_bit_count) - 1
    maximum_exponent_field = (
        target_exponent_mask if target_format.finite_nan else target_exponent_mask - 1
    )
    maximum_mantissa = (
        target_mantissa_mask - 1 if target_format.finite_nan else target_mantissa_mask
    )
    maximum_finite = Fraction(
        (1 << target_format.mantissa_bit_count) | maximum_mantissa
    ) * _pow2(
        maximum_exponent_field
        - target_format.exponent_bias
        - target_format.mantissa_bit_count
    )
    overflow_quantum = _pow2(
        maximum_exponent_field
        - target_format.exponent_bias
        - target_format.mantissa_bit_count
    )
    overflow_midpoint = maximum_finite + overflow_quantum / 2
    for sign in (0, 1):
        midpoint_bits = _encode(
            _BinaryValue("finite", sign, overflow_midpoint), source_format
        )
        samples.update(_neighbor_bits(midpoint_bits, source_format))
    return tuple(sorted(samples))


def _float_truncate_rows() -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    for selector, name in _selector_rows("core.selector.float.truncate"):
        source_name, target_name = name.split(".to.")
        source_format = _FORMATS[source_name]
        target_format = _FORMATS[target_name]
        for source_bits in _float_source_samples(source_format, target_name):
            expected = _encode(_decode(source_bits, source_format), target_format)
            rows.append((selector, source_bits, expected))
    return tuple(rows)


def _float_width_rows() -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    for selector, name in _selector_rows("core.selector.float.width"):
        source_name, target_name = name.split(".to.")
        source_format = _FORMATS[source_name]
        target_format = _FORMATS[target_name]
        for source_bits in _float_source_samples(source_format, target_name):
            expected = _encode(_decode(source_bits, source_format), target_format)
            rows.append((selector, source_bits, expected))
    return tuple(rows)


def _integer_to_float_rows() -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    for selector, name in _selector_rows("core.selector.integer.to.float"):
        source_name, target_name = name.split(".to.")
        is_signed = source_name.startswith("s")
        source_bit_count = int(source_name[1:])
        target_format = _FORMATS[target_name]
        precision = target_format.mantissa_bit_count + 1
        values = {0, 1, 2}
        for delta in (-3, -2, -1, 0, 1, 2, 3):
            candidate = (1 << precision) + delta
            if candidate >= 0:
                values.add(candidate)
        if is_signed:
            values.update((-1, -2, -(1 << (source_bit_count - 1))))
            values.add((1 << (source_bit_count - 1)) - 1)
        else:
            values.add((1 << source_bit_count) - 1)
        if target_name == "bf16" and source_bit_count == 64:
            for exponent in (31, 40, 62):
                lower = 129 << (exponent - 7)
                midpoint = lower + (1 << (exponent - 8))
                values.update((midpoint - 1, midpoint, midpoint + 1))
                if is_signed:
                    values.update((-midpoint - 1, -midpoint, -midpoint + 1))
        source_mask = (1 << source_bit_count) - 1
        for integer in sorted(values):
            if is_signed:
                if not (
                    -(1 << (source_bit_count - 1))
                    <= integer
                    < (1 << (source_bit_count - 1))
                ):
                    continue
            elif not (0 <= integer <= source_mask):
                continue
            value = _BinaryValue("finite", int(integer < 0), Fraction(abs(integer)))
            rows.append(
                (selector, integer & source_mask, _encode(value, target_format))
            )
    return tuple(rows)


def _float_to_integer_result(
    source: _BinaryValue, destination_name: str
) -> tuple[str, int]:
    if source.kind == "nan":
        return "NAN", 0
    if source.kind == "infinity":
        return "OUT_OF_RANGE", 0
    destination_is_signed = destination_name.startswith("s")
    destination_bit_count = int(destination_name[1:])
    mathematical_value = -source.magnitude if source.sign else source.magnitude
    lower = (
        -Fraction((1 << (destination_bit_count - 1)) + 1)
        if destination_is_signed
        else Fraction(-1)
    )
    upper = Fraction(
        1
        << (
            destination_bit_count - 1
            if destination_is_signed
            else destination_bit_count
        )
    )
    if not lower < mathematical_value < upper:
        return "OUT_OF_RANGE", 0
    magnitude = source.magnitude.numerator // source.magnitude.denominator
    integer = -magnitude if source.sign else magnitude
    return "NONE", integer & ((1 << destination_bit_count) - 1)


def _float_to_integer_rows() -> tuple[tuple[int, int, str, int], ...]:
    rows: list[tuple[int, int, str, int]] = []
    for selector, name in _selector_rows("core.selector.float.to.integer"):
        source_name, destination_name = name.split(".to.")
        source_format = _FORMATS[source_name]
        destination_is_signed = destination_name.startswith("s")
        destination_bit_count = int(destination_name[1:])
        exponent_mask = (1 << source_format.exponent_bit_count) - 1
        infinity = exponent_mask << source_format.mantissa_bit_count
        samples = {
            0,
            1 << (source_format.bit_count - 1),
            _encode(_BinaryValue("finite", 0, Fraction(1, 2)), source_format),
            _encode(_BinaryValue("finite", 1, Fraction(1, 2)), source_format),
            _encode(_BinaryValue("finite", 0, Fraction(3, 2)), source_format),
            _encode(_BinaryValue("finite", 1, Fraction(3, 2)), source_format),
            infinity,
            (1 << (source_format.bit_count - 1)) | infinity,
            infinity | 1,
            (1 << (source_format.bit_count - 1)) | infinity | 1,
        }
        upper = Fraction(
            1
            << (
                destination_bit_count - 1
                if destination_is_signed
                else destination_bit_count
            )
        )
        lower_magnitude = Fraction(
            (1 << (destination_bit_count - 1)) + 1 if destination_is_signed else 1
        )
        for sign, boundary in ((0, upper), (1, lower_magnitude)):
            boundary_bits = _encode(
                _BinaryValue("finite", sign, boundary), source_format
            )
            samples.update(_neighbor_bits(boundary_bits, source_format))
        for source_bits in sorted(samples):
            failure, expected = _float_to_integer_result(
                _decode(source_bits, source_format), destination_name
            )
            rows.append((selector, source_bits, failure, expected))
    return tuple(rows)


def _u64(value: int) -> str:
    return f"UINT64_C(0x{value:016X})"


def render_conversion_test_vectors() -> str:
    """Renders one multi-include exact conversion witness corpus."""

    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED FILE: DO NOT EDIT.",
        "// Independent Fraction-based conversion boundary witnesses.",
        "// clang-format off",
        "",
        "#if defined(IREE_VM_BYTECODE_DEFINE_INTEGER_TEST_ROWS)",
    ]
    lines.extend(
        f"IREE_VM_BYTECODE_INTEGER_TEST_ROW(0x{selector:02X}, {_u64(source)}, {_u64(expected)})"
        for selector, source, expected in _integer_conversion_rows()
    )
    lines.append("#elif defined(IREE_VM_BYTECODE_DEFINE_FLOAT_TRUNCATE_TEST_ROWS)")
    lines.extend(
        f"IREE_VM_BYTECODE_FLOAT_TRUNCATE_TEST_ROW(0x{selector:02X}, {_u64(source)}, {_u64(expected)})"
        for selector, source, expected in _float_truncate_rows()
    )
    lines.append("#elif defined(IREE_VM_BYTECODE_DEFINE_FLOAT_WIDTH_TEST_ROWS)")
    lines.extend(
        f"IREE_VM_BYTECODE_FLOAT_WIDTH_TEST_ROW(0x{selector:02X}, {_u64(source)}, {_u64(expected)})"
        for selector, source, expected in _float_width_rows()
    )
    lines.append("#elif defined(IREE_VM_BYTECODE_DEFINE_INTEGER_TO_FLOAT_TEST_ROWS)")
    lines.extend(
        f"IREE_VM_BYTECODE_INTEGER_TO_FLOAT_TEST_ROW(0x{selector:02X}, {_u64(source)}, {_u64(expected)})"
        for selector, source, expected in _integer_to_float_rows()
    )
    lines.append("#elif defined(IREE_VM_BYTECODE_DEFINE_FLOAT_TO_INTEGER_TEST_ROWS)")
    lines.extend(
        "IREE_VM_BYTECODE_FLOAT_TO_INTEGER_TEST_ROW("
        f"0x{selector:02X}, {_u64(source)}, {failure}, {_u64(expected)})"
        for selector, source, failure, expected in _float_to_integer_rows()
    )
    lines.extend(
        (
            "#else",
            '#error "define one VM conversion test-vector projection"',
            "#endif",
            "",
            "// clang-format on",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate exact VM conversion test vectors."
    )
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    contents = render_conversion_test_vectors()
    if (
        not arguments.output_file.is_file()
        or arguments.output_file.read_text(encoding="utf-8") != contents
    ):
        arguments.output_file.write_text(contents, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
