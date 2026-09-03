# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact integer- and floating-point scalar conversions."""

from __future__ import annotations

from typing import NamedTuple

from iree.vm.bytecode.spec.isa import (
    FailureCase,
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
)
from iree.vm.bytecode.spec.isa.core.rules import FieldRule
from iree.vm.bytecode.spec.schema import (
    U8,
    Field,
    NumericKind,
    NumericTable,
    NumericValue,
)
from iree.vm.bytecode.spec.version import CORE_0


def _selector(
    name: str, summary: str, values: tuple[tuple[str, int, str], ...]
) -> NumericTable:
    return NumericTable(
        name,
        U8,
        NumericKind.SELECTOR,
        tuple(
            NumericValue(value_name, value, CORE_0, meaning)
            for value_name, value, meaning in values
        ),
        CORE_0,
        summary,
    )


_INTEGER_WIDTHS = (1, 8, 16, 32, 64)


def _integer_convert_values() -> tuple[tuple[str, int, str], ...]:
    values: list[tuple[str, int, str]] = []
    for source_index, source_width in enumerate(_INTEGER_WIDTHS):
        for destination_width in _INTEGER_WIDTHS[source_index + 1 :]:
            values.append(
                (
                    f"s{source_width}.to.i{destination_width}",
                    len(values),
                    f"Sign-extends the low {source_width} bits to "
                    f"i{destination_width}.",
                )
            )
            values.append(
                (
                    f"u{source_width}.to.i{destination_width}",
                    len(values),
                    f"Zero-extends the low {source_width} bits to "
                    f"i{destination_width}.",
                )
            )
    for source_index, source_width in enumerate(_INTEGER_WIDTHS):
        for destination_width in _INTEGER_WIDTHS[:source_index]:
            values.append(
                (
                    f"i{source_width}.to.i{destination_width}",
                    len(values),
                    f"Preserves the low {destination_width} bits and clears "
                    "all higher bits.",
                )
            )
    return tuple(values)


def _float_to_integer_values() -> tuple[tuple[str, int, str], ...]:
    values: list[tuple[str, int, str]] = []
    for source_width in (32, 64):
        for destination_width in _INTEGER_WIDTHS:
            signed_lower = -(1 << (destination_width - 1)) - 1
            signed_upper = 1 << (destination_width - 1)
            values.append(
                (
                    f"f{source_width}.to.s{destination_width}",
                    len(values),
                    f"Truncates f{source_width} in ({signed_lower}, "
                    f"{signed_upper}) to signed i{destination_width}.",
                )
            )
            unsigned_upper = 1 << destination_width
            values.append(
                (
                    f"f{source_width}.to.u{destination_width}",
                    len(values),
                    f"Truncates f{source_width} in (-1, {unsigned_upper}) to "
                    f"unsigned i{destination_width}.",
                )
            )
    return tuple(values)


INTEGER_CONVERT_SELECTOR = _selector(
    "integer.convert",
    (
        "Selects an exact low-bit integer truncation or extension. Results clear "
        "every cell bit above their declared destination width."
    ),
    _integer_convert_values(),
)

FLOAT_EXTEND_SELECTOR = _selector(
    "float.extend",
    (
        "Selects one narrow source format to extend structurally to its exact f32 "
        "value; NaNs produce a quiet f32 arithmetic NaN."
    ),
    (
        ("f8e4m3.to.f32", 0, "Extends low E4M3FN bits exactly to f32."),
        ("f8e5m2.to.f32", 1, "Extends low E5M2 bits exactly to f32."),
        ("f16.to.f32", 2, "Extends low IEEE binary16 bits exactly to f32."),
        ("bf16.to.f32", 3, "Extends low bfloat16 bits exactly to f32."),
    ),
)

FLOAT_TRUNCATE_SELECTOR = _selector(
    "float.truncate",
    (
        "Selects source and narrow destination formats for one direct nearest-even "
        "conversion with gradual subnormals. E4M3FN overflow and infinity saturate "
        "to signed 448; other overflow produces infinity."
    ),
    (
        ("f32.to.f8e4m3", 0, "Rounds f32 directly to E4M3FN."),
        ("f32.to.f8e5m2", 1, "Rounds f32 directly to E5M2."),
        ("f32.to.f16", 2, "Rounds f32 directly to IEEE binary16."),
        ("f32.to.bf16", 3, "Rounds f32 directly to bfloat16."),
        (
            "f64.to.f8e4m3",
            4,
            "Rounds f64 directly to E4M3FN without f32 staging.",
        ),
        (
            "f64.to.f8e5m2",
            5,
            "Rounds f64 directly to E5M2 without f32 staging.",
        ),
        (
            "f64.to.f16",
            6,
            "Rounds f64 directly to IEEE binary16 without f32 staging.",
        ),
        (
            "f64.to.bf16",
            7,
            "Rounds f64 directly to bfloat16 without f32 staging.",
        ),
    ),
)

FLOAT_WIDTH_SELECTOR = _selector(
    "float.width",
    "Selects one nearest-even conversion between IEEE binary32 and binary64.",
    (
        ("f32.to.f64", 0, "Extends every finite f32 exactly to f64."),
        (
            "f64.to.f32",
            1,
            "Rounds f64 to f32 with gradual subnormals and infinity on overflow.",
        ),
    ),
)

INTEGER_TO_FLOAT_SELECTOR = _selector(
    "integer.to.float",
    (
        "Selects source signedness and width plus destination float format. The "
        "exact integer is rounded directly nearest-even without intermediate staging."
    ),
    (
        ("s32.to.f32", 0, "Rounds signed low i32 directly to f32."),
        ("u32.to.f32", 1, "Rounds unsigned low i32 directly to f32."),
        ("s32.to.f64", 2, "Rounds signed low i32 directly to f64."),
        ("u32.to.f64", 3, "Rounds unsigned low i32 directly to f64."),
        ("s64.to.f32", 4, "Rounds signed i64 directly to f32."),
        ("u64.to.f32", 5, "Rounds unsigned i64 directly to f32."),
        ("s64.to.f64", 6, "Rounds signed i64 directly to f64."),
        ("u64.to.f64", 7, "Rounds unsigned i64 directly to f64."),
        ("s32.to.bf16", 8, "Rounds signed low i32 directly to bfloat16."),
        ("u32.to.bf16", 9, "Rounds unsigned low i32 directly to bfloat16."),
        ("s64.to.bf16", 10, "Rounds signed i64 directly to bfloat16."),
        ("u64.to.bf16", 11, "Rounds unsigned i64 directly to bfloat16."),
    ),
)

FLOAT_TO_INTEGER_SELECTOR = _selector(
    "float.to.integer",
    (
        "Selects a finite f32/f64 source and signed or unsigned 1-, 8-, 16-, "
        "32-, or 64-bit integer destination. Successful values truncate toward "
        "zero; NaN fails invalid_argument and values outside the destination's "
        "strict source interval fail out_of_range."
    ),
    _float_to_integer_values(),
)

CONVERSION_SELECTORS = (
    INTEGER_CONVERT_SELECTOR,
    FLOAT_EXTEND_SELECTOR,
    FLOAT_TRUNCATE_SELECTOR,
    FLOAT_WIDTH_SELECTOR,
    INTEGER_TO_FLOAT_SELECTOR,
    FLOAT_TO_INTEGER_SELECTOR,
)


CONVERSION_FAMILY = InstructionFamily(
    name="conversion",
    since=CORE_0,
    summary="Exact integer- and floating-point scalar conversions.",
    contract=(
        "All conversion records read the complete source cell before publishing the "
        "destination, so source and destination may alias. Narrow inputs ignore bits "
        "above their declared width and narrow results clear all unused high cell "
        "bits. Floating conversions are structural operations independent of ambient "
        "host rounding, FTZ, and DAZ modes. They use round-to-nearest with ties to an "
        "even least-significant significand bit, gradual subnormals, signed zero, and "
        "quiet arithmetic NaN results whose sign and payload are unspecified.\n\nF8e4m3 "
        "is PyTorch E4M3FN: its maximum finite magnitude is 448, minimum normal is "
        "2^-6, minimum subnormal is 2^-9, 0x7E is positive 448, 0x7F is NaN, and "
        "there is no infinity. F8e5m2 has maximum finite magnitude 57344, minimum "
        "normal 2^-14, and minimum subnormal 2^-16. IEEE f16 has maximum finite "
        "magnitude 65504, minimum normal 2^-14, and minimum subnormal 2^-24. Bf16 "
        "has maximum finite magnitude (2-2^-7)*2^127, minimum normal 2^-126, and "
        "minimum subnormal 2^-133. F8e5m2, f16, bf16, f32, and f64 reserve their "
        "all-one exponent for infinity and NaN. Every narrow format has signed zero "
        "and gradual subnormals.\n\nNo conversion accesses refs or suspends."
    ),
)


def _field(
    name: str,
    summary: str,
    role: FieldRole,
    rule,
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, U8, summary), role, rule)


class _ConversionDefinition(NamedTuple):
    opcode: int
    mnemonic: str
    selector: NumericTable
    summary: str
    behavior: str
    success: str
    pseudocode: str
    preconditions: tuple[str, ...] = ()
    failures: tuple[FailureCase, ...] = ()


def _conversion(definition: _ConversionDefinition) -> Instruction:
    (
        opcode,
        mnemonic,
        selector,
        summary,
        behavior,
        success,
        pseudocode,
        preconditions,
        failures,
    ) = definition
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=CONVERSION_FAMILY,
        summary=summary,
        fields=(
            _field(
                "destination_v8",
                "Value-register ordinal receiving the result.",
                FieldRole.RESULT,
                FieldRule.REGISTER_VALUE,
            ),
            _field(
                "source_v8",
                "Source value-register ordinal.",
                FieldRole.OPERAND,
                FieldRule.REGISTER_VALUE,
            ),
            _field(
                "selector_u8",
                f"Closed {selector.name} operation selector.",
                FieldRole.IMMEDIATE,
                FieldRuleUse(FieldRule.SELECTOR, data=selector),
            ),
        ),
        semantics=None,
        behavior=behavior,
        success=(success,),
        assembly=f"%v<destination> = {mnemonic} %v<source> {{selector}}",
        pseudocode=pseudocode,
        preconditions=preconditions,
        failures=failures,
        ownership=(
            "The source is read before destination publication and no ref ownership "
            "state is accessed or changed.",
        ),
    )


CONVERSION_INSTRUCTIONS = tuple(
    _conversion(definition)
    for definition in (
        _ConversionDefinition(
            0xA0,
            "conversion.integer",
            INTEGER_CONVERT_SELECTOR,
            "Truncates or extends one integer bit pattern.",
            (
                "Performs the selected exact conversion between distinct i1, i8, "
                "i16, i32, and i64 widths. Signed extensions propagate the selected "
                "source sign bit, unsigned extensions insert zeros, and truncations "
                "preserve the selected low destination-width bits."
            ),
            (
                "destination_v8 receives the selected truncation or extension and "
                "clears every cell bit above the destination width."
            ),
            (
                "source_bits = values[source_v8];\n"
                "result_bits = evaluate_integer_conversion(selector_u8, source_bits);\n"
                "values[destination_v8] = result_bits;\n"
                "pc = pc + 4;"
            ),
        ),
        _ConversionDefinition(
            0xA1,
            "conversion.float.extend",
            FLOAT_EXTEND_SELECTOR,
            "Exactly extends a narrow floating encoding to f32.",
            (
                "Extends f8e4m3, f8e5m2, f16, or bf16 to its exact binary32 value. "
                "Signed zero and infinity are preserved where representable; a source "
                "NaN produces any quiet f32 NaN and never a signaling NaN."
            ),
            (
                "destination_v8 receives the exact binary32 result in its low 32 bits "
                "and zero in its high 32 bits."
            ),
            (
                "narrow_bits = low_source_bits(selector_u8, values[source_v8]);\n"
                "result_bits = structurally_extend_narrow_to_f32(\n"
                "    selector_u8, narrow_bits);\n"
                "values[destination_v8] = result_bits;\n"
                "pc = pc + 4;"
            ),
        ),
        _ConversionDefinition(
            0xA2,
            "conversion.float.truncate",
            FLOAT_TRUNCATE_SELECTOR,
            "Rounds f32 or f64 directly to one narrow floating encoding.",
            (
                "Rounds the selected f32 or f64 source directly to f8e4m3, f8e5m2, "
                "f16, or bf16 without an intermediate-width conversion. Finite "
                "f8e4m3 overflow and source infinity saturate to signed 448; overflow "
                "and infinity for the other formats produce signed infinity. A source "
                "NaN produces any quiet destination NaN."
            ),
            (
                "destination_v8 receives the selected narrow encoding in its low 8 or "
                "16 bits and zero in every higher cell bit."
            ),
            (
                "source_bits = selected_float_source_bits(\n"
                "    selector_u8, values[source_v8]);\n"
                "result_bits = structurally_round_float_to_narrow(\n"
                "    selector_u8, source_bits);\n"
                "values[destination_v8] = result_bits;\n"
                "pc = pc + 4;"
            ),
        ),
        _ConversionDefinition(
            0xA3,
            "conversion.float.width",
            FLOAT_WIDTH_SELECTOR,
            "Converts exactly between f32 and f64 widths.",
            (
                "Exactly extends finite f32 to f64 or rounds f64 to f32. The narrowing "
                "direction produces signed infinity on finite overflow. Both "
                "directions preserve signed zero and infinity and produce a permitted "
                "arithmetic NaN for a NaN source."
            ),
            (
                "destination_v8 receives the selected result. An f32 result clears "
                "the high 32 cell bits; an f64 result occupies the complete cell."
            ),
            (
                "result_bits = convert_float_width(selector_u8, values[source_v8]);\n"
                "values[destination_v8] = canonicalize_selected_float_cell(\n"
                "    selector_u8, result_bits);\n"
                "pc = pc + 4;"
            ),
        ),
        _ConversionDefinition(
            0xA4,
            "conversion.integer.to.float",
            INTEGER_TO_FLOAT_SELECTOR,
            "Rounds a signed or unsigned integer directly to a float.",
            (
                "Interprets the selected low 32 or 64 source bits as a signed or "
                "unsigned integer and rounds the exact mathematical value directly to "
                "f32, f64, or bf16. The bf16 cases do not stage through f32. Every "
                "source value is finite in every destination format."
            ),
            (
                "destination_v8 receives the selected floating encoding. F32 and bf16 "
                "results clear every cell bit above their 32- or 16-bit width; f64 "
                "occupies the complete cell."
            ),
            (
                "integer = decode_selected_integer(selector_u8, values[source_v8]);\n"
                "result_bits = round_integer_to_selected_float(selector_u8, integer);\n"
                "values[destination_v8] = canonicalize_selected_float_cell(\n"
                "    selector_u8, result_bits);\n"
                "pc = pc + 4;"
            ),
        ),
        _ConversionDefinition(
            0xA5,
            "conversion.float.to.integer",
            FLOAT_TO_INTEGER_SELECTOR,
            "Truncates an in-range finite float to a selected integer width.",
            (
                "Truncates a finite f32 or f64 mathematical value toward zero to a "
                "selected signed or unsigned 1-, 8-, 16-, 32-, or 64-bit integer. The "
                "successful strict source interval is (-2^(N-1)-1, 2^(N-1)) for "
                "signed N-bit results and (-1, 2^N) for unsigned results. Checks "
                "operate on source bits before any host floating-to-integer conversion."
            ),
            (
                "destination_v8 receives the truncated two's-complement or unsigned "
                "result and clears every cell bit above the destination width."
            ),
            (
                "source_bits = selected_float_source_bits(\n"
                "    selector_u8, values[source_v8]);\n"
                "if (is_nan_bits(source_bits)) {\n"
                "  fail(invalid_argument, no_message);\n"
                "}\n"
                "if (!is_finite_bits(source_bits) ||\n"
                "    !source_is_in_conversion_interval(selector_u8, source_bits)) {\n"
                "  fail(out_of_range, no_message);\n"
                "}\n"
                "result_bits = truncate_float_to_integer(selector_u8, source_bits);\n"
                "values[destination_v8] = canonicalize_selected_integer_cell(\n"
                "    selector_u8, result_bits);\n"
                "pc = pc + 4;"
            ),
            preconditions=(
                "The source is finite, not NaN, and within the strict interval for "
                "the selected integer destination.",
            ),
            failures=(
                FailureCase(
                    "invalid_argument",
                    "The source is a quiet or signaling NaN.",
                    "destination_v8 and all other VM state remain unchanged.",
                ),
                FailureCase(
                    "out_of_range",
                    (
                        "The source is infinite or outside the selected successful "
                        "interval."
                    ),
                    "destination_v8 and all other VM state remain unchanged.",
                ),
            ),
        ),
    )
)
