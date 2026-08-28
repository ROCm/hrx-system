# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 scalar conversion instructions."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
)
from model.isa.declarations import (
    core_instruction,
    instruction_field,
    value_register,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import SELECTOR
from model.schema import U8, EntityReference, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.conversion",
    since=CORE_0,
    summary="Exact integer- and floating-point scalar conversions.",
    dependencies=("core.contract.machine",),
    document_order=4,
    normative_text=(
        "All conversion records read the complete source cell before publishing "
        "the destination, so source and destination may alias. Narrow inputs "
        "ignore bits above their declared width and narrow results clear all "
        "unused high cell bits. Floating conversions are structural operations "
        "independent of ambient host rounding, FTZ, and DAZ modes. They use "
        "round-to-nearest with ties to an even least-significant significand "
        "bit, gradual subnormals, signed zero, and quiet arithmetic NaN results "
        "whose sign and payload are unspecified. f8e4m3 is PyTorch E4M3FN: it "
        "has finite values through signed 448 and no infinity. f8e5m2, f16, "
        "bf16, f32, and f64 use their standard infinity and NaN encodings. "
        "No conversion accesses refs or suspends."
    ),
)


def _selector_field(selector_name: str):
    table = SELECTOR_TABLES_BY_NAME[selector_name]
    return instruction_field(
        "selector_u8",
        3,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        f"Closed {selector_name} operation selector.",
        (RuleUse(SELECTOR.entity_id, (EntityReference(table.entity_id),)),),
    )


def _conversion(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    selector_name: str,
    description: str,
    success: tuple[str, ...],
    pseudocode: str,
    preconditions: tuple[str, ...] = (),
    failures: tuple[FailureCase, ...] = (),
):
    return core_instruction(
        entity_id=entity_id,
        since=CORE_0,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            value_register(
                "dst_v8",
                1,
                InstructionFieldRole.RESULT,
                "Destination value-register ordinal.",
            ),
            value_register(
                "src_v8",
                2,
                InstructionFieldRole.OPERAND,
                "Source value-register ordinal.",
            ),
            _selector_field(selector_name),
        ),
        state_effects=(),
        semantics=InstructionSemantics(
            description=description,
            verification=(
                "dst_v8 and src_v8 must be valid value-register ordinals.",
                f"selector_u8 must name an assigned {selector_name} value.",
            ),
            preconditions=preconditions,
            success=(*success, "The program counter advances by four bytes."),
            failures=failures,
            ownership=(
                "The instruction reads src_v8 before publishing dst_v8 and "
                "has no ref ownership effect.",
            ),
            assembly=(
                f"%v<dst> = {mnemonic} %v<src> {{selector}}",
                f"%v5 = {mnemonic} %v2 {{{selector_name}.selector}}",
            ),
            pseudocode=pseudocode,
        ),
    )


CONVERSION_INTEGER = _conversion(
    entity_id="core.instruction.conversion.integer",
    summary="Truncates or extends one integer bit pattern.",
    opcode=0xA0,
    mnemonic="conversion.integer",
    selector_name="integer.convert",
    description=(
        "Performs the selected exact conversion between distinct i1, i8, i16, "
        "i32, and i64 widths. Signed extensions propagate the selected source "
        "sign bit, unsigned extensions insert zeros, and truncations preserve "
        "the selected low destination-width bits."
    ),
    success=(
        "dst_v8 receives the selected truncation or extension. Results no "
        "wider than 32 bits clear the high 32 cell bits, and i8/i16 results "
        "also clear every bit above their selected width.",
    ),
    pseudocode=(
        "source_bits = values[src_v8];\n"
        "result_bits = evaluate_integer_conversion(selector_u8, source_bits);\n"
        "values[dst_v8] = result_bits;\n"
        "pc = pc + 4;"
    ),
)

CONVERSION_FLOAT_EXTEND = _conversion(
    entity_id="core.instruction.conversion.float.extend",
    summary="Exactly extends a narrow floating encoding to f32.",
    opcode=0xA1,
    mnemonic="conversion.float.extend",
    selector_name="float.extend",
    description=(
        "Extends f8e4m3, f8e5m2, f16, or bf16 to its exact binary32 value. "
        "Signed zero and infinity are preserved where representable; a source "
        "NaN produces any quiet f32 NaN and never a signaling NaN."
    ),
    success=(
        "dst_v8 receives the exact binary32 result in its low 32 bits and "
        "zero in its high 32 bits.",
    ),
    pseudocode=(
        "narrow_bits = low_source_bits(selector_u8, values[src_v8]);\n"
        "result_bits = structurally_extend_narrow_to_f32(\n"
        "    selector_u8, narrow_bits);\n"
        "values[dst_v8] = result_bits;\n"
        "pc = pc + 4;"
    ),
)

CONVERSION_FLOAT_TRUNCATE = _conversion(
    entity_id="core.instruction.conversion.float.truncate",
    summary="Rounds f32 or f64 directly to one narrow floating encoding.",
    opcode=0xA2,
    mnemonic="conversion.float.truncate",
    selector_name="float.truncate",
    description=(
        "Rounds the selected f32 or f64 source directly to f8e4m3, f8e5m2, "
        "f16, or bf16 without an intermediate-width conversion. Finite "
        "f8e4m3 overflow and source infinity saturate to signed 448; overflow "
        "and infinity for the other formats produce signed infinity. A source "
        "NaN produces any quiet destination NaN."
    ),
    success=(
        "dst_v8 receives the selected narrow encoding in its low 8 or 16 "
        "bits and zero in every higher cell bit.",
    ),
    pseudocode=(
        "source_bits = selected_float_source_bits(\n"
        "    selector_u8, values[src_v8]);\n"
        "result_bits = structurally_round_float_to_narrow(\n"
        "    selector_u8, source_bits);\n"
        "values[dst_v8] = result_bits;\n"
        "pc = pc + 4;"
    ),
)

CONVERSION_FLOAT_WIDTH = _conversion(
    entity_id="core.instruction.conversion.float.width",
    summary="Converts exactly between f32 and f64 widths.",
    opcode=0xA3,
    mnemonic="conversion.float.width",
    selector_name="float.width",
    description=(
        "Exactly extends finite f32 to f64 or rounds f64 to f32. The narrowing "
        "direction produces signed infinity on finite overflow. Both directions "
        "preserve signed zero and infinity and produce a permitted arithmetic "
        "NaN for a NaN source."
    ),
    success=(
        "dst_v8 receives the selected result. An f32 result clears the high "
        "32 cell bits; an f64 result occupies the complete cell.",
    ),
    pseudocode=(
        "result_bits = convert_float_width(selector_u8, values[src_v8]);\n"
        "values[dst_v8] = canonicalize_selected_float_cell(\n"
        "    selector_u8, result_bits);\n"
        "pc = pc + 4;"
    ),
)

CONVERSION_INTEGER_TO_FLOAT = _conversion(
    entity_id="core.instruction.conversion.integer.to.float",
    summary="Rounds a signed or unsigned integer directly to a float.",
    opcode=0xA4,
    mnemonic="conversion.integer.to.float",
    selector_name="integer.to.float",
    description=(
        "Interprets the selected low 32 or 64 source bits as a signed or "
        "unsigned integer and rounds the exact mathematical value directly "
        "to f32, f64, or bf16. The bf16 cases do not stage through f32. Every "
        "source value is finite in every destination format."
    ),
    success=(
        "dst_v8 receives the selected floating encoding. f32 and bf16 results "
        "clear every cell bit above their 32- or 16-bit width; f64 occupies "
        "the complete cell.",
    ),
    pseudocode=(
        "integer = decode_selected_integer(selector_u8, values[src_v8]);\n"
        "result_bits = round_integer_to_selected_float(selector_u8, integer);\n"
        "values[dst_v8] = canonicalize_selected_float_cell(\n"
        "    selector_u8, result_bits);\n"
        "pc = pc + 4;"
    ),
)

CONVERSION_FLOAT_TO_INTEGER = _conversion(
    entity_id="core.instruction.conversion.float.to.integer",
    summary="Truncates an in-range finite float to a selected integer width.",
    opcode=0xA5,
    mnemonic="conversion.float.to.integer",
    selector_name="float.to.integer",
    description=(
        "Truncates a finite f32 or f64 mathematical value toward zero to s32, "
        "u32, s64, or u64. Successful strict source intervals are "
        "(-2^31-1, 2^31), (-1, 2^32), (-2^63-1, 2^63), and (-1, 2^64), "
        "respectively. The checks operate on source bits before any host "
        "floating-to-integer conversion."
    ),
    preconditions=(
        "The source must be finite, not NaN, and within the strict interval "
        "for the selected integer destination.",
    ),
    success=(
        "dst_v8 receives the truncated two's-complement or unsigned result. "
        "A 32-bit result clears the high 32 cell bits; a 64-bit result "
        "occupies the complete cell.",
    ),
    failures=(
        FailureCase(
            "invalid_argument",
            "The source is a quiet or signaling NaN.",
            "dst_v8 and all other VM state remain unchanged.",
        ),
        FailureCase(
            "out_of_range",
            "The source is infinite or outside the selected successful interval.",
            "dst_v8 and all other VM state remain unchanged.",
        ),
    ),
    pseudocode=(
        "source_bits = selected_float_source_bits(\n"
        "    selector_u8, values[src_v8]);\n"
        "if (is_nan_bits(source_bits)) {\n"
        "  fail(invalid_argument, no_message);\n"
        "}\n"
        "if (!is_finite_bits(source_bits) ||\n"
        "    !source_is_in_conversion_interval(selector_u8, source_bits)) {\n"
        "  fail(out_of_range, no_message);\n"
        "}\n"
        "result_bits = truncate_float_to_integer(selector_u8, source_bits);\n"
        "values[dst_v8] = canonicalize_selected_integer_cell(\n"
        "    selector_u8, result_bits);\n"
        "pc = pc + 4;"
    ),
)

INSTRUCTIONS = (
    CONVERSION_INTEGER,
    CONVERSION_FLOAT_EXTEND,
    CONVERSION_FLOAT_TRUNCATE,
    CONVERSION_FLOAT_WIDTH,
    CONVERSION_INTEGER_TO_FLOAT,
    CONVERSION_FLOAT_TO_INTEGER,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
