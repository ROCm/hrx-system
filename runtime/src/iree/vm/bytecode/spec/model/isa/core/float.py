# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 floating-point instructions."""

from __future__ import annotations

from model.isa import (
    Instruction,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
)
from model.isa.declarations import (
    core_instruction,
    instruction_field,
    value_register,
    zero_padding,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import SELECTOR, ZERO
from model.schema import U8, U16, EntityReference, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.float",
    since=CORE_0,
    summary="Selected-width IEEE floating arithmetic and math operations.",
    dependencies=("core.contract.machine",),
    document_order=5,
    normative_text=(
        "F32 and f64 are IEEE 754 binary32 and binary64 bit patterns. F32 "
        "reads low 32 cell bits and clears every high result bit; f64 consumes "
        "and produces the complete cell. Each arithmetic record executes and "
        "rounds independently at its selected width using nearest-even, without "
        "f32 promotion through f64 or retained excess precision. Subnormals use "
        "gradual underflow and contraction occurs only for the explicit fma "
        "selector. Before every start or resume drive segment, the runtime saves "
        "the calling thread's floating environment and installs masked exceptions, "
        "nearest-even rounding, and preserved input/output subnormals. Every "
        "segment exit restores the saved environment, including completion, "
        "suspension, and failure; nested VM calls execute within the installed "
        "profile and provider callbacks never enter the VM inline. A host unable to "
        "establish this profile fails before bytecode executes. Floating flags "
        "and errno are unobservable, and domains, poles, division by zero, "
        "overflow, underflow, and invalid arithmetic produce floating results, "
        "never VM status failures. An arithmetic NaN has an all-one exponent, "
        "nonzero significand, and quiet bit set. Its sign is unspecified. If "
        "every NaN input has canonical payload 0x7FC00000 for f32 or "
        "0x7FF8000000000000 for f64, including invalid operations with no NaN "
        "input, the result payload is canonical; otherwise any arithmetic NaN "
        "is permitted. Neg, abs, copysign, comparisons, classification, and "
        "ordered clamp use raw payload logic where stated and never accidentally "
        "quiet or trap on signaling payloads. Host-math selectors return the "
        "named width-matched standard function result; conforming finite last "
        "bits may vary by platform and runtime release, while operation identity, "
        "selected width, staged rounding, exceptional edges, and NaN result class "
        "remain architectural. Derived selectors round every named assignment "
        "at selected width and forbid contraction, reassociation, or promotion. "
        "rsqrt is sqrt then reciprocal. Logistic uses a stable sign split; silu "
        "multiplies x by logistic(x); softplus is max(x,+0)+log1p(exp(-abs(x))); "
        "gelu.logistic computes x*logistic(scale*x). gelu.erf and gelu.tanh use "
        "their named staged formulas with f32/f64 inverse_sqrt2 payloads "
        "0x3F3504F3/0x3FE6A09E667F3BCD, cubic coefficients "
        "0x3D372713/0x3FA6E4E26D4801F7, and sqrt_2_over_pi payloads "
        "0x3F4C422A/0x3FE9884533D43651. Sinturns and costurns preserve exact "
        "periodicity by quarter-turn residual/quadrant reduction, return cardinal "
        "values structurally, and never multiply the original input by rounded "
        "tau. All operations read inputs before destination publication, are "
        "infallible after verification, access no refs, and never suspend."
    ),
)


def _value(name: str, offset: int, role: InstructionFieldRole):
    return value_register(
        name,
        offset,
        role,
        f"{role.value.replace('_', ' ').capitalize()} value-register ordinal.",
    )


def _selector(name: str, offset: int, table_name: str):
    table = SELECTOR_TABLES_BY_NAME[table_name]
    return instruction_field(
        name,
        offset,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        f"Closed {table_name} operation selector.",
        (RuleUse(SELECTOR.entity_id, (EntityReference(table.entity_id),)),),
    )


def _u16_zero(offset: int):
    return instruction_field(
        "zero_padding_u16",
        offset,
        U16.entity_id,
        InstructionFieldRole.PADDING,
        "Canonical zero padding.",
        (RuleUse(ZERO.entity_id),),
    )


def _semantics(
    *,
    description: str,
    verification: tuple[str, ...],
    success: tuple[str, ...],
    assembly: tuple[str, ...],
    pseudocode: str,
    byte_length: int,
):
    return InstructionSemantics(
        description=description,
        verification=verification,
        preconditions=(),
        success=(
            *success,
            f"The program counter advances by {byte_length} bytes.",
        ),
        failures=(),
        ownership=(
            "All inputs are read before destination publication; no ref "
            "ownership state is accessed or changed.",
        ),
        assembly=assembly,
        pseudocode=pseudocode,
    )


def _binary(
    *,
    opcode: int,
    mnemonic: str,
    description: str,
    expression: str,
    raw_bits: bool = False,
) -> Instruction:
    width = 32 if mnemonic.endswith("32") else 64
    write = "write_float_bits" if raw_bits else "write_float"
    read = "read_float_bits" if raw_bits else "read_float"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=description,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("lhs_v8", 2, InstructionFieldRole.OPERAND),
            _value("rhs_v8", 3, InstructionFieldRole.OPERAND),
        ),
        state_effects=(),
        semantics=_semantics(
            description=description,
            verification=(
                "dst_v8, lhs_v8, and rhs_v8 must be valid value-register ordinals.",
            ),
            success=(
                (
                    "dst_v8 receives the f32 result in its low 32 bits and "
                    "its high 32 cell bits are cleared."
                )
                if width == 32
                else "dst_v8 receives the complete 64-bit f64 result.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<lhs>, %v<rhs>",),
            pseudocode=(
                f"lhs = {read}(lhs_v8, {width});\n"
                f"rhs = {read}(rhs_v8, {width});\n"
                f"result = {expression};\n"
                f"{write}(dst_v8, result, {width});\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


_BINARY_INSTRUCTIONS = (
    _binary(
        opcode=0x80,
        mnemonic="float.add.f32",
        description="Adds two f32 values with one selected-width rounding.",
        expression="lhs + rhs",
    ),
    _binary(
        opcode=0x81,
        mnemonic="float.add.f64",
        description="Adds two f64 values with one selected-width rounding.",
        expression="lhs + rhs",
    ),
    _binary(
        opcode=0x82,
        mnemonic="float.sub.f32",
        description="Subtracts two f32 values with one selected-width rounding.",
        expression="lhs - rhs",
    ),
    _binary(
        opcode=0x83,
        mnemonic="float.sub.f64",
        description="Subtracts two f64 values with one selected-width rounding.",
        expression="lhs - rhs",
    ),
    _binary(
        opcode=0x84,
        mnemonic="float.mul.f32",
        description="Multiplies two f32 values with one selected-width rounding.",
        expression="lhs * rhs",
    ),
    _binary(
        opcode=0x85,
        mnemonic="float.mul.f64",
        description="Multiplies two f64 values with one selected-width rounding.",
        expression="lhs * rhs",
    ),
    _binary(
        opcode=0x86,
        mnemonic="float.div.f32",
        description="Divides two f32 values with IEEE non-stop semantics.",
        expression="lhs / rhs",
    ),
    _binary(
        opcode=0x87,
        mnemonic="float.div.f64",
        description="Divides two f64 values with IEEE non-stop semantics.",
        expression="lhs / rhs",
    ),
    _binary(
        opcode=0x88,
        mnemonic="float.rem.f32",
        description=(
            "Computes width-matched f32 fmod: x-trunc(x/y)*y, with the "
            "dividend's sign including zero."
        ),
        expression="host_math(fmod, lhs, rhs, 32)",
    ),
    _binary(
        opcode=0x89,
        mnemonic="float.rem.f64",
        description=(
            "Computes width-matched f64 fmod: x-trunc(x/y)*y, with the "
            "dividend's sign including zero."
        ),
        expression="host_math(fmod, lhs, rhs, 64)",
    ),
    _binary(
        opcode=0x96,
        mnemonic="float.copysign.f32",
        description=(
            "Copies rhs's raw f32 sign bit onto every non-sign bit from lhs, "
            "preserving signaling and noncanonical payloads."
        ),
        expression="(lhs & 0x7FFFFFFF) | (rhs & 0x80000000)",
        raw_bits=True,
    ),
    _binary(
        opcode=0x97,
        mnemonic="float.copysign.f64",
        description=(
            "Copies rhs's raw f64 sign bit onto every non-sign bit from lhs, "
            "preserving signaling and noncanonical payloads."
        ),
        expression=("(lhs & 0x7FFFFFFFFFFFFFFF) | (rhs & 0x8000000000000000)"),
        raw_bits=True,
    ),
)


def _sign_unary(
    *, opcode: int, mnemonic: str, description: str, expression: str
) -> Instruction:
    width = 32 if mnemonic.endswith("32") else 64
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=description,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("src_v8", 2, InstructionFieldRole.OPERAND),
            zero_padding("zero_padding_u8", 3, 1),
        ),
        state_effects=(),
        semantics=_semantics(
            description=description,
            verification=(
                "dst_v8 and src_v8 must be valid value-register ordinals and "
                "zero_padding_u8 must equal zero.",
            ),
            success=(
                "dst_v8 receives the exact transformed raw payload without "
                "quieting NaNs or raising a floating exception.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<src>",),
            pseudocode=(
                f"bits = read_float_bits(src_v8, {width});\n"
                f"write_float_bits(dst_v8, {expression}, {width});\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


_SIGN_UNARY_INSTRUCTIONS = (
    _sign_unary(
        opcode=0x8A,
        mnemonic="float.neg.f32",
        description="Toggles the raw f32 sign bit.",
        expression="bits ^ 0x80000000",
    ),
    _sign_unary(
        opcode=0x8B,
        mnemonic="float.neg.f64",
        description="Toggles the raw f64 sign bit.",
        expression="bits ^ 0x8000000000000000",
    ),
    _sign_unary(
        opcode=0x8C,
        mnemonic="float.abs.f32",
        description="Clears the raw f32 sign bit.",
        expression="bits & 0x7FFFFFFF",
    ),
    _sign_unary(
        opcode=0x8D,
        mnemonic="float.abs.f64",
        description="Clears the raw f64 sign bit.",
        expression="bits & 0x7FFFFFFFFFFFFFFF",
    ),
)


def _binary_selector(
    *,
    opcode: int,
    mnemonic: str,
    field_name: str,
    table_name: str,
    description: str,
    result: str,
    evaluator: str,
) -> Instruction:
    width = 32 if mnemonic.endswith("32") else 64
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=description,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("lhs_v8", 2, InstructionFieldRole.OPERAND),
            _value("rhs_v8", 3, InstructionFieldRole.OPERAND),
            _selector(field_name, 4, table_name),
            zero_padding("zero_padding_u8", 5, 3),
        ),
        state_effects=(),
        semantics=_semantics(
            description=description,
            verification=(
                f"Every register must be valid, {field_name} must be an assigned "
                f"{table_name} value, and every padding byte must be zero.",
            ),
            success=(result,),
            assembly=(f"%v<dst> = {mnemonic} %v<lhs>, %v<rhs> {{{field_name}}}",),
            pseudocode=(
                f"lhs = read_float_bits(lhs_v8, {width});\n"
                f"rhs = read_float_bits(rhs_v8, {width});\n"
                f"result = {evaluator}({field_name}, lhs, rhs, {width});\n"
                + (
                    "values[dst_v8] = canonical_bool(result);\n"
                    if table_name == "float.compare"
                    else f"write_float_bits(dst_v8, result, {width});\n"
                )
                + "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


FLOAT_MINMAX_F32 = _binary_selector(
    opcode=0x8E,
    mnemonic="float.minmax.f32",
    field_name="selector_u8",
    table_name="float.minmax",
    description=(
        "Evaluates f32 IEEE minimum/maximum or number-selecting minnum/maxnum "
        "with explicit NaN and signed-zero rules."
    ),
    result=(
        "minimum/maximum propagate an arithmetic NaN from either NaN input; "
        "minnum/maxnum return the sole numeric operand bit-for-bit and propagate "
        "only when both inputs are NaN. Opposite zeros select -0 for minima and "
        "+0 for maxima."
    ),
    evaluator="evaluate_float_minmax",
)
FLOAT_MINMAX_F64 = _binary_selector(
    opcode=0x8F,
    mnemonic="float.minmax.f64",
    field_name="selector_u8",
    table_name="float.minmax",
    description=(
        "Evaluates f64 IEEE minimum/maximum or number-selecting minnum/maxnum "
        "with explicit NaN and signed-zero rules."
    ),
    result=(
        "minimum/maximum propagate an arithmetic NaN from either NaN input; "
        "minnum/maxnum return the sole numeric operand bit-for-bit and propagate "
        "only when both inputs are NaN. Opposite zeros select -0 for minima and "
        "+0 for maxima."
    ),
    evaluator="evaluate_float_minmax",
)
FLOAT_COMPARE_F32 = _binary_selector(
    opcode=0x90,
    mnemonic="float.compare.f32",
    field_name="predicate_u8",
    table_name="float.compare",
    description="Evaluates one ordered or unordered raw-payload f32 predicate.",
    result=(
        "dst_v8 receives canonical whole-cell zero or one. Either NaN makes "
        "ordered predicates false and unordered predicates true as selected; "
        "signed zeros compare equal."
    ),
    evaluator="evaluate_float_predicate",
)
FLOAT_COMPARE_F64 = _binary_selector(
    opcode=0x91,
    mnemonic="float.compare.f64",
    field_name="predicate_u8",
    table_name="float.compare",
    description="Evaluates one ordered or unordered raw-payload f64 predicate.",
    result=(
        "dst_v8 receives canonical whole-cell zero or one. Either NaN makes "
        "ordered predicates false and unordered predicates true as selected; "
        "signed zeros compare equal."
    ),
    evaluator="evaluate_float_predicate",
)


def _classify(*, opcode: int, width: int) -> Instruction:
    mnemonic = f"float.classify.f{width}"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Classifies one raw f{width} payload as NaN, infinity, or finite.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("src_v8", 2, InstructionFieldRole.OPERAND),
            _selector("selector_u8", 3, "float.classify"),
        ),
        state_effects=(),
        semantics=_semantics(
            description=(
                "Tests raw exponent/significand bits for isnan, isinf, or "
                "isfinite without performing floating arithmetic."
            ),
            verification=(
                "dst_v8 and src_v8 must be valid and selector_u8 must name an "
                "assigned float.classify value.",
            ),
            success=(
                "dst_v8 receives canonical complete-cell zero or one without "
                "trapping on a signaling NaN payload.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<src> {{selector}}",),
            pseudocode=(
                f"bits = read_float_bits(src_v8, {width});\n"
                "values[dst_v8] = canonical_bool(\n"
                f"    classify_float(selector_u8, bits, {width}));\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


FLOAT_CLASSIFY_F32 = _classify(opcode=0x92, width=32)
FLOAT_CLASSIFY_F64 = _classify(opcode=0x93, width=64)


def _clamp(*, opcode: int, width: int) -> Instruction:
    mnemonic = f"float.clamp.f{width}"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Clamps one f{width} payload with explicit NaN policy.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("value_v8", 2, InstructionFieldRole.OPERAND),
            _value("lower_v8", 3, InstructionFieldRole.OPERAND),
            _value("upper_v8", 4, InstructionFieldRole.OPERAND),
            _selector("mode_u8", 5, "float.clamp"),
            _u16_zero(6),
        ),
        state_effects=(),
        semantics=_semantics(
            description=(
                "Ordered mode applies two ordered comparisons/selects and thus "
                "preserves a NaN value bit-for-bit and ignores a NaN bound. "
                "Number mode composes maxnum then minnum; IEEE mode composes "
                "maximum then minimum. Every mode is total when lower > upper."
            ),
            verification=(
                "All four registers must be valid, mode_u8 assigned in "
                "float.clamp, and zero_padding_u16 zero.",
            ),
            success=(
                "dst_v8 receives the exact selected-width composition, including "
                "its mode-specific NaN and signed-zero behavior.",
            ),
            assembly=(
                f"%v<dst> = {mnemonic} %v<value>, %v<lower>, %v<upper> {{mode}}",
            ),
            pseudocode=(
                f"value = read_float_bits(value_v8, {width});\n"
                f"lower = read_float_bits(lower_v8, {width});\n"
                f"upper = read_float_bits(upper_v8, {width});\n"
                "result = evaluate_float_clamp(\n"
                f"    mode_u8, value, lower, upper, {width});\n"
                f"write_float_bits(dst_v8, result, {width});\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


FLOAT_CLAMP_F32 = _clamp(opcode=0x94, width=32)
FLOAT_CLAMP_F64 = _clamp(opcode=0x95, width=64)


def _math_unary(*, opcode: int, width: int) -> Instruction:
    mnemonic = f"float.math.unary.f{width}"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Evaluates one closed unary f{width} math operation.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("src_v8", 2, InstructionFieldRole.OPERAND),
            _selector("selector_u8", 3, "float.math.unary"),
        ),
        state_effects=(),
        semantics=_semantics(
            description=(
                "Evaluates the selector's exact-width, width-matched host-math, "
                "staged-derived, or reduced-host-math contract under the family "
                "floating profile. Structural sign uses raw bits."
            ),
            verification=(
                "dst_v8 and src_v8 must be valid and selector_u8 must name one "
                "of the 37 version-zero float.math.unary operations.",
            ),
            success=(
                "dst_v8 receives the selector-defined result, exact edges, stage "
                "rounding, and arithmetic-NaN class at selected width.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<src> {{selector}}",),
            pseudocode=(
                f"source_bits = read_float_bits(src_v8, {width});\n"
                "result = evaluate_float_math_unary(\n"
                f"    selector_u8, source_bits, {width});\n"
                f"write_float(dst_v8, result, {width});\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


FLOAT_MATH_UNARY_F32 = _math_unary(opcode=0x98, width=32)
FLOAT_MATH_UNARY_F64 = _math_unary(opcode=0x99, width=64)


FLOAT_MATH_BINARY_F32 = _binary_selector(
    opcode=0x9A,
    mnemonic="float.math.binary.f32",
    field_name="selector_u8",
    table_name="float.math.binary",
    description="Evaluates f32 pow, atan2, or staged scaled-logistic GELU.",
    result=(
        "dst_v8 receives the selector-defined result with width-matched host "
        "math for pow/atan2 or three selected-width stages for gelu.logistic."
    ),
    evaluator="evaluate_float_math_binary",
)
FLOAT_MATH_BINARY_F64 = _binary_selector(
    opcode=0x9B,
    mnemonic="float.math.binary.f64",
    field_name="selector_u8",
    table_name="float.math.binary",
    description="Evaluates f64 pow, atan2, or staged scaled-logistic GELU.",
    result=(
        "dst_v8 receives the selector-defined result with width-matched host "
        "math for pow/atan2 or three selected-width stages for gelu.logistic."
    ),
    evaluator="evaluate_float_math_binary",
)


def _math_ternary(*, opcode: int, width: int) -> Instruction:
    mnemonic = f"float.math.ternary.f{width}"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Computes one fused f{width} multiply-add.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("a_v8", 2, InstructionFieldRole.OPERAND),
            _value("b_v8", 3, InstructionFieldRole.OPERAND),
            _value("c_v8", 4, InstructionFieldRole.OPERAND),
            _selector("selector_u8", 5, "float.math.ternary"),
            _u16_zero(6),
        ),
        state_effects=(),
        semantics=_semantics(
            description=(
                "Computes infinitely precise a*b+c and rounds once to selected "
                "width. Fma is the sole version-zero ternary selector."
            ),
            verification=(
                "All four registers must be valid, selector_u8 must select fma, "
                "and zero_padding_u16 must equal zero.",
            ),
            success=(
                "dst_v8 receives the IEEE fused-multiply-add result and inherited "
                "arithmetic-NaN behavior after one final rounding.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<a>, %v<b>, %v<c> {{fma}}",),
            pseudocode=(
                "result = fused_multiply_add(\n"
                f"    read_float(a_v8, {width}), read_float(b_v8, {width}),\n"
                f"    read_float(c_v8, {width}), {width});\n"
                f"write_float(dst_v8, result, {width});\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


FLOAT_MATH_TERNARY_F32 = _math_ternary(opcode=0x9C, width=32)
FLOAT_MATH_TERNARY_F64 = _math_ternary(opcode=0x9D, width=64)

_INSTRUCTIONS_BY_OPCODE = {
    instruction.opcode: instruction
    for instruction in (
        *_BINARY_INSTRUCTIONS,
        *_SIGN_UNARY_INSTRUCTIONS,
        FLOAT_MINMAX_F32,
        FLOAT_MINMAX_F64,
        FLOAT_COMPARE_F32,
        FLOAT_COMPARE_F64,
        FLOAT_CLASSIFY_F32,
        FLOAT_CLASSIFY_F64,
        FLOAT_CLAMP_F32,
        FLOAT_CLAMP_F64,
        FLOAT_MATH_UNARY_F32,
        FLOAT_MATH_UNARY_F64,
        FLOAT_MATH_BINARY_F32,
        FLOAT_MATH_BINARY_F64,
        FLOAT_MATH_TERNARY_F32,
        FLOAT_MATH_TERNARY_F64,
    )
}
INSTRUCTIONS = tuple(
    _INSTRUCTIONS_BY_OPCODE[opcode] for opcode in sorted(_INSTRUCTIONS_BY_OPCODE)
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
