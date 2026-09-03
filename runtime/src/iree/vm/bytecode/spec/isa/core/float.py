# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selected-width IEEE floating-point instructions."""

from __future__ import annotations

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec.isa import (
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
)
from iree.vm.bytecode.spec.isa.core.rules import FieldRule
from iree.vm.bytecode.spec.schema import (
    U8,
    U16,
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


FLOAT_MINMAX_SELECTOR = _selector(
    "float.minmax",
    (
        "Selects IEEE minimum/maximum or number-selecting minnum/maxnum. "
        "Numeric ordering governs ordinary values; minima choose -0 and maxima "
        "choose +0 from opposite signed zeros."
    ),
    (
        (
            "minimum",
            0,
            "Returns the numeric minimum and an arithmetic NaN if either operand "
            "is NaN.",
        ),
        (
            "maximum",
            1,
            "Returns the numeric maximum and an arithmetic NaN if either operand "
            "is NaN.",
        ),
        (
            "minnum",
            2,
            "Returns the sole numeric operand bit-for-bit, or an arithmetic NaN "
            "when both are NaN.",
        ),
        (
            "maxnum",
            3,
            "Returns the sole numeric operand bit-for-bit, or an arithmetic NaN "
            "when both are NaN.",
        ),
    ),
)

FLOAT_MATH_UNARY_SELECTOR = _selector(
    "float.math.unary",
    (
        "Selects one unary operation under the floating family profile. "
        "Width-matched host math uses the named f32/f64 function; staged "
        "operations round every named assignment at the selected width."
    ),
    (
        (
            "exp",
            0,
            "Returns width-matched exp(x): -infinity becomes +0, +infinity "
            "becomes +infinity, and either zero becomes 1.",
        ),
        (
            "exp2",
            1,
            "Returns width-matched 2^x: -infinity becomes +0, +infinity becomes "
            "+infinity, and either zero becomes 1.",
        ),
        (
            "expm1",
            2,
            "Returns width-matched exp(x)-1: -infinity becomes -1, +infinity "
            "becomes +infinity, and signed zero is preserved.",
        ),
        (
            "log",
            3,
            "Returns the width-matched natural logarithm: either zero becomes "
            "-infinity, negative inputs produce an arithmetic NaN, and +infinity "
            "is preserved.",
        ),
        (
            "log2",
            4,
            "Returns the width-matched base-two logarithm: either zero becomes "
            "-infinity, negative inputs produce an arithmetic NaN, and +infinity "
            "is preserved.",
        ),
        (
            "log10",
            5,
            "Returns the width-matched base-ten logarithm: either zero becomes "
            "-infinity, negative inputs produce an arithmetic NaN, and +infinity "
            "is preserved.",
        ),
        (
            "log1p",
            6,
            "Returns width-matched log(1+x): -1 becomes -infinity, inputs below "
            "-1 produce an arithmetic NaN, and signed zero is preserved.",
        ),
        (
            "sqrt",
            7,
            "Returns correctly rounded selected-width square root, preserving "
            "signed zero and +infinity; negative nonzero inputs produce an "
            "arithmetic NaN.",
        ),
        (
            "rsqrt",
            8,
            "Rounds root=sqrt(x), then result=1/root at selected width without "
            "contraction. +0 becomes +infinity, -0 becomes -infinity, +infinity "
            "becomes +0, and negative nonzero inputs produce an arithmetic NaN.",
        ),
        (
            "cbrt",
            9,
            "Returns the width-matched cube root and preserves signed zero and "
            "signed infinity.",
        ),
        (
            "sin",
            10,
            "Returns width-matched radian sine, preserving signed zero; either "
            "infinity produces an arithmetic NaN.",
        ),
        (
            "cos",
            11,
            "Returns width-matched radian cosine: either zero becomes 1 and either "
            "infinity produces an arithmetic NaN.",
        ),
        (
            "sinturns",
            12,
            "Returns sin(2*pi*x) using an exact remainder and quotient quadrant for "
            "division by 0.25, structural cardinal values, and selected-width tau "
            "applied only to a non-cardinal residual. Signed zero is preserved; a "
            "positive half-integer returns +0, a negative half-integer returns -0, "
            "and an odd quarter-turn returns exact signed one. Either infinity or "
            "NaN produces an arithmetic NaN.",
        ),
        (
            "costurns",
            13,
            "Returns cos(2*pi*x) using an exact remainder and quotient quadrant for "
            "division by 0.25, structural cardinal values, and selected-width tau "
            "applied only to a non-cardinal residual. Either zero becomes 1, a "
            "half-integer returns exact signed one, and an odd quarter-turn returns "
            "+0. Either infinity or NaN produces an arithmetic NaN.",
        ),
        (
            "tan",
            14,
            "Returns width-matched radian tangent, preserving signed zero; either "
            "infinity produces an arithmetic NaN.",
        ),
        (
            "asin",
            15,
            "Returns width-matched inverse sine, preserving signed zero; magnitudes "
            "above one produce an arithmetic NaN.",
        ),
        (
            "acos",
            16,
            "Returns width-matched inverse cosine: one becomes +0 and magnitudes "
            "above one produce an arithmetic NaN.",
        ),
        (
            "atan",
            17,
            "Returns width-matched inverse tangent, preserving signed zero; signed "
            "infinity approaches signed pi/2.",
        ),
        (
            "sinh",
            18,
            "Returns width-matched hyperbolic sine and preserves signed zero and "
            "signed infinity.",
        ),
        (
            "cosh",
            19,
            "Returns width-matched hyperbolic cosine: either zero becomes 1 and "
            "either infinity becomes +infinity.",
        ),
        (
            "tanh",
            20,
            "Returns width-matched hyperbolic tangent, preserving signed zero and "
            "mapping signed infinity to signed one.",
        ),
        (
            "asinh",
            21,
            "Returns width-matched inverse hyperbolic sine and preserves signed zero "
            "and signed infinity.",
        ),
        (
            "acosh",
            22,
            "Returns width-matched inverse hyperbolic cosine: one becomes +0, values "
            "below one produce an arithmetic NaN, and +infinity is preserved.",
        ),
        (
            "atanh",
            23,
            "Returns width-matched inverse hyperbolic tangent, preserving signed "
            "zero, mapping signed one to signed infinity, and producing an arithmetic "
            "NaN for magnitudes above one.",
        ),
        (
            "erf",
            24,
            "Returns the width-matched error function, preserving signed zero and "
            "mapping signed infinity to signed one.",
        ),
        (
            "erfc",
            25,
            "Returns the width-matched complementary error function: -infinity "
            "becomes 2 and +infinity becomes +0.",
        ),
        (
            "logistic",
            26,
            "Rounds each stage of the stable sign split: for x>=0, e=exp(-x), "
            "denominator=1+e, result=1/denominator; otherwise e=exp(x), "
            "denominator=1+e, result=e/denominator. -infinity becomes +0, either "
            "zero becomes 0.5, +infinity becomes 1, and NaN produces an arithmetic "
            "NaN.",
        ),
        (
            "silu",
            27,
            "Rounds sigmoid=logistic(x), then result=x*sigmoid. -infinity and NaN "
            "produce an arithmetic NaN, signed zero is preserved, and +infinity is "
            "preserved.",
        ),
        (
            "softplus",
            28,
            "Returns an arithmetic NaN for NaN; otherwise rounds magnitude=abs(x), "
            "tail=exp(-magnitude), residual=log1p(tail), "
            "positive=ordered_max(x,+0), and result=positive+residual. -infinity "
            "becomes +0, either zero becomes log(2), and +infinity is preserved.",
        ),
        (
            "ceil",
            29,
            "Returns the integral value toward positive infinity while preserving "
            "signed zero and signed infinity.",
        ),
        (
            "floor",
            30,
            "Returns the integral value toward negative infinity while preserving "
            "signed zero and signed infinity.",
        ),
        (
            "round",
            31,
            "Returns the nearest integral value with halfway cases away from zero, "
            "preserving signed zero and signed infinity.",
        ),
        (
            "roundeven",
            32,
            "Returns the nearest integral value with halfway cases to even, preserving "
            "signed zero and signed infinity.",
        ),
        (
            "trunc",
            33,
            "Returns the integral value toward zero while preserving signed zero and "
            "signed infinity.",
        ),
        (
            "sign",
            34,
            "Classifies raw bits without arithmetic: NaNs and either zero become +0; "
            "negative nonzero values, including -infinity, become -1; positive "
            "nonzero values, including +infinity, become +1.",
        ),
        (
            "gelu.erf",
            35,
            "Rounds scaled=x*inverse_sqrt2, erf_value=host_erf(scaled), "
            "one_plus_erf=1+erf_value, half_x=0.5*x, and "
            "result=half_x*one_plus_erf. -infinity becomes an arithmetic NaN, "
            "+infinity is preserved, and signed zero is preserved.",
        ),
        (
            "gelu.tanh",
            36,
            "Rounds x_squared=x*x, x_cubed=x_squared*x, "
            "cubic_term=cubic_coefficient*x_cubed, inner_sum=x+cubic_term, "
            "scaled=sqrt_2_over_pi*inner_sum, tanh_value=host_tanh(scaled), "
            "one_plus_tanh=1+tanh_value, half_x=0.5*x, and "
            "result=half_x*one_plus_tanh. -infinity becomes an arithmetic NaN, "
            "+infinity is preserved, and signed zero is preserved.",
        ),
    ),
)

FLOAT_MATH_BINARY_SELECTOR = _selector(
    "float.math.binary",
    (
        "Selects one binary operation under the floating family profile; staged "
        "operations round every named assignment at selected width."
    ),
    (
        (
            "pow",
            0,
            "Returns width-matched pow(left, right) with C/IEC 60559 edges, "
            "including pow(NaN,+/-0)=1; negative noninteger domains produce an "
            "arithmetic NaN.",
        ),
        (
            "atan2",
            1,
            "Returns width-matched atan2(left, right) with C/IEC 60559 quadrants "
            "and signed-zero behavior.",
        ),
        (
            "gelu.logistic",
            2,
            "Treats left as x and right as scale, then rounds scaled=scale*x, "
            "sigmoid=logistic(scaled), and result=x*sigmoid. The stages apply for "
            "every zero, infinity, and NaN combination.",
        ),
    ),
)

FLOAT_MATH_TERNARY_SELECTOR = _selector(
    "float.math.ternary",
    "Selects one ternary operation under the floating family profile.",
    (
        (
            "fma",
            0,
            "Computes infinitely precise a*b+c and rounds once to selected width.",
        ),
    ),
)

FLOAT_COMPARE_SELECTOR = _selector(
    "float.compare",
    (
        "Selects an ordered or unordered IEEE predicate. Ordered predicates "
        "require both operands to be non-NaN; unordered predicates are true when "
        "either operand is NaN. Signed zeros compare equal."
    ),
    (
        ("oeq", 0, "True when neither operand is NaN and left equals right."),
        (
            "ogt",
            1,
            "True when neither operand is NaN and left is greater than right.",
        ),
        (
            "oge",
            2,
            "True when neither operand is NaN and left is at least right.",
        ),
        (
            "olt",
            3,
            "True when neither operand is NaN and left is less than right.",
        ),
        (
            "ole",
            4,
            "True when neither operand is NaN and left is at most right.",
        ),
        (
            "one",
            5,
            "True when neither operand is NaN and left differs from right.",
        ),
        ("ord", 6, "True when neither operand is NaN."),
        ("ueq", 7, "True when either operand is NaN or left equals right."),
        (
            "ugt",
            8,
            "True when either operand is NaN or left is greater than right.",
        ),
        (
            "uge",
            9,
            "True when either operand is NaN or left is at least right.",
        ),
        (
            "ult",
            10,
            "True when either operand is NaN or left is less than right.",
        ),
        (
            "ule",
            11,
            "True when either operand is NaN or left is at most right.",
        ),
        (
            "une",
            12,
            "True when either operand is NaN or left differs from right.",
        ),
        ("uno", 13, "True when either operand is NaN."),
    ),
)

FLOAT_CLASSIFY_SELECTOR = _selector(
    "float.classify",
    "Selects a raw exponent/significand classification without floating arithmetic.",
    (
        ("isnan", 0, "True for quiet or signaling NaN payloads."),
        ("isinf", 1, "True for either signed infinity and false for NaNs."),
        (
            "isfinite",
            2,
            "True for zero, subnormal, and normal finite payloads.",
        ),
    ),
)

FLOAT_CLAMP_SELECTOR = _selector(
    "float.clamp",
    (
        "Selects one exact selected-width clamp composition. All modes are "
        "defined when lower exceeds upper."
    ),
    (
        (
            "ordered",
            0,
            "Starts with value, selects lower when result<lower, then upper when "
            "result>upper; NaN comparisons are false.",
        ),
        (
            "number",
            1,
            "Computes minnum(maxnum(value, lower), upper).",
        ),
        (
            "ieee",
            2,
            "Computes minimum(maximum(value, lower), upper).",
        ),
    ),
)

FLOAT_SELECTORS = (
    FLOAT_MINMAX_SELECTOR,
    FLOAT_MATH_UNARY_SELECTOR,
    FLOAT_MATH_BINARY_SELECTOR,
    FLOAT_MATH_TERNARY_SELECTOR,
    FLOAT_COMPARE_SELECTOR,
    FLOAT_CLASSIFY_SELECTOR,
    FLOAT_CLAMP_SELECTOR,
)


FLOAT_FAMILY = InstructionFamily(
    name="float",
    since=CORE_0,
    summary="Selected-width IEEE floating arithmetic and math operations.",
    contract=(
        "f32 and f64 are IEEE 754 binary32 and binary64 bit patterns. f32 reads "
        "low 32 cell bits and clears every high result bit; f64 consumes and "
        "produces the complete cell. Each arithmetic record executes and rounds "
        "independently at its selected width using nearest-even, without f32 "
        "promotion through f64 or retained excess precision. Subnormals use gradual "
        "underflow and contraction occurs only for the explicit fma selector.\n\nBefore "
        "every start or resume drive segment, the runtime saves the calling thread's "
        "floating environment and installs masked exceptions, nearest-even rounding, "
        "and preserved input/output subnormals. Every segment exit restores the saved "
        "environment, including completion, suspension, and failure. Nested VM calls "
        "inherit the installed profile. Synchronous provider phases invoked by the "
        "drive execute under and must preserve it; asynchronous wake and completion "
        "callbacks execute no VM numeric work and inherit no VM floating contract. A "
        "host unable to establish this profile fails before bytecode executes.\n\n"
        "Floating flags and errno are unobservable, and domains, poles, "
        "division by zero, overflow, underflow, and invalid arithmetic produce "
        "floating results, never VM status failures. An arithmetic NaN has an all-one "
        "exponent, nonzero significand, and quiet bit set. Its sign is unspecified. An "
        "invalid operation with no NaN input, or an operation whose every NaN input "
        "has canonical payload 0x7FC00000 for f32 or 0x7FF8000000000000 for f64 "
        "when input signs are ignored, produces that canonical payload. An operation "
        "with a noncanonical NaN input may produce any arithmetic NaN payload.\n\n"
        "Neg, abs, copysign, comparisons, classification, and ordered clamp use raw "
        "payload logic where stated and never accidentally quiet or trap on signaling "
        "payloads. Host-math selectors return the named width-matched standard "
        "function result; conforming finite last bits may vary by platform and runtime "
        "release, while operation identity, selected width, staged rounding, "
        "exceptional edges, and NaN result class remain architectural. Derived "
        "selectors round every named assignment at selected width and forbid "
        "contraction, reassociation, or promotion. No instruction carries an ambient "
        "fast-math or approximation mode. Rsqrt is sqrt then reciprocal. "
        "Logistic uses a stable sign split; silu multiplies x by logistic(x); softplus "
        "is max(x,+0)+log1p(exp(-abs(x))); gelu.logistic computes "
        "x*logistic(scale*x). Gelu.erf and gelu.tanh use their named staged formulas "
        "with f32/f64 inverse_sqrt2 payloads 0x3F3504F3/0x3FE6A09E667F3BCD, cubic "
        "coefficients 0x3D372713/0x3FA6E4E26D4801F7, and sqrt_2_over_pi payloads "
        "0x3F4C422A/0x3FE9884533D43651. Sinturns and costurns preserve exact "
        "periodicity by quarter-turn residual/quadrant reduction, return cardinal "
        "values structurally, and never multiply the original input by rounded tau.\n\n"
        "All operations read inputs before destination publication, are infallible "
        "after verification, access no refs, and never suspend."
    ),
)


def _field(
    name: str,
    encoding,
    summary: str,
    role: FieldRole,
    rule,
    *,
    element_count: int = 1,
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary, element_count), role, rule)


def _value(name: str, role: FieldRole, summary: str) -> InstructionField:
    return _field(name, U8, summary, role, FieldRule.REGISTER_VALUE)


def _selected_value(
    name: str,
    role: FieldRole,
    summary: str,
    table: NumericTable,
) -> InstructionField:
    return _field(name, U8, summary, role, FieldRuleUse(FieldRule.SELECTOR, data=table))


def _padding(encoding=U8, *, element_count: int = 1) -> InstructionField:
    return _field(
        f"zero_padding_{encoding.name}",
        encoding,
        "Canonical zero padding.",
        FieldRole.PADDING,
        FieldRule.ZERO,
        element_count=element_count,
    )


def _result(width: int) -> str:
    if width == 32:
        return "destination_v8 receives the f32 result and clears its high cell half."
    return "destination_v8 receives the complete 64-bit f64 result."


class _FloatDataPath(enum.Enum):
    ARITHMETIC = "arithmetic"
    RAW_BITS = "raw_bits"


class _BinaryDefinition(NamedTuple):
    opcode: int
    mnemonic: str
    summary: str
    expression: str
    data_path: _FloatDataPath = _FloatDataPath.ARITHMETIC


def _binary(definition: _BinaryDefinition) -> Instruction:
    opcode, mnemonic, summary, expression, data_path = definition
    width = 32 if mnemonic.endswith("32") else 64
    read = "read_float_bits" if data_path == _FloatDataPath.RAW_BITS else "read_float"
    write = (
        "write_float_bits" if data_path == _FloatDataPath.RAW_BITS else "write_float"
    )
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FLOAT_FAMILY,
        summary=summary,
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("left_v8", FieldRole.OPERAND, "Left value-register ordinal."),
            _value("right_v8", FieldRole.OPERAND, "Right value-register ordinal."),
        ),
        semantics=None,
        behavior=f"Reads both operands before evaluating one {width}-bit result.",
        success=(_result(width),),
        assembly=f"%v<destination> = {mnemonic} %v<left>, %v<right>",
        pseudocode=(
            f"left = {read}(left_v8, {width});\n"
            f"right = {read}(right_v8, {width});\n"
            f"result = {expression};\n"
            f"{write}(destination_v8, result, {width});\n"
            "pc = pc + 4;"
        ),
    )


_BINARY_DEFINITIONS = (
    _BinaryDefinition(
        0x80,
        "float.add.f32",
        "Adds two f32 values with one selected-width rounding.",
        "left + right",
    ),
    _BinaryDefinition(
        0x81,
        "float.add.f64",
        "Adds two f64 values with one selected-width rounding.",
        "left + right",
    ),
    _BinaryDefinition(
        0x82,
        "float.sub.f32",
        "Subtracts two f32 values with one selected-width rounding.",
        "left - right",
    ),
    _BinaryDefinition(
        0x83,
        "float.sub.f64",
        "Subtracts two f64 values with one selected-width rounding.",
        "left - right",
    ),
    _BinaryDefinition(
        0x84,
        "float.mul.f32",
        "Multiplies two f32 values with one selected-width rounding.",
        "left * right",
    ),
    _BinaryDefinition(
        0x85,
        "float.mul.f64",
        "Multiplies two f64 values with one selected-width rounding.",
        "left * right",
    ),
    _BinaryDefinition(
        0x86,
        "float.div.f32",
        "Divides two f32 values with IEEE non-stop semantics.",
        "left / right",
    ),
    _BinaryDefinition(
        0x87,
        "float.div.f64",
        "Divides two f64 values with IEEE non-stop semantics.",
        "left / right",
    ),
    _BinaryDefinition(
        0x88,
        "float.rem.f32",
        "Computes width-matched f32 fmod. A numeric result has the dividend's sign, "
        "including zero; a zero divisor or infinite dividend produces an arithmetic "
        "NaN, while an infinite divisor returns a finite dividend unchanged.",
        "host_math(fmod, left, right, 32)",
    ),
    _BinaryDefinition(
        0x89,
        "float.rem.f64",
        "Computes width-matched f64 fmod. A numeric result has the dividend's sign, "
        "including zero; a zero divisor or infinite dividend produces an arithmetic "
        "NaN, while an infinite divisor returns a finite dividend unchanged.",
        "host_math(fmod, left, right, 64)",
    ),
    _BinaryDefinition(
        0x96,
        "float.copysign.f32",
        "Copies the raw f32 sign while preserving every non-sign payload bit.",
        "(left & 0x7FFFFFFF) | (right & 0x80000000)",
        _FloatDataPath.RAW_BITS,
    ),
    _BinaryDefinition(
        0x97,
        "float.copysign.f64",
        "Copies the raw f64 sign while preserving every non-sign payload bit.",
        "(left & 0x7FFFFFFFFFFFFFFF) | (right & 0x8000000000000000)",
        _FloatDataPath.RAW_BITS,
    ),
)


class _SignDefinition(NamedTuple):
    opcode: int
    mnemonic: str
    summary: str
    expression: str


def _sign_unary(definition: _SignDefinition) -> Instruction:
    opcode, mnemonic, summary, expression = definition
    width = 32 if mnemonic.endswith("32") else 64
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FLOAT_FAMILY,
        summary=summary,
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("source_v8", FieldRole.OPERAND, "Source value-register ordinal."),
            _padding(),
        ),
        semantics=None,
        behavior=(
            "Transforms the raw payload without floating arithmetic, quieting NaNs, "
            "or raising a floating exception."
        ),
        success=(
            "destination_v8 receives the exact transformed payload"
            + (" and clears its high cell half." if width == 32 else "."),
        ),
        assembly=f"%v<destination> = {mnemonic} %v<source>",
        pseudocode=(
            f"bits = read_float_bits(source_v8, {width});\n"
            f"write_float_bits(destination_v8, {expression}, {width});\n"
            "pc = pc + 4;"
        ),
    )


_SIGN_DEFINITIONS = (
    _SignDefinition(
        0x8A, "float.neg.f32", "Toggles the raw f32 sign bit.", "bits ^ 0x80000000"
    ),
    _SignDefinition(
        0x8B,
        "float.neg.f64",
        "Toggles the raw f64 sign bit.",
        "bits ^ 0x8000000000000000",
    ),
    _SignDefinition(
        0x8C, "float.abs.f32", "Clears the raw f32 sign bit.", "bits & 0x7FFFFFFF"
    ),
    _SignDefinition(
        0x8D,
        "float.abs.f64",
        "Clears the raw f64 sign bit.",
        "bits & 0x7FFFFFFFFFFFFFFF",
    ),
)


class _SelectedBinaryDefinition(NamedTuple):
    opcode: int
    mnemonic: str
    selector_name: str
    selector: NumericTable
    summary: str
    result: str
    evaluator: str


def _selected_binary(definition: _SelectedBinaryDefinition) -> Instruction:
    opcode, mnemonic, selector_name, selector, summary, result, evaluator = definition
    width = 32 if mnemonic.endswith("32") else 64
    reads = (
        "read_float" if selector is FLOAT_MATH_BINARY_SELECTOR else "read_float_bits"
    )
    if selector is FLOAT_COMPARE_SELECTOR:
        publish = "values[destination_v8] = canonical_bool(result);\n"
    elif selector is FLOAT_MINMAX_SELECTOR:
        publish = f"write_float_bits(destination_v8, result, {width});\n"
    else:
        publish = f"write_float(destination_v8, result, {width});\n"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FLOAT_FAMILY,
        summary=summary,
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("left_v8", FieldRole.OPERAND, "Left value-register ordinal."),
            _value("right_v8", FieldRole.OPERAND, "Right value-register ordinal."),
            _selected_value(
                selector_name,
                FieldRole.IMMEDIATE,
                f"Closed {selector.name} operation selector.",
                selector,
            ),
            _padding(element_count=3),
        ),
        semantics=None,
        behavior=(
            f"Evaluates the selected {width}-bit {selector.name} operation under "
            "the selector's exact contract."
        ),
        success=(result,),
        assembly=(
            f"%v<destination> = {mnemonic} %v<left>, %v<right> "
            + ("{predicate}" if selector is FLOAT_COMPARE_SELECTOR else "{selector}")
        ),
        pseudocode=(
            f"left = {reads}(left_v8, {width});\n"
            f"right = {reads}(right_v8, {width});\n"
            f"result = {evaluator}({selector_name}, left, right, {width});\n"
            + publish
            + "pc = pc + 8;"
        ),
    )


_SELECTED_BINARY_DEFINITIONS = (
    _SelectedBinaryDefinition(
        0x8E,
        "float.minmax.f32",
        "selector_u8",
        FLOAT_MINMAX_SELECTOR,
        "Selects f32 IEEE or number-selecting minimum/maximum.",
        (
            "Minimum/maximum propagate an arithmetic NaN from either NaN input; "
            "minnum/maxnum return the sole numeric operand bit-for-bit and propagate "
            "only when both inputs are NaN. Opposite zeros select -0 for minima and "
            "+0 for maxima. The f32 result clears the high cell half."
        ),
        "evaluate_float_minmax",
    ),
    _SelectedBinaryDefinition(
        0x8F,
        "float.minmax.f64",
        "selector_u8",
        FLOAT_MINMAX_SELECTOR,
        "Selects f64 IEEE or number-selecting minimum/maximum.",
        (
            "Minimum/maximum propagate an arithmetic NaN from either NaN input; "
            "minnum/maxnum return the sole numeric operand bit-for-bit and propagate "
            "only when both inputs are NaN. Opposite zeros select -0 for minima and "
            "+0 for maxima."
        ),
        "evaluate_float_minmax",
    ),
    _SelectedBinaryDefinition(
        0x90,
        "float.compare.f32",
        "predicate_u8",
        FLOAT_COMPARE_SELECTOR,
        "Evaluates one ordered or unordered raw-payload f32 predicate.",
        (
            "destination_v8 receives canonical complete-cell zero or one. Either NaN "
            "makes ordered predicates false and unordered predicates true as "
            "selected; signed zeros compare equal."
        ),
        "evaluate_float_predicate",
    ),
    _SelectedBinaryDefinition(
        0x91,
        "float.compare.f64",
        "predicate_u8",
        FLOAT_COMPARE_SELECTOR,
        "Evaluates one ordered or unordered raw-payload f64 predicate.",
        (
            "destination_v8 receives canonical complete-cell zero or one. Either NaN "
            "makes ordered predicates false and unordered predicates true as "
            "selected; signed zeros compare equal."
        ),
        "evaluate_float_predicate",
    ),
    _SelectedBinaryDefinition(
        0x9A,
        "float.math.binary.f32",
        "selector_u8",
        FLOAT_MATH_BINARY_SELECTOR,
        "Evaluates f32 pow, atan2, or staged scaled-logistic GELU.",
        (
            "destination_v8 receives the selector-defined result with width-matched "
            "host math for pow/atan2 or three selected-width stages for "
            "gelu.logistic, then clears the high cell half."
        ),
        "evaluate_float_math_binary",
    ),
    _SelectedBinaryDefinition(
        0x9B,
        "float.math.binary.f64",
        "selector_u8",
        FLOAT_MATH_BINARY_SELECTOR,
        "Evaluates f64 pow, atan2, or staged scaled-logistic GELU.",
        (
            "destination_v8 receives the selector-defined result with width-matched "
            "host math for pow/atan2 or three selected-width stages for "
            "gelu.logistic."
        ),
        "evaluate_float_math_binary",
    ),
)


def _classify(opcode: int, width: int) -> Instruction:
    mnemonic = f"float.classify.f{width}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FLOAT_FAMILY,
        summary=f"Classifies one raw f{width} payload without arithmetic.",
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("source_v8", FieldRole.OPERAND, "Source value-register ordinal."),
            _selected_value(
                "selector_u8",
                FieldRole.IMMEDIATE,
                "Closed float.classify operation selector.",
                FLOAT_CLASSIFY_SELECTOR,
            ),
        ),
        semantics=None,
        behavior=(
            "Tests raw exponent and significand bits for NaN, infinity, or finite "
            "without performing floating arithmetic."
        ),
        success=(
            "destination_v8 receives canonical complete-cell zero or one without "
            "trapping on a signaling NaN payload.",
        ),
        assembly=f"%v<destination> = {mnemonic} %v<source> {{selector}}",
        pseudocode=(
            f"bits = read_float_bits(source_v8, {width});\n"
            "values[destination_v8] = canonical_bool(\n"
            f"    classify_float(selector_u8, bits, {width}));\n"
            "pc = pc + 4;"
        ),
    )


def _clamp(opcode: int, width: int) -> Instruction:
    mnemonic = f"float.clamp.f{width}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FLOAT_FAMILY,
        summary=f"Clamps one f{width} payload with explicit NaN policy.",
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("value_v8", FieldRole.OPERAND, "Value to clamp."),
            _value("lower_v8", FieldRole.OPERAND, "Lower-bound payload."),
            _value("upper_v8", FieldRole.OPERAND, "Upper-bound payload."),
            _selected_value(
                "mode_u8",
                FieldRole.IMMEDIATE,
                "Closed float.clamp operation selector.",
                FLOAT_CLAMP_SELECTOR,
            ),
            _padding(U16),
        ),
        semantics=None,
        behavior=(
            "Ordered mode applies two ordered comparisons and selects, preserving a "
            "NaN value bit-for-bit and ignoring a NaN bound. Number mode composes "
            "maxnum then minnum; IEEE mode composes maximum then minimum. Every mode "
            "is total when lower exceeds upper."
        ),
        success=(
            "destination_v8 receives the exact selected-width composition, including "
            "its mode-specific NaN and signed-zero behavior"
            + (" and clears the high cell half." if width == 32 else "."),
        ),
        assembly=(
            f"%v<destination> = {mnemonic} %v<value>, %v<lower>, %v<upper> {{mode}}"
        ),
        pseudocode=(
            f"value = read_float_bits(value_v8, {width});\n"
            f"lower = read_float_bits(lower_v8, {width});\n"
            f"upper = read_float_bits(upper_v8, {width});\n"
            "result = evaluate_float_clamp(\n"
            f"    mode_u8, value, lower, upper, {width});\n"
            f"write_float_bits(destination_v8, result, {width});\n"
            "pc = pc + 8;"
        ),
    )


def _math_unary(opcode: int, width: int) -> Instruction:
    mnemonic = f"float.math.unary.f{width}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FLOAT_FAMILY,
        summary=f"Evaluates one closed unary f{width} math operation.",
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("source_v8", FieldRole.OPERAND, "Source value-register ordinal."),
            _selected_value(
                "selector_u8",
                FieldRole.IMMEDIATE,
                "Closed float.math.unary operation selector.",
                FLOAT_MATH_UNARY_SELECTOR,
            ),
        ),
        semantics=None,
        behavior=(
            "Evaluates the selector's exact-width, width-matched host-math, "
            "staged-derived, or reduced-host-math contract under the family floating "
            "profile. Structural sign uses raw bits."
        ),
        success=(
            "destination_v8 receives the selector-defined result, exact edges, stage "
            "rounding, and arithmetic-NaN class"
            + (" and clears the high cell half." if width == 32 else "."),
        ),
        assembly=f"%v<destination> = {mnemonic} %v<source> {{selector}}",
        pseudocode=(
            f"source_bits = read_float_bits(source_v8, {width});\n"
            "result = evaluate_float_math_unary(\n"
            f"    selector_u8, source_bits, {width});\n"
            f"write_float(destination_v8, result, {width});\n"
            "pc = pc + 4;"
        ),
    )


def _math_ternary(opcode: int, width: int) -> Instruction:
    mnemonic = f"float.math.ternary.f{width}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FLOAT_FAMILY,
        summary=f"Computes one fused f{width} multiply-add.",
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("a_v8", FieldRole.OPERAND, "Multiplicand A payload."),
            _value("b_v8", FieldRole.OPERAND, "Multiplicand B payload."),
            _value("c_v8", FieldRole.OPERAND, "Addend C payload."),
            _selected_value(
                "selector_u8",
                FieldRole.IMMEDIATE,
                "Closed float.math.ternary operation selector.",
                FLOAT_MATH_TERNARY_SELECTOR,
            ),
            _padding(U16),
        ),
        semantics=None,
        behavior=(
            "Computes infinitely precise a*b+c and rounds once to selected width. "
            "Fma is the sole version-zero ternary selector."
        ),
        success=(
            "destination_v8 receives the IEEE fused-multiply-add result and inherited "
            "arithmetic-NaN behavior after one final rounding"
            + (" and clears the high cell half." if width == 32 else "."),
        ),
        assembly=f"%v<destination> = {mnemonic} %v<a>, %v<b>, %v<c> {{selector}}",
        pseudocode=(
            "result = fused_multiply_add(\n"
            f"    read_float(a_v8, {width}), read_float(b_v8, {width}),\n"
            f"    read_float(c_v8, {width}), {width});\n"
            f"write_float(destination_v8, result, {width});\n"
            "pc = pc + 8;"
        ),
    )


_INSTRUCTIONS_BY_OPCODE = {
    instruction.opcode: instruction
    for instruction in (
        *(_binary(definition) for definition in _BINARY_DEFINITIONS),
        *(_sign_unary(definition) for definition in _SIGN_DEFINITIONS),
        *(_selected_binary(definition) for definition in _SELECTED_BINARY_DEFINITIONS),
        _classify(0x92, 32),
        _classify(0x93, 64),
        _clamp(0x94, 32),
        _clamp(0x95, 64),
        _math_unary(0x98, 32),
        _math_unary(0x99, 64),
        _math_ternary(0x9C, 32),
        _math_ternary(0x9D, 64),
    )
}
FLOAT_INSTRUCTIONS = tuple(
    _INSTRUCTIONS_BY_OPCODE[opcode] for opcode in sorted(_INSTRUCTIONS_BY_OPCODE)
)
