# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core and HAL 0.0 closed instruction-selector tables."""

from __future__ import annotations

from model.schema import (
    U8,
    NumericTable,
    NumericTableKind,
    NumericValue,
    UnknownNumericValuePolicy,
)
from model.specification import CORE_0, HAL_0, Version


def _selector_table(
    entity_id: str,
    since: Version,
    selector_name: str,
    summary: str,
    values: tuple[tuple[int, str, int, str], ...],
) -> tuple[NumericTable, tuple[NumericValue, ...]]:
    table = NumericTable(
        entity_id=entity_id,
        since=since,
        summary=summary,
        encoding_id=U8.entity_id,
        table_kind=NumericTableKind.SELECTOR,
        unknown_value_policy=UnknownNumericValuePolicy.REJECT,
    )
    table_values = tuple(
        NumericValue(
            entity_id=f"{entity_id}.{name}",
            since=Version(since.domain, since.major, since_minor),
            summary=value_summary,
            table_id=entity_id,
            name=name,
            value=value,
        )
        for value, name, since_minor, value_summary in values
    )
    return table, table_values


_INTEGER_WIDTHS = (1, 8, 16, 32, 64)


def parse_integer_conversion_name(name: str) -> tuple[str, int, int]:
    """Returns the operation and widths encoded by an integer selector name."""

    try:
        source_name, destination_name = name.split(".to.")
        source_kind = source_name[0]
        source_bit_count = int(source_name[1:])
        if not destination_name.startswith("i"):
            raise ValueError
        destination_bit_count = int(destination_name[1:])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid integer conversion selector {name!r}") from exc
    if source_bit_count not in _INTEGER_WIDTHS:
        raise ValueError(f"invalid integer conversion selector {name!r}")
    if destination_bit_count not in _INTEGER_WIDTHS:
        raise ValueError(f"invalid integer conversion selector {name!r}")
    if source_kind == "s" and source_bit_count < destination_bit_count:
        operation = "sign_extend"
    elif source_kind == "u" and source_bit_count < destination_bit_count:
        operation = "zero_extend"
    elif source_kind == "i" and source_bit_count > destination_bit_count:
        operation = "truncate"
    else:
        raise ValueError(f"invalid integer conversion selector {name!r}")
    return operation, source_bit_count, destination_bit_count


def _integer_convert_values() -> tuple[tuple[int, str, int, str], ...]:
    values: list[tuple[int, str, int, str]] = []
    for source_index, source_width in enumerate(_INTEGER_WIDTHS):
        for destination_width in _INTEGER_WIDTHS[source_index + 1 :]:
            values.append(
                (
                    len(values),
                    f"s{source_width}.to.i{destination_width}",
                    0,
                    f"Sign-extends the low {source_width} bits to "
                    f"i{destination_width}.",
                )
            )
            values.append(
                (
                    len(values),
                    f"u{source_width}.to.i{destination_width}",
                    0,
                    f"Zero-extends the low {source_width} bits to "
                    f"i{destination_width}.",
                )
            )
    for source_index, source_width in enumerate(_INTEGER_WIDTHS):
        for destination_width in _INTEGER_WIDTHS[:source_index]:
            values.append(
                (
                    len(values),
                    f"i{source_width}.to.i{destination_width}",
                    0,
                    f"Preserves the low {destination_width} bits and clears "
                    "all higher bits.",
                )
            )
    return tuple(values)


_SELECTOR_DEFINITIONS = (
    _selector_table(
        "core.selector.memory.format",
        CORE_0,
        "memory.format",
        (
            "Selects the integer lane width and lane count transferred between "
            "one value-register run and consecutive little-endian bytes."
        ),
        (
            (0, "i8.x1", 0, "Transfers one 8-bit lane spanning one byte."),
            (1, "i8.x2", 0, "Transfers two 8-bit lanes spanning two bytes."),
            (2, "i8.x4", 0, "Transfers four 8-bit lanes spanning four bytes."),
            (3, "i8.x8", 0, "Transfers eight 8-bit lanes spanning eight bytes."),
            (4, "i16.x1", 0, "Transfers one 16-bit lane spanning two bytes."),
            (5, "i16.x2", 0, "Transfers two 16-bit lanes spanning four bytes."),
            (6, "i16.x4", 0, "Transfers four 16-bit lanes spanning eight bytes."),
            (7, "i16.x8", 0, "Transfers eight 16-bit lanes spanning 16 bytes."),
            (8, "i32.x1", 0, "Transfers one 32-bit lane spanning four bytes."),
            (9, "i32.x2", 0, "Transfers two 32-bit lanes spanning eight bytes."),
            (10, "i32.x4", 0, "Transfers four 32-bit lanes spanning 16 bytes."),
            (11, "i32.x8", 0, "Transfers eight 32-bit lanes spanning 32 bytes."),
            (12, "i64.x1", 0, "Transfers one 64-bit lane spanning eight bytes."),
            (13, "i64.x2", 0, "Transfers two 64-bit lanes spanning 16 bytes."),
            (14, "i64.x4", 0, "Transfers four 64-bit lanes spanning 32 bytes."),
            (15, "i64.x8", 0, "Transfers eight 64-bit lanes spanning 64 bytes."),
        ),
    ),
    _selector_table(
        "core.selector.control.call.target",
        CORE_0,
        "control.call.target",
        "Selects the ordinal table used by a direct control.call record.",
        (
            (
                0,
                "local",
                0,
                "The ordinal names a function defined by the current module.",
            ),
            (
                1,
                "required_import",
                0,
                "The ordinal names an import that must resolve while linking.",
            ),
            (
                2,
                "optional_import",
                0,
                "The ordinal names a weak import; calling it unresolved fails not_found.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.integer.compare",
        CORE_0,
        "integer.compare",
        (
            "Selects an equality, signed-order, or unsigned-order predicate over "
            "the instruction-selected low integer width."
        ),
        (
            (0, "eq", 0, "True when both low-width bit patterns are equal."),
            (1, "ne", 0, "True when both low-width bit patterns differ."),
            (2, "slt", 0, "True when lhs is signed less than rhs."),
            (3, "sle", 0, "True when lhs is signed less than or equal to rhs."),
            (4, "sgt", 0, "True when lhs is signed greater than rhs."),
            (5, "sge", 0, "True when lhs is signed greater than or equal to rhs."),
            (6, "ult", 0, "True when lhs is unsigned less than rhs."),
            (7, "ule", 0, "True when lhs is unsigned less than or equal to rhs."),
            (8, "ugt", 0, "True when lhs is unsigned greater than rhs."),
            (9, "uge", 0, "True when lhs is unsigned greater than or equal to rhs."),
        ),
    ),
    _selector_table(
        "core.selector.float.minmax",
        CORE_0,
        "float.minmax",
        (
            "Selects IEEE minimum/maximum or number-selecting minnum/maxnum. "
            "Numeric ordering governs ordinary values; minima choose -0 and "
            "maxima choose +0 from opposite signed zeros."
        ),
        (
            (
                0,
                "minimum",
                0,
                "Returns the numeric minimum and an arithmetic NaN if either operand is NaN.",
            ),
            (
                1,
                "maximum",
                0,
                "Returns the numeric maximum and an arithmetic NaN if either operand is NaN.",
            ),
            (
                2,
                "minnum",
                0,
                "Returns the sole numeric operand bit-for-bit, or an arithmetic NaN when both are NaN.",
            ),
            (
                3,
                "maxnum",
                0,
                "Returns the sole numeric operand bit-for-bit, or an arithmetic NaN when both are NaN.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.float.math.unary",
        CORE_0,
        "float.math.unary",
        (
            "Selects one unary operation under the floating family profile. "
            "Width-matched host math uses the named f32/f64 function; staged "
            "operations round every named assignment at the selected width."
        ),
        (
            (0, "exp", 0, "Returns the width-matched exponential exp(x)."),
            (1, "exp2", 0, "Returns the width-matched base-two exponential 2^x."),
            (2, "expm1", 0, "Returns the width-matched exp(x)-1 operation."),
            (3, "log", 0, "Returns the width-matched natural logarithm."),
            (4, "log2", 0, "Returns the width-matched base-two logarithm."),
            (5, "log10", 0, "Returns the width-matched base-ten logarithm."),
            (6, "log1p", 0, "Returns the width-matched log(1+x) operation."),
            (7, "sqrt", 0, "Returns correctly rounded selected-width square root."),
            (
                8,
                "rsqrt",
                0,
                "Rounds sqrt(x), then rounds selected-width 1/root without contraction.",
            ),
            (9, "cbrt", 0, "Returns the width-matched cube root."),
            (10, "sin", 0, "Returns width-matched sine with x in radians."),
            (11, "cos", 0, "Returns width-matched cosine with x in radians."),
            (
                12,
                "sinturns",
                0,
                "Returns sin(2*pi*x) using exact quarter-turn reduction and structural cardinal results.",
            ),
            (
                13,
                "costurns",
                0,
                "Returns cos(2*pi*x) using exact quarter-turn reduction and structural cardinal results.",
            ),
            (14, "tan", 0, "Returns width-matched tangent with x in radians."),
            (15, "asin", 0, "Returns the width-matched inverse sine."),
            (16, "acos", 0, "Returns the width-matched inverse cosine."),
            (17, "atan", 0, "Returns the width-matched inverse tangent."),
            (18, "sinh", 0, "Returns the width-matched hyperbolic sine."),
            (19, "cosh", 0, "Returns the width-matched hyperbolic cosine."),
            (20, "tanh", 0, "Returns the width-matched hyperbolic tangent."),
            (21, "asinh", 0, "Returns the width-matched inverse hyperbolic sine."),
            (22, "acosh", 0, "Returns the width-matched inverse hyperbolic cosine."),
            (23, "atanh", 0, "Returns the width-matched inverse hyperbolic tangent."),
            (24, "erf", 0, "Returns the width-matched error function."),
            (25, "erfc", 0, "Returns the width-matched complementary error function."),
            (
                26,
                "logistic",
                0,
                "Uses the stable sign split: 1/(1+exp(-x)) for x>=0 and exp(x)/(1+exp(x)) otherwise.",
            ),
            (
                27,
                "silu",
                0,
                "Rounds logistic(x), then rounds x*logistic(x).",
            ),
            (
                28,
                "softplus",
                0,
                "Computes ordered_max(x,+0)+log1p(exp(-abs(x))) in selected-width stages.",
            ),
            (29, "ceil", 0, "Returns the integral value toward positive infinity."),
            (30, "floor", 0, "Returns the integral value toward negative infinity."),
            (
                31,
                "round",
                0,
                "Returns the nearest integral value with halfway cases away from zero.",
            ),
            (
                32,
                "roundeven",
                0,
                "Returns the nearest integral value with halfway cases to even.",
            ),
            (33, "trunc", 0, "Returns the integral value toward zero."),
            (
                34,
                "sign",
                0,
                "Returns +0 for NaNs and either zero, -1 for negative nonzero values, and +1 otherwise.",
            ),
            (
                35,
                "gelu.erf",
                0,
                "Computes (0.5*x)*(1+erf(x*inverse_sqrt2)) in selected-width stages.",
            ),
            (
                36,
                "gelu.tanh",
                0,
                "Computes (0.5*x)*(1+tanh(sqrt_2_over_pi*(x+cubic_coefficient*x^3))) in selected-width stages.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.float.math.binary",
        CORE_0,
        "float.math.binary",
        (
            "Selects one binary operation under the floating family profile; "
            "staged operations round every named assignment at selected width."
        ),
        (
            (0, "pow", 0, "Returns the width-matched pow(lhs, rhs) operation."),
            (1, "atan2", 0, "Returns the width-matched atan2(lhs, rhs) operation."),
            (
                2,
                "gelu.logistic",
                0,
                "Treats lhs as x and rhs as scale, then computes x*logistic(scale*x) in three stages.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.float.math.ternary",
        CORE_0,
        "float.math.ternary",
        "Selects one ternary operation under the floating family profile.",
        (
            (
                0,
                "fma",
                0,
                "Computes infinitely precise a*b+c and rounds once to selected width.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.float.compare",
        CORE_0,
        "float.compare",
        (
            "Selects an ordered or unordered IEEE predicate. Ordered predicates "
            "require both operands to be non-NaN; unordered predicates are true "
            "when either operand is NaN. Signed zeros compare equal."
        ),
        (
            (0, "oeq", 0, "True when neither operand is NaN and lhs equals rhs."),
            (
                1,
                "ogt",
                0,
                "True when neither operand is NaN and lhs is greater than rhs.",
            ),
            (2, "oge", 0, "True when neither operand is NaN and lhs is at least rhs."),
            (3, "olt", 0, "True when neither operand is NaN and lhs is less than rhs."),
            (4, "ole", 0, "True when neither operand is NaN and lhs is at most rhs."),
            (5, "one", 0, "True when neither operand is NaN and lhs differs from rhs."),
            (6, "ord", 0, "True when neither operand is NaN."),
            (7, "ueq", 0, "True when either operand is NaN or lhs equals rhs."),
            (
                8,
                "ugt",
                0,
                "True when either operand is NaN or lhs is greater than rhs.",
            ),
            (9, "uge", 0, "True when either operand is NaN or lhs is at least rhs."),
            (10, "ult", 0, "True when either operand is NaN or lhs is less than rhs."),
            (11, "ule", 0, "True when either operand is NaN or lhs is at most rhs."),
            (12, "une", 0, "True when either operand is NaN or lhs differs from rhs."),
            (13, "uno", 0, "True when either operand is NaN."),
        ),
    ),
    _selector_table(
        "core.selector.float.classify",
        CORE_0,
        "float.classify",
        "Selects a raw exponent/significand classification without floating arithmetic.",
        (
            (0, "isnan", 0, "True for quiet or signaling NaN payloads."),
            (1, "isinf", 0, "True for either signed infinity and false for NaNs."),
            (2, "isfinite", 0, "True for zero, subnormal, and normal finite payloads."),
        ),
    ),
    _selector_table(
        "core.selector.float.clamp",
        CORE_0,
        "float.clamp",
        (
            "Selects one exact selected-width clamp composition. All modes are "
            "defined when lower exceeds upper."
        ),
        (
            (
                0,
                "ordered",
                0,
                "Starts with value, selects lower when result<lower, then upper when result>upper; NaN comparisons are false.",
            ),
            (
                1,
                "number",
                0,
                "Computes minnum(maxnum(value, lower), upper).",
            ),
            (
                2,
                "ieee",
                0,
                "Computes minimum(maximum(value, lower), upper).",
            ),
        ),
    ),
    _selector_table(
        "core.selector.integer.convert",
        CORE_0,
        "integer.convert",
        (
            "Selects an exact low-bit integer truncation or extension. Results "
            "clear every cell bit above their declared destination width."
        ),
        _integer_convert_values(),
    ),
    _selector_table(
        "core.selector.float.extend",
        CORE_0,
        "float.extend",
        (
            "Selects one narrow source format to extend structurally to its exact "
            "f32 value; NaNs produce a quiet f32 arithmetic NaN."
        ),
        (
            (0, "f8e4m3.to.f32", 0, "Extends low E4M3FN bits exactly to f32."),
            (1, "f8e5m2.to.f32", 0, "Extends low E5M2 bits exactly to f32."),
            (2, "f16.to.f32", 0, "Extends low IEEE binary16 bits exactly to f32."),
            (3, "bf16.to.f32", 0, "Extends low bfloat16 bits exactly to f32."),
        ),
    ),
    _selector_table(
        "core.selector.float.truncate",
        CORE_0,
        "float.truncate",
        (
            "Selects source and narrow destination formats for one direct "
            "nearest-even conversion with gradual subnormals. E4M3FN overflow "
            "and infinity saturate to signed 448; other overflow produces infinity."
        ),
        (
            (0, "f32.to.f8e4m3", 0, "Rounds f32 directly to E4M3FN."),
            (1, "f32.to.f8e5m2", 0, "Rounds f32 directly to E5M2."),
            (2, "f32.to.f16", 0, "Rounds f32 directly to IEEE binary16."),
            (3, "f32.to.bf16", 0, "Rounds f32 directly to bfloat16."),
            (
                4,
                "f64.to.f8e4m3",
                0,
                "Rounds f64 directly to E4M3FN without f32 staging.",
            ),
            (5, "f64.to.f8e5m2", 0, "Rounds f64 directly to E5M2 without f32 staging."),
            (
                6,
                "f64.to.f16",
                0,
                "Rounds f64 directly to IEEE binary16 without f32 staging.",
            ),
            (
                7,
                "f64.to.bf16",
                0,
                "Rounds f64 directly to bfloat16 without f32 staging.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.float.width",
        CORE_0,
        "float.width",
        "Selects one nearest-even conversion between IEEE binary32 and binary64.",
        (
            (0, "f32.to.f64", 0, "Extends every finite f32 exactly to f64."),
            (
                1,
                "f64.to.f32",
                0,
                "Rounds f64 to f32 with gradual subnormals and infinity on overflow.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.integer.to.float",
        CORE_0,
        "integer.to.float",
        (
            "Selects source signedness/width and destination float format. The "
            "exact integer is rounded directly nearest-even without intermediate staging."
        ),
        (
            (0, "s32.to.f32", 0, "Rounds signed low i32 directly to f32."),
            (1, "u32.to.f32", 0, "Rounds unsigned low i32 directly to f32."),
            (2, "s32.to.f64", 0, "Rounds signed low i32 directly to f64."),
            (3, "u32.to.f64", 0, "Rounds unsigned low i32 directly to f64."),
            (4, "s64.to.f32", 0, "Rounds signed i64 directly to f32."),
            (5, "u64.to.f32", 0, "Rounds unsigned i64 directly to f32."),
            (6, "s64.to.f64", 0, "Rounds signed i64 directly to f64."),
            (7, "u64.to.f64", 0, "Rounds unsigned i64 directly to f64."),
            (8, "s32.to.bf16", 0, "Rounds signed low i32 directly to bfloat16."),
            (9, "u32.to.bf16", 0, "Rounds unsigned low i32 directly to bfloat16."),
            (10, "s64.to.bf16", 0, "Rounds signed i64 directly to bfloat16."),
            (11, "u64.to.bf16", 0, "Rounds unsigned i64 directly to bfloat16."),
        ),
    ),
    _selector_table(
        "core.selector.float.to.integer",
        CORE_0,
        "float.to.integer",
        (
            "Selects a finite f32/f64 source and integer destination. Successful "
            "values truncate toward zero; NaN fails invalid_argument and values "
            "outside the destination's strict source interval fail out_of_range."
        ),
        (
            (0, "f32.to.s32", 0, "Truncates f32 in (-2^31-1, 2^31) to signed i32."),
            (1, "f32.to.u32", 0, "Truncates f32 in (-1, 2^32) to unsigned i32."),
            (2, "f32.to.s64", 0, "Truncates f32 in (-2^63-1, 2^63) to signed i64."),
            (3, "f32.to.u64", 0, "Truncates f32 in (-1, 2^64) to unsigned i64."),
            (4, "f64.to.s32", 0, "Truncates f64 in (-2^31-1, 2^31) to signed i32."),
            (5, "f64.to.u32", 0, "Truncates f64 in (-1, 2^32) to unsigned i32."),
            (6, "f64.to.s64", 0, "Truncates f64 in (-2^63-1, 2^63) to signed i64."),
            (7, "f64.to.u64", 0, "Truncates f64 in (-1, 2^64) to unsigned i64."),
        ),
    ),
    _selector_table(
        "core.selector.control.status",
        CORE_0,
        "control.status",
        (
            "Selects one non-OK architectural status code for control.fail; zero "
            "is deliberately absent because OK is not a failure."
        ),
        (
            (1, "cancelled", 0, "Produces the canonical cancelled status code."),
            (2, "unknown", 0, "Produces the canonical unknown status code."),
            (
                3,
                "invalid_argument",
                0,
                "Produces the canonical invalid_argument status code.",
            ),
            (
                4,
                "deadline_exceeded",
                0,
                "Produces the canonical deadline_exceeded status code.",
            ),
            (5, "not_found", 0, "Produces the canonical not_found status code."),
            (
                6,
                "already_exists",
                0,
                "Produces the canonical already_exists status code.",
            ),
            (
                7,
                "permission_denied",
                0,
                "Produces the canonical permission_denied status code.",
            ),
            (
                8,
                "resource_exhausted",
                0,
                "Produces the canonical resource_exhausted status code.",
            ),
            (
                9,
                "failed_precondition",
                0,
                "Produces the canonical failed_precondition status code.",
            ),
            (10, "aborted", 0, "Produces the canonical aborted status code."),
            (11, "out_of_range", 0, "Produces the canonical out_of_range status code."),
            (
                12,
                "unimplemented",
                0,
                "Produces the canonical unimplemented status code.",
            ),
            (13, "internal", 0, "Produces the canonical internal status code."),
            (14, "unavailable", 0, "Produces the canonical unavailable status code."),
            (15, "data_loss", 0, "Produces the canonical data_loss status code."),
            (
                16,
                "unauthenticated",
                0,
                "Produces the canonical unauthenticated status code.",
            ),
            (18, "incompatible", 0, "Produces the canonical incompatible status code."),
        ),
    ),
    _selector_table(
        "core.selector.buffer.atomic.kind",
        CORE_0,
        "buffer.atomic.kind",
        (
            "Selects the exact replacement function for an atomic carrier. Integer "
            "arithmetic wraps at carrier width; floating operations inherit the "
            "selected-width floating profile and float.minmax rules."
        ),
        (
            (0, "exchange.integer", 0, "Replaces the carrier bits with operand bits."),
            (
                1,
                "exchange.float",
                0,
                "Replaces the f32/f64 carrier bits with operand bits.",
            ),
            (2, "add.integer", 0, "Replaces with unsigned modular old+operand."),
            (3, "add.float", 0, "Replaces with selected-width IEEE old+operand."),
            (4, "subtract.integer", 0, "Replaces with unsigned modular old-operand."),
            (5, "and.integer", 0, "Replaces with the bitwise old AND operand."),
            (6, "or.integer", 0, "Replaces with the bitwise old OR operand."),
            (7, "xor.integer", 0, "Replaces with the bitwise old XOR operand."),
            (
                8,
                "minimum.signed",
                0,
                "Replaces with the two's-complement signed minimum.",
            ),
            (
                9,
                "maximum.signed",
                0,
                "Replaces with the two's-complement signed maximum.",
            ),
            (10, "minimum.unsigned", 0, "Replaces with the unsigned minimum."),
            (11, "maximum.unsigned", 0, "Replaces with the unsigned maximum."),
            (12, "minimum.float", 0, "Replaces with float.minmax minimum."),
            (13, "maximum.float", 0, "Replaces with float.minmax maximum."),
            (14, "minnum.float", 0, "Replaces with float.minmax minnum."),
            (15, "maxnum.float", 0, "Replaces with float.minmax maxnum."),
        ),
    ),
    _selector_table(
        "core.selector.buffer.atomic.carrier",
        CORE_0,
        "buffer.atomic.carrier",
        "Selects the naturally aligned raw-storage carrier width.",
        (
            (0, "i32", 0, "Uses a four-byte carrier and low 32 value-cell bits."),
            (1, "i64", 0, "Uses an eight-byte carrier and the complete value cell."),
        ),
    ),
    _selector_table(
        "core.selector.buffer.atomic.ordering",
        CORE_0,
        "buffer.atomic.ordering",
        (
            "Selects the minimum C11-style synchronization ordering. A target may "
            "strengthen but never weaken it."
        ),
        (
            (0, "relaxed", 0, "Guarantees atomicity without inter-operation ordering."),
            (1, "acquire", 0, "Applies acquire ordering to the operation's read."),
            (2, "release", 0, "Applies release ordering to the operation's write."),
            (
                3,
                "acq_rel",
                0,
                "Applies acquire ordering to the read and release ordering to the write.",
            ),
            (
                4,
                "seq_cst",
                0,
                "Participates in one sequentially consistent total order.",
            ),
        ),
    ),
    _selector_table(
        "core.selector.buffer.atomic.scope",
        CORE_0,
        "buffer.atomic.scope",
        (
            "Selects the minimum synchronization domain. CPU interpretation may "
            "strengthen a narrower domain to process/system scope."
        ),
        (
            (0, "thread", 0, "Requires ordering only within the current thread."),
            (
                1,
                "subgroup",
                0,
                "Requires ordering among invocations in one execution subgroup.",
            ),
            (
                2,
                "workgroup",
                0,
                "Requires ordering among invocations in one workgroup.",
            ),
            (3, "device", 0, "Requires ordering among agents on one logical device."),
            (
                4,
                "system",
                0,
                "Requires ordering across every participating system agent.",
            ),
        ),
    ),
    _selector_table(
        "hal.selector.cmd.dispatch.barrier_before",
        HAL_0,
        "hal.cmd.dispatch.barrier_before",
        "Selects whether command dispatch records a fixed phase barrier first.",
        (
            (0, "none", 0, "Records only the dispatch."),
            (
                1,
                "all",
                0,
                "Records a full non-host device execution/memory barrier covering RAW, WAR, and WAW reuse, then dispatches.",
            ),
        ),
    ),
    _selector_table(
        "hal.selector.collective.kind",
        HAL_0,
        "hal.collective.kind",
        (
            "Selects channel communication shape and therefore the contextual "
            "send/receive packets and element counts required on the current rank."
        ),
        (
            (
                0,
                "all_gather",
                0,
                "Every rank sends element_count elements and receives channel_count*element_count elements.",
            ),
            (
                1,
                "all_reduce",
                0,
                "Every rank sends and receives element_count elements using the selected reduction.",
            ),
            (
                2,
                "all_to_all",
                0,
                "Every rank sends and receives element_count elements, which must be divisible by channel_count.",
            ),
            (
                3,
                "broadcast",
                0,
                "Root rank param sends element_count elements; every other rank receives them, and param must be in range.",
            ),
            (
                4,
                "reduce",
                0,
                "Every rank sends element_count elements and root rank param alone receives the selected reduction.",
            ),
            (
                5,
                "reduce_scatter",
                0,
                "Every rank sends channel_count*element_count elements and receives element_count reduced elements.",
            ),
            (
                6,
                "send",
                0,
                "Sends element_count elements to peer rank param and uses no receive side.",
            ),
            (
                7,
                "recv",
                0,
                "Receives element_count elements from peer rank param and uses no send side.",
            ),
            (
                8,
                "send_recv",
                0,
                "Uses signed low/high 16-bit target/source ranks; target -1 skips send and source -1 fills receive with zeros.",
            ),
        ),
    ),
    _selector_table(
        "hal.selector.collective.reduction",
        HAL_0,
        "hal.collective.reduction",
        (
            "Selects the HAL collective reduction. Reducing kinds require a "
            "non-none value; nonreducing kinds require none."
        ),
        (
            (0, "none", 0, "Performs no element reduction."),
            (1, "sum", 0, "Reduces corresponding selected-type elements by sum."),
            (
                2,
                "product",
                0,
                "Reduces corresponding selected-type elements by product.",
            ),
            (
                3,
                "minimum",
                0,
                "Reduces corresponding selected-type elements by minimum.",
            ),
            (
                4,
                "maximum",
                0,
                "Reduces corresponding selected-type elements by maximum.",
            ),
            (
                5,
                "average",
                0,
                "Reduces corresponding selected-type elements by average.",
            ),
        ),
    ),
    _selector_table(
        "hal.selector.collective.element",
        HAL_0,
        "hal.collective.element",
        (
            "Selects the element interpretation and byte width used for checked "
            "collective range sizing and provider reduction."
        ),
        (
            (0, "si8", 0, "Uses signed 8-bit integer elements of width one byte."),
            (1, "ui8", 0, "Uses unsigned 8-bit integer elements of width one byte."),
            (2, "si16", 0, "Uses signed 16-bit integer elements of width two bytes."),
            (3, "ui16", 0, "Uses unsigned 16-bit integer elements of width two bytes."),
            (4, "si32", 0, "Uses signed 32-bit integer elements of width four bytes."),
            (
                5,
                "ui32",
                0,
                "Uses unsigned 32-bit integer elements of width four bytes.",
            ),
            (6, "si64", 0, "Uses signed 64-bit integer elements of width eight bytes."),
            (
                7,
                "ui64",
                0,
                "Uses unsigned 64-bit integer elements of width eight bytes.",
            ),
            (8, "f16", 0, "Uses IEEE binary16 elements of width two bytes."),
            (9, "f32", 0, "Uses IEEE binary32 elements of width four bytes."),
            (10, "f64", 0, "Uses IEEE binary64 elements of width eight bytes."),
            (11, "bf16", 0, "Uses bfloat16 elements of width two bytes."),
        ),
    ),
    _selector_table(
        "hal.selector.semaphore.await.mode",
        HAL_0,
        "hal.semaphore.await.mode",
        "Selects the satisfaction rule and successful result for a timepoint set.",
        (
            (
                0,
                "all",
                0,
                "Completes when every timepoint is satisfied and returns UINT64_MAX; an empty set succeeds.",
            ),
            (
                1,
                "any",
                0,
                "Requires a nonempty set, completes when one timepoint is satisfied, and returns a selected zero-based index.",
            ),
        ),
    ),
    _selector_table(
        "hal.selector.semaphore.await.timeout_kind",
        HAL_0,
        "hal.semaphore.await.timeout_kind",
        "Selects the signed-i64 nanosecond interpretation of the timeout register.",
        (
            (
                0,
                "relative",
                0,
                "Zero polls, positive finite values become one saturating monotonic deadline, INT64_MAX waits indefinitely, and negatives fail.",
            ),
            (
                1,
                "absolute",
                0,
                "INT64_MIN polls, INT64_MAX waits indefinitely, and every other value is an exact monotonic deadline.",
            ),
        ),
    ),
)

SELECTOR_TABLES = tuple(definition[0] for definition in _SELECTOR_DEFINITIONS)
SELECTOR_VALUES = tuple(
    value for definition in _SELECTOR_DEFINITIONS for value in definition[1]
)
SELECTOR_TABLES_BY_NAME = {
    (
        ("hal." if table.since.domain == "hal" else "")
        + table.entity_id.partition(".selector.")[2]
    ): table
    for table in SELECTOR_TABLES
}
_MEMORY_FORMAT_TABLE_ID = SELECTOR_TABLES_BY_NAME["memory.format"].entity_id


def memory_format_lane_count(selector_value: int) -> int:
    """Decodes the lane count carried by a memory.format selector value."""

    return 1 << (selector_value & 0x3)


MEMORY_FORMAT_MAXIMUM_LANE_COUNT = max(
    memory_format_lane_count(value.value)
    for value in SELECTOR_VALUES
    if value.table_id == _MEMORY_FORMAT_TABLE_ID
)
ENTITIES = (*SELECTOR_TABLES, *SELECTOR_VALUES)
