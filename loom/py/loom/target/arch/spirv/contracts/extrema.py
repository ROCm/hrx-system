# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Floating extrema using integer ordering, independent of float controls."""

from dataclasses import dataclass

from loom.dialect.scalar import arithmetic as scalar
from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.descriptor_rule import (
    IntegerRecipe,
    descriptor_feature_guards,
    logical_core_descriptor,
)
from loom.target.arch.spirv.extended_math import EXTREMA_MATH_OPERATIONS
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    Vector,
)


@dataclass(frozen=True, slots=True)
class _FloatEncoding:
    # Authored element type, including software-only formats.
    source_type: str
    # Total source encoding width, including the sign bit.
    bit_width: int
    # Largest magnitude encoding that is not NaN, including infinity if present.
    largest_non_nan: int
    # Canonical quiet NaN encoding for propagating extrema.
    quiet_nan: int


_ENCODINGS = (
    _FloatEncoding("f16", 16, 0x7C00, 0x7E00),
    _FloatEncoding("bf16", 16, 0x7F80, 0x7FC0),
    _FloatEncoding("f32", 32, 0x7F800000, 0x7FC00000),
    _FloatEncoding("f64", 64, 0x7FF0000000000000, 0x7FF8000000000000),
    _FloatEncoding("f8E4M3", 8, 0x7E, 0x7F),
    _FloatEncoding("f8E5M2", 8, 0x7C, 0x7E),
)

_OPERATIONS = (
    ("minnumf", scalar.scalar_minnumf, vector.vector_minnumf, "number"),
    ("maxnumf", scalar.scalar_maxnumf, vector.vector_maxnumf, "number"),
    ("minimumf", scalar.scalar_minimumf, vector.vector_minimumf, "ieee"),
    ("maximumf", scalar.scalar_maximumf, vector.vector_maximumf, "ieee"),
    ("clampf", scalar.scalar_clampf, vector.vector_clampf, "ordered"),
    ("clampf", scalar.scalar_clampf, vector.vector_clampf, "number"),
    ("clampf", scalar.scalar_clampf, vector.vector_clampf, "ieee"),
)


def _boolean(
    recipe: IntegerRecipe, operation: str, lhs: ValueRef, rhs: ValueRef
) -> ValueRef:
    suffix = "bool" if recipe.lanes == 1 else f"v{recipe.lanes}bool"
    return recipe.emit(f"logical_{operation}.{suffix}", {"lhs": lhs, "rhs": rhs})


def _extremum(
    recipe: IntegerRecipe,
    encoding: _FloatEncoding,
    lhs: ValueRef,
    rhs: ValueRef,
    *,
    minimum: bool,
    mode: str,
    result: ValueRef | None = None,
) -> ValueRef:
    binary = recipe.binary
    constant = recipe.constant
    zero = constant(0)
    # Signed bit ordering agrees with numeric ordering except when both
    # encodings are negative. Inverting that comparison also admits equality,
    # which selects an identical payload. It orders -0 before +0.
    lhs_order = lhs
    rhs_order = rhs
    if encoding.bit_width != recipe.integer_width:
        shift = constant(recipe.integer_width - encoding.bit_width)
        lhs_order = binary("shift_left_logical", lhs, shift)
        rhs_order = binary("shift_left_logical", rhs, shift)
    both_negative = binary(
        "s_less_than", binary("bitwise_and", lhs_order, rhs_order), zero
    )
    raw_less = binary("s_less_than", lhs_order, rhs_order)
    less = _boolean(recipe, "not_equal", raw_less, both_negative)
    numeric = recipe.select(less, lhs if minimum else rhs, rhs if minimum else lhs)
    magnitude_mask = constant((1 << (encoding.bit_width - 1)) - 1)
    lhs_magnitude = binary("bitwise_and", lhs, magnitude_mask)
    rhs_magnitude = binary("bitwise_and", rhs, magnitude_mask)
    largest_non_nan = constant(encoding.largest_non_nan)
    lhs_nan = binary("s_greater_than", lhs_magnitude, largest_non_nan)
    rhs_nan = binary("s_greater_than", rhs_magnitude, largest_non_nan)
    if mode == "number":
        # One NaN selects the numeric peer; two NaNs produce a quiet NaN even
        # when the selected input payload was signaling.
        rhs_or_nan = recipe.select(rhs_nan, constant(encoding.quiet_nan), rhs)
        return recipe.select(
            lhs_nan, rhs_or_nan, recipe.select(rhs_nan, lhs, numeric), result=result
        )
    unordered = _boolean(recipe, "or", lhs_nan, rhs_nan)
    if mode == "ieee":
        return recipe.select(
            unordered, constant(encoding.quiet_nan), numeric, result=result
        )
    # Ordered clamp retains the first operand on unordered comparisons or a
    # signed-zero tie. Every other tie already has identical encodings.
    both_zero = binary(
        "i_equal", binary("bitwise_or", lhs_magnitude, rhs_magnitude), zero
    )
    keep_lhs = _boolean(recipe, "or", unordered, both_zero)
    return recipe.select(keep_lhs, lhs, numeric, result=result)


def _native_ordered_clamp_rule(element: str, source_lanes: int) -> DescriptorRule:
    lanes = max(1, source_lanes)
    suffix = element if lanes == 1 else f"v{lanes}{element}"
    pattern = Scalar(element) if source_lanes == 0 else Vector(element, lanes=lanes)
    recipe = IntegerRecipe(int(element.removeprefix("f")), lanes)
    value = ValueRef.operand("value")
    lower = ValueRef.operand("lower")
    upper = ValueRef.operand("upper")
    below = recipe.emit(
        f"f_ord_less_than.preserve.{suffix}", {"lhs": value, "rhs": lower}
    )
    bounded = recipe.emit(
        f"select.{suffix}",
        {"condition": below, "true_value": lower, "false_value": value},
    )
    above = recipe.emit(
        f"f_ord_greater_than.preserve.{suffix}",
        {"lhs": bounded, "rhs": upper},
    )
    recipe.emit(
        f"select.{suffix}",
        {"condition": above, "true_value": upper, "false_value": bounded},
        result=ValueRef.result("result"),
    )
    # The private preserve variants demand DenormPreserve and
    # SignedZeroInfNanPreserve. Ordinary cmpf retains native target semantics.
    return DescriptorRule(
        source_op=scalar.scalar_clampf if source_lanes == 0 else vector.vector_clampf,
        descriptor=recipe.emits[-1].descriptor,
        guards=(
            *(
                Guard.value_type(name, pattern)
                for name in ("value", "lower", "upper", "result")
            ),
            Guard.enum_attr_equals("mode", "ordered"),
            *descriptor_feature_guards(*(step.descriptor for step in recipe.emits)),
        ),
        emit=tuple(recipe.emits),
    )


def _native_extrema_rules() -> tuple[DescriptorRule, ...]:
    # GLSL special-value behavior requires float_controls2, not the older
    # SignedZeroInfNanPreserve. Native extrema need all three source permissions
    # and the width's DenormPreserve feature, which the descriptors require.
    rules = []
    for element in EXTREMA_MATH_OPERATIONS[0].source_types:
        for source_lanes in (0, 1, 2, 3, 4):
            lanes = max(1, source_lanes)
            suffix = element if lanes == 1 else f"v{lanes}{element}"
            pattern = (
                Scalar(element) if source_lanes == 0 else Vector(element, lanes=lanes)
            )
            for operation, scalar_op, vector_op, mode in _OPERATIONS:
                operands = (
                    ("value", "lower", "upper")
                    if operation == "clampf"
                    else ("lhs", "rhs")
                )
                native_operation = (
                    "minnumf" if operation.startswith("min") else "maxnumf"
                )
                descriptor = logical_core_descriptor(
                    f"spirv.op_ext_inst.glsl_std_450.{native_operation}.{suffix}"
                )
                lower_bounded = ValueRef.temporary("lower_bounded")
                clamp = operation == "clampf"
                emits = [
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.operand(operands[0]),
                            "rhs": ValueRef.operand(operands[1]),
                        },
                        results={
                            "dst": lower_bounded if clamp else ValueRef.result("result")
                        },
                        result_types={"dst": DescriptorResultType()} if clamp else None,
                    )
                ]
                if clamp:
                    # GLSL NClamp is undefined for reversed bounds; Loom's
                    # sequential clamp is total for those inputs as well.
                    descriptor = logical_core_descriptor(
                        f"spirv.op_ext_inst.glsl_std_450.minnumf.{suffix}"
                    )
                    emits.append(
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={
                                "lhs": lower_bounded,
                                "rhs": ValueRef.operand("upper"),
                            },
                            results={"dst": ValueRef.result("result")},
                        )
                    )
                rules.append(
                    DescriptorRule(
                        source_op=scalar_op if source_lanes == 0 else vector_op,
                        descriptor=descriptor,
                        guards=(
                            *(
                                Guard.value_type(name, pattern)
                                for name in (*operands, "result")
                            ),
                            *((Guard.enum_attr_equals("mode", mode),) if clamp else ()),
                            *(
                                Guard.instance_flags_has_all("fastmath", flag)
                                for flag in EXTREMA_MATH_OPERATIONS[
                                    0
                                ].required_fastmath_flags
                            ),
                            *descriptor_feature_guards(
                                *(step.descriptor for step in emits)
                            ),
                        ),
                        emit=tuple(emits),
                    )
                )
            rules.append(_native_ordered_clamp_rule(element, source_lanes))
    return tuple(rules)


def extrema_rules() -> tuple[DescriptorRule, ...]:
    """Returns valid native fast paths before portable integer recipes."""
    rules = list(_native_extrema_rules())
    for encoding in _ENCODINGS:
        representations = []
        if encoding.source_type in ("f16", "bf16", "f32", "f64"):
            representations.extend(
                (encoding.bit_width, lanes, True) for lanes in (0, 1, 2, 3, 4)
            )
            if encoding.bit_width == 16:
                representations.extend((32, lanes, True) for lanes in (0, 1))
        if encoding.source_type in ("bf16", "f8E4M3", "f8E5M2"):
            representations.append(
                (32 if encoding.source_type == "bf16" else 8, 0, False)
            )
        for integer_width, source_lanes, native_value in representations:
            lanes = max(1, source_lanes)
            pattern = (
                Scalar(encoding.source_type)
                if source_lanes == 0
                else Vector(encoding.source_type, lanes=lanes)
            )
            for operation, scalar_op, vector_op, mode in _OPERATIONS:
                recipe = IntegerRecipe(integer_width, lanes)
                operands = (
                    ("value", "lower", "upper")
                    if operation == "clampf"
                    else ("lhs", "rhs")
                )
                bits = [ValueRef.operand(operand) for operand in operands]
                if native_value:
                    bits = [
                        recipe.import_float_bits(
                            value, encoding.source_type, encoding.bit_width
                        )
                        for value in bits
                    ]
                destination = None if native_value else ValueRef.result("result")
                result = _extremum(
                    recipe,
                    encoding,
                    bits[0],
                    bits[1],
                    minimum=operation.startswith("min"),
                    mode=mode,
                    result=destination if operation != "clampf" else None,
                )
                if operation == "clampf":
                    result = _extremum(
                        recipe,
                        encoding,
                        result,
                        bits[2],
                        minimum=True,
                        mode=mode,
                        result=destination,
                    )
                if native_value:
                    recipe.export_float_bits(
                        result, encoding.source_type, encoding.bit_width
                    )
                rules.append(
                    DescriptorRule(
                        source_op=scalar_op if source_lanes == 0 else vector_op,
                        descriptor=recipe.emits[-1].descriptor,
                        guards=(
                            *(
                                Guard.value_type(name, pattern)
                                for name in (*operands, "result")
                            ),
                            *(
                                (Guard.enum_attr_equals("mode", mode),)
                                if operation == "clampf"
                                else ()
                            ),
                            *descriptor_feature_guards(
                                *(step.descriptor for step in recipe.emits)
                            ),
                        ),
                        emit=tuple(recipe.emits),
                        report_key="integer_float_extrema",
                    )
                )
    return tuple(rules)
