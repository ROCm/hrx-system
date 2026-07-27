# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar expression TileLang/TIR node converters."""

from __future__ import annotations

from typing import cast

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    ExpressionOptions,
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import dtype, node_kind, node_text, source_name
from loom.importers.tilelang.ops.topology import integer_value
from loom.ir import I1, INDEX, ShapedType, Type, TypeKind


def register(registry: TileLangConverterRegistry) -> None:
    for node_name in ("IntImm", "FloatImm"):
        registry.register_expression(node_name, convert_immediate)
    registry.register_expression("Var", convert_var)
    registry.register_expression("Cast", convert_cast)
    registry.register_expression("Not", convert_not)
    registry.register_expression("Select", convert_select)
    for node_name in sorted(
        set(_BINARY_INDEX_OPS) | set(_BINARY_INTEGER_OPS) | set(_BINARY_FLOAT_OPS)
    ):
        registry.register_expression(node_name, convert_binary_expr)
    for node_name in _COMPARISON_PREDICATES:
        registry.register_expression(node_name, convert_compare)
    for node_name in _BOOLEAN_BINARY_OPS:
        registry.register_expression(node_name, convert_boolean_binary)


def convert_immediate(
    expr: object,
    context: TileLangConversionContext,
    _converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import an integer or floating-point immediate."""

    value = getattr(expr, "value", expr)
    value_type = "index" if options.index_like else dtype(expr)
    result_type = context.type_converter.map_dtype(
        value_type,
        index_like=options.index_like,
    )
    result = context.ensure_typed_constant(
        value,
        result_type,
        _constant_base_name(value, index_like=options.index_like),
    )
    context.map_value(expr, result, value_type)
    return result


def _constant_base_name(value: object, *, index_like: bool) -> str:
    if not index_like:
        return "const"
    text = str(value)
    if text.isdecimal():
        return f"c{text}"
    if text.startswith("-") and text[1:].isdecimal():
        return f"cm{text[1:]}"
    return "c"


def convert_var(
    expr: object,
    context: TileLangConversionContext,
    _converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Resolve a mapped TIR variable."""

    mapped_var = context.mapped(expr)
    if mapped_var is None:
        context.record_blocked(node_text(expr), "variable is not mapped")
        return None
    if options.index_like:
        return _coerce_index_like_var(expr, mapped_var, context)
    return mapped_var


def _coerce_index_like_var(
    expr: object,
    mapped_var: ValueRef,
    context: TileLangConversionContext,
) -> ValueRef | None:
    value_type = str(mapped_var.type)
    if value_type in ("index", "offset"):
        return mapped_var
    if not _is_integer_type(value_type):
        context.record_blocked(
            node_text(expr),
            f"variable of type {value_type} cannot be used as an index",
        )
        return None
    existing = context.mapped_index_value(expr)
    if existing is not None:
        return existing
    name = context.fresh_name(f"{source_name(expr, fallback='idx')}_idx")
    result = context.builder.index.cast(
        input=mapped_var,
        results=[INDEX],
        name=name,
    )
    context.map_index_value(expr, result)
    context.record_converted(node_text(expr), f"{context.ssa(result)} = index.cast")
    return result


def convert_cast(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Import a scalar cast expression."""

    value = getattr(expr, "value", None)
    input_value = converter.convert_expr(value, context)
    if input_value is None:
        context.record_blocked(node_text(expr), "cast operand is not mapped")
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    input_type = input_value.type
    if str(input_type) == str(result_type):
        context.map_value(expr, input_value, str(result_type))
        return input_value
    if _is_vector_type(input_type) or _is_vector_type(result_type):
        return _convert_vector_cast(
            expr,
            context,
            input_value,
            input_type,
            result_type,
            value,
        )
    if _is_index_type(input_type) or _is_index_type(result_type):
        result = context.builder.index.cast(
            input=input_value,
            results=[result_type],
            name=context.fresh_name("cast"),
        )
        context.map_value(expr, result, str(result_type))
        return result
    builder_name = _cast_builder_name(
        dtype(value), input_type, dtype(expr), result_type
    )
    if builder_name is None:
        context.record_blocked(
            node_text(expr),
            f"no scalar cast builder from {input_type} to {result_type}",
        )
        return None
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, builder_name)(
            input=input_value,
            results=[result_type],
            name=context.fresh_name(builder_name),
        ),
    )
    context.map_value(expr, result, str(result_type))
    return result


def _convert_vector_cast(
    expr: object,
    context: TileLangConversionContext,
    input_value: ValueRef,
    input_type: Type,
    result_type: Type,
    source_value: object,
) -> ValueRef | None:
    if not _is_vector_type(input_type) or not _is_vector_type(result_type):
        context.record_blocked(
            node_text(expr),
            "vector cast input and result must both be vector-typed",
        )
        return None
    input_vector_type = cast(ShapedType, input_type)
    result_vector_type = cast(ShapedType, result_type)
    if input_vector_type.dims != result_vector_type.dims:
        context.record_blocked(
            node_text(expr),
            "vector cast input and result shapes must match",
        )
        return None
    builder_name = _cast_builder_name(
        dtype(source_value),
        input_vector_type.element_type,
        dtype(expr),
        result_vector_type.element_type,
    )
    if builder_name is None:
        context.record_blocked(
            node_text(expr),
            f"no vector cast builder from {input_type} to {result_type}",
        )
        return None
    result = cast(
        ValueRef,
        getattr(context.builder.vector, builder_name)(
            input=input_value,
            results=[result_vector_type],
            name=context.fresh_name(builder_name),
        ),
    )
    context.map_value(expr, result, str(result_vector_type))
    return result


def convert_binary_expr(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import a scalar or index binary expression."""

    source_lhs = getattr(expr, "a", None)
    source_rhs = getattr(expr, "b", None)
    index_like = (
        options.index_like
        or _source_is_index_like(source_lhs, context)
        or _source_is_index_like(source_rhs, context)
    )
    kind = type(expr).__name__
    if index_like:
        madd = _convert_index_madd_expr(
            expr,
            source_lhs,
            source_rhs,
            context,
            converter,
            kind,
        )
        if madd is not None:
            return madd
        if kind not in _BINARY_INDEX_OPS:
            context.record_blocked(node_text(expr), f"no index builder for {kind}")
            return None
    lhs = converter.convert_expr(
        source_lhs,
        context,
        index_like=index_like,
    )
    if lhs is None:
        context.record_blocked(node_text(expr), "binary operands are not mapped")
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    if (
        not index_like
        and kind == "FloorMod"
        and _is_integer_type(str(result_type))
        and not _is_unsigned_dtype(dtype(expr))
        and not _is_vector_type(lhs.type)
        and _is_integer_type(str(lhs.type))
    ):
        result = _convert_signed_floor_mod_power_of_two_expr(
            expr,
            context,
            lhs,
            result_type,
        )
        if result is not None:
            return result
    rhs_index_like = index_like or _is_index_type(lhs.type)
    rhs = converter.convert_expr(
        source_rhs,
        context,
        index_like=rhs_index_like,
    )
    if rhs is None:
        context.record_blocked(node_text(expr), "binary operands are not mapped")
        return None
    index_like = index_like or _is_index_type(lhs.type) or _is_index_type(rhs.type)
    if index_like and not _is_index_type(lhs.type):
        lhs = converter.convert_expr(source_lhs, context, index_like=True)
        if lhs is None:
            context.record_blocked(node_text(expr), "binary operands are not mapped")
            return None
    if index_like and not _is_index_type(rhs.type):
        rhs = converter.convert_expr(source_rhs, context, index_like=True)
        if rhs is None:
            context.record_blocked(node_text(expr), "binary operands are not mapped")
            return None
    if _is_vector_type(lhs.type) or _is_vector_type(rhs.type):
        return _convert_vector_binary_expr(expr, context, lhs, rhs)
    if index_like:
        builder_name = _BINARY_INDEX_OPS[kind]
        builder = getattr(context.builder.index, builder_name)
        result = cast(
            ValueRef,
            builder(
                lhs=lhs,
                rhs=rhs,
                results=[INDEX],
                name=context.fresh_name(builder_name),
            ),
        )
        context.map_value(expr, result, "index")
        return result
    if str(result_type).startswith("f"):
        scalar_builder_name = _BINARY_FLOAT_OPS.get(kind)
    elif _is_unsigned_dtype(dtype(expr)):
        scalar_builder_name = _BINARY_UNSIGNED_INTEGER_OPS.get(kind)
    elif kind == "FloorMod" and _is_integer_type(str(result_type)):
        return _convert_signed_floor_mod_expr(expr, context, lhs, rhs, result_type)
    else:
        scalar_builder_name = _BINARY_INTEGER_OPS.get(kind)
    if scalar_builder_name is None:
        context.record_blocked(node_text(expr), f"no scalar builder for {kind}")
        return None
    builder = getattr(context.builder.scalar, scalar_builder_name)
    builder_kwargs = (
        context.float_operation_kwargs() if str(result_type).startswith("f") else {}
    )
    result = cast(
        ValueRef,
        builder(
            lhs=lhs,
            rhs=rhs,
            results=[result_type],
            name=context.fresh_name(scalar_builder_name),
            **builder_kwargs,
        ),
    )
    context.map_value(expr, result, str(result_type))
    return result


def _convert_signed_floor_mod_power_of_two_expr(
    expr: object,
    context: TileLangConversionContext,
    lhs: ValueRef,
    result_type: Type,
) -> ValueRef | None:
    divisor = integer_value(getattr(expr, "b", None))
    if divisor is None or divisor <= 0 or divisor & (divisor - 1) != 0:
        return None
    mask = context.ensure_typed_constant(
        divisor - 1,
        result_type,
        "floor_mod_mask",
    )
    result = cast(
        ValueRef,
        context.builder.scalar.andi(
            lhs=lhs,
            rhs=mask,
            results=[result_type],
            name=context.fresh_name("floor_mod"),
        ),
    )
    context.map_value(expr, result, str(result_type))
    return result


def _convert_signed_floor_mod_expr(
    expr: object,
    context: TileLangConversionContext,
    lhs: ValueRef,
    rhs: ValueRef,
    result_type: Type,
) -> ValueRef:
    remainder = cast(
        ValueRef,
        context.builder.scalar.remsi(
            lhs=lhs,
            rhs=rhs,
            results=[result_type],
            name=context.fresh_name("remsi"),
        ),
    )
    zero = context.ensure_typed_constant("0", result_type, "zero")
    divisor_nonnegative = context.builder.scalar.cmpi(
        predicate="sge",
        lhs=rhs,
        rhs=zero,
        results=[I1],
        name=context.fresh_name("divisor_nonnegative"),
    )
    remainder_nonnegative = context.builder.scalar.cmpi(
        predicate="sge",
        lhs=remainder,
        rhs=zero,
        results=[I1],
        name=context.fresh_name("remainder_nonnegative"),
    )
    positive_case = context.builder.scalar.andi(
        lhs=divisor_nonnegative,
        rhs=remainder_nonnegative,
        results=[I1],
        name=context.fresh_name("positive_floor_mod"),
    )
    divisor_negative = context.builder.scalar.cmpi(
        predicate="slt",
        lhs=rhs,
        rhs=zero,
        results=[I1],
        name=context.fresh_name("divisor_negative"),
    )
    remainder_nonpositive = context.builder.scalar.cmpi(
        predicate="sle",
        lhs=remainder,
        rhs=zero,
        results=[I1],
        name=context.fresh_name("remainder_nonpositive"),
    )
    negative_case = context.builder.scalar.andi(
        lhs=divisor_negative,
        rhs=remainder_nonpositive,
        results=[I1],
        name=context.fresh_name("negative_floor_mod"),
    )
    keep_remainder = context.builder.scalar.ori(
        lhs=positive_case,
        rhs=negative_case,
        results=[I1],
        name=context.fresh_name("floor_mod_keep_remainder"),
    )
    adjusted = context.builder.scalar.addi(
        lhs=remainder,
        rhs=rhs,
        results=[result_type],
        name=context.fresh_name("floor_mod_adjusted"),
    )
    result = context.builder.scf.select(
        condition=keep_remainder,
        true_value=remainder,
        false_value=adjusted,
        results=[result_type],
        name=context.fresh_name("floor_mod"),
    )
    context.map_value(expr, result, str(result_type))
    return result


def _convert_index_madd_expr(
    expr: object,
    source_lhs: object,
    source_rhs: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    kind: str,
) -> ValueRef | None:
    if kind != "Add":
        return None
    if node_kind(source_lhs) == "Mul":
        return _build_index_madd_expr(
            expr,
            source_lhs,
            source_rhs,
            context,
            converter,
        )
    if node_kind(source_rhs) == "Mul":
        return _build_index_madd_expr(
            expr,
            source_rhs,
            source_lhs,
            context,
            converter,
        )
    return None


def _build_index_madd_expr(
    expr: object,
    source_product: object,
    source_addend: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> ValueRef | None:
    converter.operation_counts[node_kind(source_product)] += 1
    a = converter.convert_expr(
        getattr(source_product, "a", None),
        context,
        index_like=True,
    )
    b = converter.convert_expr(
        getattr(source_product, "b", None),
        context,
        index_like=True,
    )
    c = converter.convert_expr(source_addend, context, index_like=True)
    if a is None or b is None or c is None:
        context.record_blocked(node_text(expr), "index madd operands are not mapped")
        return None
    result = context.builder.index.madd(
        a=a,
        b=b,
        c=c,
        results=[INDEX],
        name=context.fresh_name("madd"),
    )
    context.map_value(expr, result, "index")
    return result


def convert_compare(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Import a scalar comparison expression."""

    source_lhs = getattr(expr, "a", None)
    source_rhs = getattr(expr, "b", None)
    index_like = _source_is_index_like(source_lhs, context) or _source_is_index_like(
        source_rhs, context
    )
    lhs = converter.convert_expr(source_lhs, context, index_like=index_like)
    if lhs is None:
        context.record_blocked(node_text(expr), "comparison operands are not mapped")
        return None
    rhs_index_like = index_like or _is_index_type(lhs.type)
    rhs = converter.convert_expr(source_rhs, context, index_like=rhs_index_like)
    if rhs is None:
        context.record_blocked(node_text(expr), "comparison operands are not mapped")
        return None
    index_like = index_like or _is_index_type(lhs.type) or _is_index_type(rhs.type)
    if index_like and not _is_index_type(lhs.type):
        lhs = converter.convert_expr(source_lhs, context, index_like=True)
        if lhs is None:
            context.record_blocked(
                node_text(expr), "comparison operands are not mapped"
            )
            return None
    if index_like and not _is_index_type(rhs.type):
        rhs = converter.convert_expr(source_rhs, context, index_like=True)
        if rhs is None:
            context.record_blocked(
                node_text(expr), "comparison operands are not mapped"
            )
            return None
    kind = type(expr).__name__
    if _is_vector_type(lhs.type) or _is_vector_type(rhs.type):
        return _convert_vector_compare(expr, context, kind, source_lhs, lhs, rhs)
    source_type = str(lhs.type)
    if source_type in ("index", "offset"):
        predicate = _index_predicate(kind)
        builder = context.builder.index.cmp
        target = "index.cmp"
    elif _is_float_type(source_type):
        predicate = _FLOAT_COMPARISON_PREDICATES[kind]
        builder = context.builder.scalar.cmpf
        target = "scalar.cmpf"
    else:
        predicate = _integer_predicate(kind, dtype(source_lhs))
        builder = context.builder.scalar.cmpi
        target = "scalar.cmpi"
    result = builder(
        predicate=predicate,
        lhs=lhs,
        rhs=rhs,
        results=[I1],
        name=context.fresh_name("cmp"),
    )
    context.map_value(expr, result, "i1")
    context.record_converted(node_text(expr), f"{context.ssa(result)} = {target}")
    return result


def _convert_vector_binary_expr(
    expr: object,
    context: TileLangConversionContext,
    lhs: ValueRef,
    rhs: ValueRef,
) -> ValueRef | None:
    kind = type(expr).__name__
    if not _is_vector_type(lhs.type) or not _is_vector_type(rhs.type):
        context.record_blocked(
            node_text(expr),
            "vector binary operands must both be vector-typed",
        )
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    if not _is_vector_type(result_type):
        context.record_blocked(
            node_text(expr), "vector binary result is not vector-typed"
        )
        return None
    vector_type = cast(ShapedType, result_type)
    if _is_float_type(str(vector_type.element_type)):
        builder_name = _BINARY_FLOAT_OPS.get(kind)
    elif _is_unsigned_dtype(dtype(expr)):
        builder_name = _BINARY_UNSIGNED_INTEGER_OPS.get(kind)
    else:
        builder_name = _BINARY_INTEGER_OPS.get(kind)
    if builder_name is None:
        context.record_blocked(node_text(expr), f"no vector builder for {kind}")
        return None
    builder = getattr(context.builder.vector, builder_name)
    builder_kwargs = (
        context.float_operation_kwargs()
        if _is_float_type(str(vector_type.element_type))
        else {}
    )
    result = cast(
        ValueRef,
        builder(
            lhs=lhs,
            rhs=rhs,
            results=[vector_type],
            name=context.fresh_name(builder_name),
            **builder_kwargs,
        ),
    )
    context.map_value(expr, result, str(vector_type))
    return result


def _convert_vector_compare(
    expr: object,
    context: TileLangConversionContext,
    kind: str,
    source_lhs: object,
    lhs: ValueRef,
    rhs: ValueRef,
) -> ValueRef | None:
    if not _is_vector_type(lhs.type) or not _is_vector_type(rhs.type):
        context.record_blocked(
            node_text(expr),
            "vector comparison operands must both be vector-typed",
        )
        return None
    source_type = cast(ShapedType, lhs.type)
    result_type = ShapedType(TypeKind.VECTOR, I1, source_type.dims)
    element_type = str(source_type.element_type)
    if _is_float_type(element_type):
        predicate = _FLOAT_COMPARISON_PREDICATES[kind]
        builder = context.builder.vector.cmpf
        target = "vector.cmpf"
    else:
        predicate = _integer_predicate(kind, dtype(source_lhs))
        builder = context.builder.vector.cmpi
        target = "vector.cmpi"
    result = builder(
        predicate=predicate,
        lhs=lhs,
        rhs=rhs,
        results=[result_type],
        name=context.fresh_name("cmp"),
    )
    context.map_value(expr, result, str(result_type))
    context.record_converted(node_text(expr), f"{context.ssa(result)} = {target}")
    return result


def convert_boolean_binary(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Import scalar boolean and/or expressions."""

    lhs = converter.convert_expr(getattr(expr, "a", None), context)
    rhs = converter.convert_expr(getattr(expr, "b", None), context)
    if lhs is None or rhs is None:
        context.record_blocked(node_text(expr), "boolean operands are not mapped")
        return None
    builder_name = _BOOLEAN_BINARY_OPS[type(expr).__name__]
    if _is_vector_type(lhs.type) or _is_vector_type(rhs.type):
        if not _is_vector_type(lhs.type) or not _is_vector_type(rhs.type):
            context.record_blocked(
                node_text(expr),
                "vector boolean operands must both be vector-typed",
            )
            return None
        result = cast(
            ValueRef,
            getattr(context.builder.vector, builder_name)(
                lhs=lhs,
                rhs=rhs,
                results=[lhs.type],
                name=context.fresh_name(builder_name),
            ),
        )
        context.map_value(expr, result, str(result.type))
        return result
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, builder_name)(
            lhs=lhs,
            rhs=rhs,
            results=[I1],
            name=context.fresh_name(builder_name),
        ),
    )
    context.map_value(expr, result, "i1")
    return result


def convert_not(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Import scalar boolean negation."""

    value = converter.convert_expr(getattr(expr, "a", None), context)
    if value is None:
        context.record_blocked(node_text(expr), "not operand is not mapped")
        return None
    true_value = context.ensure_constant("1", "bool", "true")
    if _is_vector_type(value.type):
        true_vector = context.builder.vector.splat(
            scalar=true_value,
            results=[value.type],
            name=context.fresh_name("true"),
        )
        result = context.builder.vector.xori(
            lhs=value,
            rhs=true_vector,
            results=[value.type],
            name=context.fresh_name("not"),
        )
        context.map_value(expr, result, str(result.type))
        return result
    result = context.builder.scalar.xori(
        lhs=value,
        rhs=true_value,
        results=[I1],
        name=context.fresh_name("not"),
    )
    context.map_value(expr, result, "i1")
    return result


def convert_select(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Import a scalar select expression as scf.select."""

    condition = converter.convert_expr(getattr(expr, "condition", None), context)
    true_value = converter.convert_expr(getattr(expr, "true_value", None), context)
    false_value = converter.convert_expr(getattr(expr, "false_value", None), context)
    if condition is None or true_value is None or false_value is None:
        context.record_blocked(node_text(expr), "select operands are not mapped")
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    if (
        _is_vector_type(condition.type)
        or _is_vector_type(true_value.type)
        or _is_vector_type(false_value.type)
    ):
        if (
            not _is_vector_type(condition.type)
            or not _is_vector_type(true_value.type)
            or not _is_vector_type(false_value.type)
            or not _is_vector_type(result_type)
        ):
            context.record_blocked(
                node_text(expr),
                "vector select operands and result must all be vector-typed",
            )
            return None
        result = context.builder.vector.select(
            condition=condition,
            true_value=true_value,
            false_value=false_value,
            results=[result_type],
            name=context.fresh_name("select"),
        )
        context.map_value(expr, result, str(result_type))
        return result
    result = context.builder.scf.select(
        condition=condition,
        true_value=true_value,
        false_value=false_value,
        results=[result_type],
        name=context.fresh_name("select"),
    )
    context.map_value(expr, result, str(result_type))
    return result


def _source_is_index_like(
    source: object,
    context: TileLangConversionContext,
) -> bool:
    mapped = context.mapped(source)
    return mapped is not None and _is_index_type(mapped.type)


def _is_vector_type(value_type: object) -> bool:
    return (
        isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VECTOR
    )


def _is_index_type(value_type: object) -> bool:
    return str(value_type) in ("index", "offset")


def _is_float_type(value_type: str) -> bool:
    return value_type in ("f16", "bf16", "f32", "f64")


def _is_integer_type(value_type: str) -> bool:
    return (
        value_type != "i1"
        and len(value_type) > 1
        and value_type[0] == "i"
        and value_type[1].isdigit()
    )


def _is_unsigned_dtype(source_dtype: object) -> bool:
    return str(source_dtype).startswith("uint")


def _integer_predicate(kind: str, source_dtype: object) -> str:
    predicates = (
        _UNSIGNED_COMPARISON_PREDICATES
        if _is_unsigned_dtype(source_dtype)
        else _SIGNED_COMPARISON_PREDICATES
    )
    return predicates[kind]


def _index_predicate(kind: str) -> str:
    return _SIGNED_COMPARISON_PREDICATES[kind]


def _cast_builder_name(
    source_dtype: object,
    input_type: Type,
    result_dtype: object,
    result_type: Type,
) -> str | None:
    input_text = str(input_type)
    result_text = str(result_type)
    if _is_float_type(input_text) and _is_float_type(result_text):
        return (
            "extf"
            if _type_bit_width(result_text) > _type_bit_width(input_text)
            else "fptrunc"
        )
    if _is_float_type(input_text):
        return "fptoui" if _is_unsigned_dtype(result_dtype) else "fptosi"
    if _is_float_type(result_text):
        return "uitofp" if _is_unsigned_dtype(source_dtype) else "sitofp"
    if input_text in ("index", "offset") or result_text in ("index", "offset"):
        return None
    if _type_bit_width(result_text) > _type_bit_width(input_text):
        return "extui" if _is_unsigned_dtype(source_dtype) else "extsi"
    if _type_bit_width(result_text) < _type_bit_width(input_text):
        return "trunci"
    return "bitcast"


def _type_bit_width(value_type: str) -> int:
    if value_type == "bf16":
        return 16
    if len(value_type) >= 2 and value_type[1:].isdigit():
        return int(value_type[1:])
    return 0


_BINARY_INDEX_OPS = {
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "FloorDiv": "div",
    "FloorMod": "rem",
    "Mod": "rem",
    "Min": "min",
    "Max": "max",
}

_BINARY_INTEGER_OPS = {
    "Add": "addi",
    "Sub": "subi",
    "Mul": "muli",
    "FloorDiv": "floordivsi",
    "Mod": "remsi",
    "Div": "divsi",
    "Min": "minsi",
    "Max": "maxsi",
}

_BINARY_UNSIGNED_INTEGER_OPS = {
    "Add": "addi",
    "Sub": "subi",
    "Mul": "muli",
    "FloorDiv": "divui",
    "FloorMod": "remui",
    "Mod": "remui",
    "Div": "divui",
    "Min": "minui",
    "Max": "maxui",
}

_BINARY_FLOAT_OPS = {
    "Add": "addf",
    "Sub": "subf",
    "Mul": "mulf",
    "Div": "divf",
    "Min": "minimumf",
    "Max": "maximumf",
}

_COMPARISON_PREDICATES = {
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
}

_SIGNED_COMPARISON_PREDICATES = {
    "EQ": "eq",
    "NE": "ne",
    "LT": "slt",
    "LE": "sle",
    "GT": "sgt",
    "GE": "sge",
}

_UNSIGNED_COMPARISON_PREDICATES = {
    "EQ": "eq",
    "NE": "ne",
    "LT": "ult",
    "LE": "ule",
    "GT": "ugt",
    "GE": "uge",
}

_FLOAT_COMPARISON_PREDICATES = {
    "EQ": "oeq",
    "NE": "one",
    "LT": "olt",
    "LE": "ole",
    "GT": "ogt",
    "GE": "oge",
}

_BOOLEAN_BINARY_OPS = {
    "And": "andi",
    "Or": "ori",
}
