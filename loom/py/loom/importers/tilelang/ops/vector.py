# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/licenses/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vector expression TileLang/TIR node converters."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    ExpressionOptions,
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import dtype, node_text
from loom.importers.tilelang.ops.topology import integer_value


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_expression("Broadcast", convert_broadcast)
    registry.register_expression("Ramp", convert_ramp)


def convert_broadcast(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Import a TIR Broadcast expression as vector.splat."""

    scalar = converter.convert_expr(getattr(expr, "value", None), context)
    lanes = vector_lanes(expr)
    if scalar is None or lanes is None:
        context.record_blocked(node_text(expr), "broadcast operands are not mapped")
        return None
    result_type = context.type_converter.vector_type(scalar.type, lanes)
    result = context.builder.vector.splat(
        scalar=scalar,
        results=[result_type],
        name=context.fresh_name("splat"),
    )
    context.map_value(expr, result, str(result_type))
    context.record_converted(node_text(expr), f"{context.ssa(result)} = vector.splat")
    return result


def convert_ramp(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import a TIR Ramp expression as vector.iota."""

    base = converter.convert_expr(
        getattr(expr, "base", None),
        context,
        index_like=options.index_like,
    )
    stride = converter.convert_expr(
        getattr(expr, "stride", None),
        context,
        index_like=options.index_like,
    )
    lanes = vector_lanes(expr)
    if base is None or stride is None or lanes is None:
        context.record_blocked(node_text(expr), "ramp operands are not mapped")
        return None
    element_dtype = "index" if options.index_like else dtype(expr)
    result_type = context.type_converter.vector_type(element_dtype, lanes)
    result = context.builder.vector.iota(
        base=base,
        step=stride,
        results=[result_type],
        name=context.fresh_name("ramp"),
    )
    context.map_value(expr, result, str(result_type))
    context.record_converted(node_text(expr), f"{context.ssa(result)} = vector.iota")
    return result


def vector_lanes(expr: object) -> int | None:
    """Returns the lane count from a TIR vector expression."""

    lanes = integer_value(getattr(expr, "lanes", None))
    if lanes is not None:
        return lanes
    text = str(dtype(expr))
    _head, separator, tail = text.rpartition("x")
    if separator and tail.isdecimal():
        return int(tail)
    return None
