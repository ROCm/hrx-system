# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR arith ops."""

from __future__ import annotations

import math
from typing import Any

from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp

INTEGER_CMPI_PREDICATES = (
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
)

BINARY_OPS = {
    "arith.addf": ("scalar.addf", "vector.addf"),
    "arith.addi": ("scalar.addi", "vector.addi"),
    "arith.andi": ("scalar.andi", "vector.andi"),
    "arith.divf": ("scalar.divf", "vector.divf"),
    "arith.maximumf": ("scalar.maximumf", "vector.maximumf"),
    "arith.maxnumf": ("scalar.maxnumf", "vector.maxnumf"),
    "arith.minimumf": ("scalar.minimumf", "vector.minimumf"),
    "arith.minnumf": ("scalar.minnumf", "vector.minnumf"),
    "arith.mulf": ("scalar.mulf", "vector.mulf"),
    "arith.muli": ("scalar.muli", "vector.muli"),
    "arith.ori": ("scalar.ori", "vector.ori"),
    "arith.rotli": ("scalar.rotli", "vector.rotli"),
    "arith.rotri": ("scalar.rotri", "vector.rotri"),
    "arith.shli": ("scalar.shli", "vector.shli"),
    "arith.shrsi": ("scalar.shrsi", "vector.shrsi"),
    "arith.shrui": ("scalar.shrui", "vector.shrui"),
    "arith.subf": ("scalar.subf", "vector.subf"),
    "arith.subi": ("scalar.subi", "vector.subi"),
    "arith.xori": ("scalar.xori", "vector.xori"),
}

INDEX_BINARY_OPS = {
    "arith.addi": "index.add",
    "arith.andi": "index.andi",
    "arith.muli": "index.mul",
    "arith.ori": "index.ori",
    "arith.rotli": "index.rotli",
    "arith.rotri": "index.rotri",
    "arith.shli": "index.shli",
    "arith.shrsi": "index.shrsi",
    "arith.shrui": "index.shrui",
    "arith.subi": "index.sub",
    "arith.xori": "index.xori",
}

CAST_OPS = {
    "arith.bitcast": ("scalar.bitcast", "vector.bitcast"),
    "arith.extf": ("scalar.extf", "vector.extf"),
    "arith.extsi": ("scalar.extsi", "vector.extsi"),
    "arith.extui": ("scalar.extui", "vector.extui"),
    "arith.fptosi": ("scalar.fptosi", "vector.fptosi"),
    "arith.fptoui": ("scalar.fptoui", "vector.fptoui"),
    "arith.fptrunc": ("scalar.fptrunc", "vector.fptrunc"),
    "arith.sitofp": ("scalar.sitofp", "vector.sitofp"),
    "arith.trunci": ("scalar.trunci", "vector.trunci"),
    "arith.uitofp": ("scalar.uitofp", "vector.uitofp"),
}

UNARY_OPS = {
    "arith.ctlz": ("scalar.ctlzi", "vector.ctlzi", "index.ctlzi"),
    "arith.ctpop": ("scalar.ctpopi", "vector.ctpopi", "index.ctpopi"),
    "arith.cttz": ("scalar.cttzi", "vector.cttzi", "index.cttzi"),
}


def register(registry: ConverterRegistry) -> None:
    registry.register("arith.constant", convert_constant)
    registry.register("arith.cmpi", convert_cmpi)
    registry.register("arith.index_cast", convert_index_cast)
    registry.register("arith.index_castsi", convert_index_cast)
    registry.register("arith.index_castui", convert_index_cast)
    for op_name in BINARY_OPS:
        registry.register(op_name, convert_binary)
    for op_name in CAST_OPS:
        registry.register(op_name, convert_cast)
    for op_name in UNARY_OPS:
        registry.register(op_name, convert_unary)


def convert_constant(op: SourceOp, context: MlirConversionContext) -> bool:
    result = op.result()
    result_type = op.result_type()
    value_attr = op.attr("value")
    if value_attr is None:
        context.record_blocked(op.text, "arith.constant has no value attribute")
        return True
    if not context.require_top_level(op, _constant_target(value_attr, result_type)):
        return True

    if result_type in ("index", "offset"):
        value = str(value_attr.value)
        loom_result = context.build_constant(
            value, result_type, context.result_name(result)
        )
        context.map_result(result, loom_result, result_type)
        context.remember_constant(value, result_type, loom_result)
        context.record_converted(
            op.text,
            f"{context.ssa(loom_result)} = index.constant ... : {result_type}",
        )
        return True

    if result_type.startswith("vector<") and getattr(value_attr, "is_splat", False):
        splat = value_attr.get_splat_value()
        element_type = str(splat.type)
        scalar_value = _format_attr_value(splat)
        scalar = context.ensure_constant(
            scalar_value,
            element_type,
            f"{result.get_name().removeprefix('%')}_scalar",
        )
        loom_result = context.builder.vector.splat(
            scalar=scalar,
            results=[context.type(result_type)],
            name=context.result_name(result),
        )
        context.map_result(result, loom_result, result_type)
        context.record_converted(
            op.text,
            (
                f"{context.ssa(scalar)} = scalar.constant ... : {element_type}",
                f"{context.ssa(loom_result)} = vector.splat ...",
            ),
        )
        return True

    if _is_scalar_type(result_type):
        value = _format_attr_value(value_attr)
        loom_result = context.build_constant(
            value, result_type, context.result_name(result)
        )
        context.map_result(result, loom_result, result_type)
        context.remember_constant(value, result_type, loom_result)
        context.record_converted(
            op.text,
            f"{context.ssa(loom_result)} = scalar.constant ... : {result_type}",
        )
        return True

    context.record_blocked(
        op.text,
        f"arith.constant type `{result_type}` needs a value-kind converter",
    )
    return True


def convert_cmpi(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "index.cmp/scalar.cmpi"):
        return True
    mapped, missing = context.mapped_operands((op.operand(0), op.operand(1)))
    if mapped is None:
        context.record_blocked(
            op.text, f"missing converted operands: {', '.join(missing)}"
        )
        return True
    source_type = op.operand_type(0)
    target_op = "index.cmp" if source_type in ("index", "offset") else "scalar.cmpi"
    predicate = _integer_cmpi_predicate(op.attr("predicate"))
    if predicate is None:
        context.record_blocked(op.text, "unknown arith.cmpi predicate value")
        return True
    if target_op == "index.cmp":
        result = context.builder.index.cmp(
            predicate=predicate,
            lhs=mapped[0],
            rhs=mapped[1],
            results=[context.type("i1")],
            name=context.result_name(op.result(), "cmp"),
        )
    else:
        result = context.builder.scalar.cmpi(
            predicate=predicate,
            lhs=mapped[0],
            rhs=mapped[1],
            results=[context.type("i1")],
            name=context.result_name(op.result(), "cmp"),
        )
    context.map_result(op.result(), result, "i1")
    context.record_converted(op.text, f"{context.ssa(result)} = {target_op} ...")
    return True


def convert_binary(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "scalar/vector arithmetic"):
        return True
    mapped, missing = context.mapped_operands((op.operand(0), op.operand(1)))
    if mapped is None:
        context.record_blocked(
            op.text, f"missing converted operands: {', '.join(missing)}"
        )
        return True
    scalar_op, vector_op = BINARY_OPS[op.op_name]
    result_type = op.result_type()
    if result_type in {"index", "offset"}:
        target_op = INDEX_BINARY_OPS.get(op.op_name)
        if target_op is None:
            context.record_blocked(
                op.text,
                f"{op.op_name} does not map to address-domain arithmetic",
            )
            return True
    else:
        target_op = vector_op if result_type.startswith("vector<") else scalar_op
    result = context.build_binary(
        target_op,
        mapped[0],
        mapped[1],
        result_type,
        context.result_name(op.result()),
    )
    context.map_result(op.result(), result, result_type)
    context.record_converted(op.text, f"{context.ssa(result)} = {target_op} ...")
    return True


def convert_unary(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "scalar/vector/index unary arithmetic"):
        return True
    mapped = context.mapped(op.operand())
    if mapped is None:
        context.record_blocked(
            op.text, f"missing unary arithmetic operand: {op.operand().get_name()}"
        )
        return True
    scalar_op, vector_op, index_op = UNARY_OPS[op.op_name]
    result_type = op.result_type()
    if result_type == "index":
        target_op = index_op
    elif result_type == "offset":
        context.record_blocked(
            op.text,
            f"{op.op_name} does not map to offset-domain arithmetic",
        )
        return True
    else:
        target_op = vector_op if result_type.startswith("vector<") else scalar_op
    dialect_name, op_name = target_op.split(".", 1)
    result = getattr(getattr(context.builder, dialect_name), op_name)(
        input=mapped,
        results=[context.type(result_type)],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, result_type)
    context.record_converted(op.text, f"{context.ssa(result)} = {target_op} ...")
    return True


def convert_index_cast(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "index.cast"):
        return True
    mapped = context.mapped(op.operand())
    if mapped is None:
        context.record_blocked(
            op.text, f"missing arith index cast operand: {op.operand().get_name()}"
        )
        return True
    result_type = op.result_type()
    result = context.builder.index.cast(
        input=mapped,
        results=[context.type(result_type)],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, result_type)
    context.record_converted(op.text, f"{context.ssa(result)} = index.cast ...")
    return True


def convert_cast(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "scalar/vector cast"):
        return True
    mapped = context.mapped(op.operand())
    if mapped is None:
        context.record_blocked(
            op.text, f"missing cast operand: {op.operand().get_name()}"
        )
        return True
    scalar_op, vector_op = CAST_OPS[op.op_name]
    result_type = op.result_type()
    target_op = vector_op if result_type.startswith("vector<") else scalar_op
    dialect_name, op_name = target_op.split(".", 1)
    result = getattr(getattr(context.builder, dialect_name), op_name)(
        input=mapped,
        results=[context.type(result_type)],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, result_type)
    context.record_converted(op.text, f"{context.ssa(result)} = {target_op} ...")
    return True


def _constant_target(value_attr: object, result_type: str) -> str:
    if result_type in ("index", "offset"):
        return "index.constant"
    if result_type.startswith("vector<") and getattr(value_attr, "is_splat", False):
        return "scalar.constant + vector.splat"
    return "scalar.constant"


def _is_scalar_type(result_type: str) -> bool:
    return result_type in {"i1", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"}


def _format_attr_value(attr: Any) -> str:
    value = attr.value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return str(value)
    return str(value)


def _integer_cmpi_predicate(attr: Any | None) -> str | None:
    if attr is None:
        return None
    value = attr.value
    if value < 0 or value >= len(INTEGER_CMPI_PREDICATES):
        return None
    return INTEGER_CMPI_PREDICATES[value]
