# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR vector ops."""

from __future__ import annotations

from typing import Any

from loom.builder import ValueRef
from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp


def register(registry: ConverterRegistry) -> None:
    registry.register("vector.broadcast", convert_broadcast)
    registry.register("vector.extract", convert_extract)
    registry.register("vector.insert", convert_insert)
    registry.register("vector.multi_reduction", convert_multi_reduction)
    registry.register("vector.reduction", convert_reduction)
    registry.register("vector.transfer_read", convert_transfer_read)
    registry.register("vector.transfer_write", convert_transfer_write)


def convert_broadcast(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "vector.splat/vector.broadcast"):
        return True
    source = op.operand()
    mapped = context.mapped(source)
    if mapped is None:
        context.record_blocked(
            op.text, f"missing converted operand: {source.get_name()}"
        )
        return True
    source_type = op.operand_type()
    result_type = op.result_type()
    if source_type.startswith("vector<"):
        result = context.builder.vector.broadcast(
            source=mapped,
            results=[context.type(result_type)],
            name=context.result_name(op.result()),
        )
        target = "vector.broadcast"
    else:
        result = context.builder.vector.splat(
            scalar=mapped,
            results=[context.type(result_type)],
            name=context.result_name(op.result()),
        )
        target = "vector.splat"
    context.map_result(op.result(), result, result_type)
    context.record_converted(op.text, f"{context.ssa(result)} = {target} ...")
    return True


def convert_extract(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "vector.extract"):
        return True
    source = op.operand()
    mapped = context.mapped(source)
    if mapped is None:
        context.record_blocked(
            op.text, f"missing converted vector: {source.get_name()}"
        )
        return True
    static_position = context.attr_decoder.dense_i64_array(op.attr("static_position"))
    result = context.builder.vector.extract(
        source=mapped,
        indices=list(static_position),
        results=[context.type(op.result_type())],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, op.result_type())
    context.record_converted(op.text, f"{context.ssa(result)} = vector.extract ...")
    return True


def convert_insert(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "vector.insert"):
        return True
    mapped, missing = context.mapped_operands((op.operand(0), op.operand(1)))
    if mapped is None:
        context.record_blocked(
            op.text, f"missing vector.insert operands: {', '.join(missing)}"
        )
        return True
    static_position = context.attr_decoder.dense_i64_array(op.attr("static_position"))
    result = context.builder.vector.insert(
        value=mapped[0],
        dest=mapped[1],
        indices=list(static_position),
        results=[context.type(op.result_type())],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, op.result_type())
    context.record_converted(op.text, f"{context.ssa(result)} = vector.insert ...")
    return True


def convert_reduction(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "vector.reduce"):
        return True
    source = op.operand()
    mapped = context.mapped(source)
    if mapped is None:
        context.record_blocked(
            op.text, f"missing converted vector: {source.get_name()}"
        )
        return True
    result_type = op.result_type()
    kind = _vector_reduction_kind(op.attr("kind"), result_type)
    if kind is None:
        context.record_blocked(op.text, "unsupported vector.reduction kind")
        return True
    init = _reduction_identity(kind, result_type, context)
    if init is None:
        context.record_blocked(
            op.text, f"vector.reduction<{kind}> needs explicit identity support"
        )
        return True
    result = context.builder.vector.reduce(
        kind=kind,
        input=mapped,
        init=init,
        results=[context.type(result_type)],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, result_type)
    context.record_converted(
        op.text, f"{context.ssa(result)} = vector.reduce<{kind}> ..."
    )
    return True


def convert_multi_reduction(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "vector.reduce.axes"):
        return True
    mapped, missing = context.mapped_operands((op.operand(0), op.operand(1)))
    if mapped is None:
        context.record_blocked(
            op.text, f"missing vector.multi_reduction operands: {', '.join(missing)}"
        )
        return True
    result_type = op.result_type()
    kind = _vector_reduction_kind(op.attr("kind"), result_type)
    if kind is None:
        context.record_blocked(op.text, "unsupported vector.multi_reduction kind")
        return True
    axes = context.attr_decoder.dense_i64_array(op.attr("reduction_dims"))
    result = context.builder.vector.reduce_axes(
        kind=kind,
        input=mapped[0],
        init=mapped[1],
        axes=list(axes),
        results=[context.type(result_type)],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, result_type)
    context.record_converted(
        op.text, f"{context.ssa(result)} = vector.reduce.axes<{kind}> ..."
    )
    return True


def convert_transfer_read(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "vector.load/vector.load.mask"):
        return True
    operands = _transfer_read_operands(op, context)
    if operands is None:
        return True
    view_value, indices_value = operands
    view = context.mapped(view_value)
    indices, missing = _mapped_values(indices_value, context)
    if view is None:
        missing = (view_value.get_name(), *missing)
    if missing or view is None:
        context.record_blocked(
            op.text, f"missing converted operands: {', '.join(missing)}"
        )
        return True
    if context.mapped_value_type(view_value) is None:
        context.record_blocked(op.text, "missing Loom view type for transfer source")
        return True
    vector_type = op.result_type()
    result = context.builder.vector.load(
        view=view,
        indices=list(indices),
        results=[context.type(vector_type)],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, vector_type)
    context.record_converted(op.text, f"{context.ssa(result)} = vector.load ...")
    return True


def convert_transfer_write(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "vector.store/vector.store.mask"):
        return True
    operands = _transfer_write_operands(op, context)
    if operands is None:
        return True
    value_value, view_value, indices_value = operands
    value = context.mapped(value_value)
    view = context.mapped(view_value)
    indices, missing = _mapped_values(indices_value, context)
    if value is None:
        missing = (value_value.get_name(), *missing)
    if view is None:
        missing = (view_value.get_name(), *missing)
    if missing or value is None or view is None:
        context.record_blocked(
            op.text, f"missing converted operands: {', '.join(missing)}"
        )
        return True
    if context.mapped_value_type(view_value) is None:
        context.record_blocked(
            op.text, "missing Loom view type for transfer destination"
        )
        return True
    context.builder.vector.store(
        value=value,
        view=view,
        indices=list(indices),
    )
    context.record_converted(op.text, "vector.store ...")
    return True


def _transfer_read_operands(
    op: SourceOp,
    context: MlirConversionContext,
) -> tuple[Any, tuple[Any, ...]] | None:
    segments = context.attr_decoder.dense_i32_array(op.attr("operandSegmentSizes"))
    index_count = segments[1]
    padding_count = segments[2]
    mask_count = segments[3]
    if mask_count:
        context.record_blocked(op.text, "masked vector.transfer_read needs load.mask")
        return None
    if padding_count and not _all_in_bounds(op.attr("in_bounds")):
        context.record_blocked(
            op.text,
            "padded vector.transfer_read needs load.mask or explicit boundary logic",
        )
        return None
    return op.operand(0), tuple(op.operands[1 : 1 + index_count])


def _transfer_write_operands(
    op: SourceOp,
    context: MlirConversionContext,
) -> tuple[Any, Any, tuple[Any, ...]] | None:
    segments = context.attr_decoder.dense_i32_array(op.attr("operandSegmentSizes"))
    index_count = segments[2]
    mask_count = segments[3]
    if mask_count:
        context.record_blocked(op.text, "masked vector.transfer_write needs store.mask")
        return None
    if not _all_in_bounds(op.attr("in_bounds")):
        context.record_blocked(
            op.text,
            "out-of-bounds vector.transfer_write needs store.mask or boundary logic",
        )
        return None
    return op.operand(0), op.operand(1), tuple(op.operands[2 : 2 + index_count])


def _all_in_bounds(attr: object | None) -> bool:
    if attr is None:
        return False
    text = str(attr).strip()
    if not text.startswith("[") or not text.endswith("]"):
        return False
    values = [piece.strip() for piece in text[1:-1].split(",") if piece.strip()]
    return bool(values) and all(value == "true" for value in values)


def _mapped_values(
    values: tuple[Any, ...],
    context: MlirConversionContext,
) -> tuple[tuple[ValueRef, ...], tuple[str, ...]]:
    mapped = []
    missing: list[str] = []
    for value in values:
        loom_value = context.mapped(value)
        if loom_value is None:
            missing.append(value.get_name())
        else:
            mapped.append(loom_value)
    return tuple(mapped), tuple(missing)


def _vector_reduction_kind(attr: object | None, result_type: str) -> str | None:
    text = str(attr)
    prefix = "#vector.kind<"
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    kind = text[len(prefix) : -1]
    if kind == "add":
        return "addf" if _is_float_type(result_type) else "addi"
    if kind == "mul":
        return "mulf" if _is_float_type(result_type) else "muli"
    if kind in {
        "minsi",
        "maxsi",
        "minui",
        "maxui",
        "minimumf",
        "maximumf",
        "minnumf",
        "maxnumf",
        "andi",
        "ori",
        "xori",
    }:
        return kind
    return None


def _reduction_identity(
    kind: str,
    result_type: str,
    context: MlirConversionContext,
) -> ValueRef | None:
    if kind in {"addf", "addi", "ori", "xori"}:
        return context.ensure_constant(
            "0.0" if kind == "addf" else "0",
            result_type,
            f"zero_{result_type}",
        )
    if kind == "mulf":
        return context.ensure_constant("1.0", result_type, f"one_{result_type}")
    if kind in {"muli", "andi"}:
        return context.ensure_constant(
            "1" if kind == "muli" else "-1",
            result_type,
            f"identity_{result_type}",
        )
    return None


def _is_float_type(value_type: str) -> bool:
    return _element_type(value_type).startswith(("f", "bf"))


def _element_type(value_type: str) -> str:
    if not value_type.startswith("vector<") or not value_type.endswith(">"):
        return value_type
    body = value_type[len("vector<") : -1]
    return body.rsplit("x", 1)[-1]
