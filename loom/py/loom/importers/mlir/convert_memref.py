# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR memref ops at the kernel import boundary."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp
from loom.ir import BUFFER_TYPE


def register(registry: ConverterRegistry) -> None:
    registry.register("memref.alloc", convert_alloc)
    registry.register("memref.load", convert_load)


def convert_alloc(op: SourceOp, context: MlirConversionContext) -> bool:
    if op.operands:
        context.record_blocked(op.text, "dynamic memref.alloc sizes are not imported")
        return True
    result_type = op.result_type()
    byte_length = _static_memref_byte_length(result_type)
    if byte_length is None:
        context.record_blocked(
            op.text, f"cannot compute static byte length for `{result_type}`"
        )
        return True
    alignment_attr = op.attr("alignment")
    alignment = (
        context.attr_decoder.integer(alignment_attr)
        if alignment_attr is not None
        else None
    )
    view_type = context.type_converter.memref_to_view_type_text(result_type)
    result_name = context.result_name(op.result(), "alloc")
    buffer = context.builder.buffer.alloca(
        byte_length=context.ensure_constant(
            str(byte_length),
            "offset",
            f"{result_name}_bytes",
        ),
        base_alignment=alignment or _element_alignment(result_type),
        memory_space=_memory_space(result_type),
        results=[BUFFER_TYPE],
        name=context.fresh_name(f"{result_name}_buffer"),
    )
    view = context.builder.buffer.view(
        buffer=buffer,
        byte_offset=context.ensure_constant("0", "offset", "zero"),
        results=[context.type(view_type)],
        name=result_name,
    )
    context.map_result(op.result(), view, view_type)
    context.record_converted(
        op.text,
        (
            f"{context.ssa(buffer)} = buffer.alloca ...",
            f"{context.ssa(view)} = buffer.view ...",
        ),
    )
    return True


def convert_load(op: SourceOp, context: MlirConversionContext) -> bool:
    view_value = op.operand()
    view = context.mapped(view_value)
    indices, missing = _mapped_values(op.operands[1:], context)
    if view is None:
        missing = (view_value.get_name(), *missing)
    if missing or view is None:
        context.record_blocked(
            op.text, f"missing memref.load operands: {', '.join(missing)}"
        )
        return True
    result_type = op.result_type()
    if result_type.startswith("vector<"):
        result = context.builder.vector.load(
            view=view,
            indices=list(indices),
            results=[context.type(result_type)],
            name=context.result_name(op.result()),
        )
        target = "vector.load"
    else:
        result = context.builder.view.load(
            view=view,
            indices=list(indices),
            results=[context.type(result_type)],
            name=context.result_name(op.result()),
        )
        target = "view.load"
    context.map_result(op.result(), result, result_type)
    context.record_converted(op.text, f"{context.ssa(result)} = {target} ...")
    return True


def _mapped_values(
    values: tuple[object, ...],
    context: MlirConversionContext,
) -> tuple[tuple[ValueRef, ...], tuple[str, ...]]:
    mapped: list[ValueRef] = []
    missing: list[str] = []
    for value in values:
        ref = context.mapped(value)
        if ref is None:
            missing.append(context.source_name(value))
        else:
            mapped.append(ref)
    return tuple(mapped), tuple(missing)


def _static_memref_byte_length(memref_type: str) -> int | None:
    shape_and_element = _memref_shape_and_element(memref_type)
    if shape_and_element is None:
        return None
    pieces = _split_top_level_x(shape_and_element)
    if len(pieces) < 2 or any(piece == "?" for piece in pieces[:-1]):
        return None
    element = pieces[-1]
    element_count = 1
    if element.startswith("vector<") and element.endswith(">"):
        nested = _split_top_level_x(element[7:-1])
        if len(nested) < 2 or any(piece == "?" for piece in nested[:-1]):
            return None
        for dim in nested[:-1]:
            element_count *= int(dim)
        element = nested[-1]
    for dim in pieces[:-1]:
        element_count *= int(dim)
    element_size = _ELEMENT_BYTE_SIZES.get(element)
    return None if element_size is None else element_count * element_size


def _element_alignment(memref_type: str) -> int:
    shape_and_element = _memref_shape_and_element(memref_type)
    if shape_and_element is None:
        return 1
    element = _split_top_level_x(shape_and_element)[-1]
    if element.startswith("vector<") and element.endswith(">"):
        element = _split_top_level_x(element[7:-1])[-1]
    return _ELEMENT_BYTE_SIZES.get(element, 1)


def _memory_space(memref_type: str) -> str:
    if "#gpu.address_space<workgroup>" in memref_type or memref_type.endswith(", 3>"):
        return "workgroup"
    return "private"


def _memref_shape_and_element(memref_type: str) -> str | None:
    if not memref_type.startswith("memref<") or not memref_type.endswith(">"):
        return None
    return _split_top_level_comma(memref_type[7:-1])[0].strip()


def _split_top_level_comma(text: str) -> tuple[str, ...]:
    return _split_top_level(text, ",")


def _split_top_level_x(text: str) -> tuple[str, ...]:
    return _split_top_level(text, "x")


def _split_top_level(text: str, separator: str) -> tuple[str, ...]:
    pieces: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == separator and depth == 0:
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    return tuple(pieces)


_ELEMENT_BYTE_SIZES = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "index": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}
