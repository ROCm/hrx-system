# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR scf structural ops."""

from __future__ import annotations

from typing import Any

from loom.builder import ValueRef
from loom.importers.mlir.converter import ConverterRegistry, immediate_region_ops
from loom.importers.mlir.model import MlirConversionContext, SourceOp
from loom.ir import Region


def register(registry: ConverterRegistry) -> None:
    registry.register("scf.for", convert_for)
    registry.register("scf.forall", convert_forall)
    registry.register("scf.forall.in_parallel", convert_forall_in_parallel)
    registry.register("scf.if", convert_if)
    registry.register("scf.yield", convert_yield)


def convert_for(op: SourceOp, context: MlirConversionContext) -> bool:
    if _has_unmapped_dialect_type(op.text):
        context.record_blocked(
            op.text,
            "scf.for with dialect-specific iter/result types needs type support",
        )
        return True
    mapped_bounds, missing_bounds = context.mapped_operands(
        (op.operand(0), op.operand(1), op.operand(2))
    )
    init_args = tuple(op.operands[3:])
    mapped_init_args, missing_init_args = context.mapped_operands(init_args)
    missing = missing_bounds + missing_init_args
    if mapped_bounds is None or mapped_init_args is None:
        context.record_blocked(
            op.text, f"missing scf.for operands: {', '.join(missing)}"
        )
        return True

    source_region = op.operation.regions[0]
    block = next(iter(source_region.blocks))
    block_args = tuple(block.arguments)
    result_types = tuple(str(result.type) for result in op.results)
    loop_body = context.builder.region(
        args=[
            (
                context.source_name(block_arg).removeprefix("%"),
                context.type(str(block_arg.type)),
            )
            for block_arg in block_args
        ]
    )
    child = context.fork(preview_block=loop_body.blocks[0])
    child.registry = context.registry
    for block_arg, loom_arg_id in zip(
        block_args, loop_body.blocks[0].arg_ids, strict=True
    ):
        child.map_value(
            block_arg,
            ValueRef(loom_arg_id, context.builder.ir),
            str(block_arg.type),
        )

    registry = context.registry
    if not isinstance(registry, ConverterRegistry):
        raise ValueError("conversion context has no registry")
    with context.builder.insertion_block(loop_body.blocks[0]):
        for child_op in immediate_region_ops(
            op.operation,
            1,
            region=source_region,
        ):
            registry.convert(child_op, child)

    candidate_lines = ["scf.for region body preview:"]
    candidate_lines.extend(child.preview_lines())
    if child.blocked:
        candidate_lines.append("  // blockers:")
        candidate_lines.extend(
            f"  //   {record.source} -> {record.target[0]}" for record in child.blocked
        )
    context.record_region_candidate(op.text, candidate_lines)

    if child.blocked:
        context.merge_child_records(child)
        context.record_blocked(
            op.text,
            (
                "scf.for region import blocked by "
                f"{len(child.blocked)} nested op(s); see structured region candidate"
            ),
        )
        return True

    context.merge_child_records(child)
    loop_results = context.builder.scf.for_(
        lower_bound=mapped_bounds[0],
        upper_bound=mapped_bounds[1],
        step=mapped_bounds[2],
        iter_args=list(mapped_init_args),
        results=[context.type(result_type) for result_type in result_types],
        body=loop_body,
        names=[context.result_name(result) for result in op.results]
        if op.results
        else None,
    )
    if op.results:
        for result, loom_result, result_type in zip(
            op.results,
            loop_results,
            result_types,
            strict=True,
        ):
            context.map_result(result, loom_result, result_type)
    context.record_converted(op.text, "scf.for region")
    return True


def convert_forall(op: SourceOp, context: MlirConversionContext) -> bool:
    axes = _iree_workgroup_mapping_axes(op.attr("mapping"))
    if not axes:
        context.record_blocked(
            op.text,
            "scf.forall import currently requires IREE workgroup mappings",
        )
        return True
    if op.results:
        context.record_blocked(
            op.text,
            "scf.forall with shared output results needs tensor-level import",
        )
        return True
    source_region = op.operation.regions[0]
    blocks = tuple(source_region.blocks)
    if len(blocks) != 1:
        context.record_blocked(op.text, "scf.forall expected one body block")
        return True
    block_args = tuple(blocks[0].arguments)
    if len(block_args) != len(axes):
        context.record_blocked(
            op.text,
            "scf.forall workgroup mapping count does not match block arguments",
        )
        return True

    child = context.fork(preview_block=context.preview_block)
    child.registry = context.registry
    for block_arg, axis in zip(block_args, axes, strict=True):
        workgroup_id = context.builder.kernel.workgroup_id(
            dimension=axis,
            results=[context.type("index")],
            name=child.fresh_name(f"wg{axis}"),
        )
        child.map_value(block_arg, workgroup_id, "index")

    registry = context.registry
    if not isinstance(registry, ConverterRegistry):
        raise ValueError("conversion context has no registry")
    for child_op in immediate_region_ops(
        op.operation,
        1,
        region=source_region,
    ):
        registry.convert(child_op, child)

    context.merge_child_records(child)
    if child.blocked:
        context.record_blocked(
            op.text,
            (
                "scf.forall workgroup region import blocked by "
                f"{len(child.blocked)} nested op(s)"
            ),
        )
        return True
    context.record_converted(
        op.text,
        "scf.forall workgroup mapping inlined "
        f"({', '.join(f'kernel.workgroup.id<{axis}>' for axis in axes)})",
    )
    return True


def convert_if(op: SourceOp, context: MlirConversionContext) -> bool:
    if _has_unmapped_dialect_type(op.text):
        context.record_blocked(
            op.text,
            "scf.if with dialect-specific result types needs type support",
        )
        return True
    mapped_condition = context.mapped(op.operand())
    if mapped_condition is None:
        context.record_blocked(
            op.text,
            f"missing scf.if condition operand: {op.operand().get_name()}",
        )
        return True

    source_regions = tuple(op.operation.regions)
    result_types = tuple(str(result.type) for result in op.results)
    then_region = _convert_single_block_region(
        op.operation,
        source_regions[0],
        context,
        "then",
    )
    if len(source_regions) > 1 and tuple(source_regions[1].blocks):
        else_region = _convert_single_block_region(
            op.operation,
            source_regions[1],
            context,
            "else",
        )
    else:
        else_region = context.builder.region()
        with context.builder.insertion_block(else_region.blocks[0]):
            context.builder.scf.yield_()

    if then_region is None or else_region is None:
        context.record_blocked(op.text, "scf.if region import blocked by nested op(s)")
        return True

    results = context.builder.scf.if_(
        condition=mapped_condition,
        results=[context.type(result_type) for result_type in result_types],
        then_region=then_region,
        else_region=else_region,
        names=[context.result_name(result) for result in op.results]
        if op.results
        else None,
    )
    if op.results:
        for source_result, loom_result, result_type in zip(
            op.results,
            results,
            result_types,
            strict=True,
        ):
            context.map_result(source_result, loom_result, result_type)
    context.record_converted(op.text, "scf.if region")
    return True


def convert_forall_in_parallel(op: SourceOp, context: MlirConversionContext) -> bool:
    context.record_converted(op.text, "scf.forall.in_parallel")
    return True


def convert_yield(op: SourceOp, context: MlirConversionContext) -> bool:
    if not op.operands:
        context.builder.scf.yield_()
        context.record_converted(op.text, "scf.yield")
        return True
    mapped, missing = context.mapped_operands(op.operands)
    if mapped is None:
        context.record_blocked(
            op.text, f"missing scf.yield operands: {', '.join(missing)}"
        )
        return True
    context.builder.scf.yield_(values=list(mapped))
    context.record_converted(op.text, "scf.yield")
    return True


def _convert_single_block_region(
    owner_operation: Any,
    source_region: Any,
    context: MlirConversionContext,
    label: str,
) -> Region | None:
    blocks = tuple(source_region.blocks)
    if len(blocks) != 1:
        context.record_blocked(f"scf.if {label} region", "expected one block")
        return None
    region = context.builder.region()
    child = context.fork(preview_block=region.blocks[0])
    child.registry = context.registry
    registry = context.registry
    if not isinstance(registry, ConverterRegistry):
        raise ValueError("conversion context has no registry")
    with context.builder.insertion_block(region.blocks[0]):
        last_op_name = None
        for child_op in immediate_region_ops(
            owner_operation,
            1,
            region=source_region,
        ):
            last_op_name = child_op.op_name
            registry.convert(child_op, child)
        if last_op_name != "scf.yield":
            context.builder.scf.yield_()
            child.record_converted(f"scf.if {label} implicit yield", "scf.yield")
    context.merge_child_records(child)
    if child.blocked:
        return None
    return region


def _has_unmapped_dialect_type(text: str) -> bool:
    return "!" in text


def _iree_workgroup_mapping_axes(attr: object | None) -> tuple[str, ...]:
    if attr is None:
        return ()
    text = str(attr).strip()
    if not text.startswith("[") or not text.endswith("]"):
        return ()
    body = text[1:-1].strip()
    if not body:
        return ()
    axes: list[str] = []
    prefix = "#iree_codegen.workgroup_mapping<"
    for piece in body.split(","):
        mapping = piece.strip()
        if not mapping.startswith(prefix) or not mapping.endswith(">"):
            return ()
        axis = mapping[len(prefix) : -1]
        if axis not in {"x", "y", "z"}:
            return ()
        axes.append(axis)
    return tuple(axes)
