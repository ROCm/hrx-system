# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR AMDGPU ops at the kernel import boundary."""

from __future__ import annotations

from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp


def register(registry: ConverterRegistry) -> None:
    registry.register("amdgpu.fat_raw_buffer_cast", convert_fat_raw_buffer_cast)
    registry.register("amdgpu.gather_to_lds", convert_gather_to_lds)
    registry.register("amdgpu.lds_barrier", convert_lds_barrier)
    registry.register("amdgpu.memory_counter_wait", convert_memory_counter_wait)


def convert_fat_raw_buffer_cast(
    op: SourceOp,
    context: MlirConversionContext,
) -> bool:
    """Reuses the source view when AMDGPU raw-buffer metadata preserves layout."""

    mapped = context.mapped(op.operand())
    if mapped is None:
        context.record_blocked(
            op.text,
            f"missing amdgpu.fat_raw_buffer_cast source: {op.operand().get_name()}",
        )
        return True
    operand_segments = context.attr_decoder.dense_i32_array(
        op.attr("operandSegmentSizes")
    )
    if _has_optional_operand(operand_segments, 1):
        context.record_blocked(
            op.text,
            "amdgpu.fat_raw_buffer_cast valid bytes needs view bounds facts",
        )
        return True
    if _has_optional_operand(operand_segments, 2):
        context.record_blocked(
            op.text,
            "amdgpu.fat_raw_buffer_cast cache swizzle stride needs "
            "access encoding facts",
        )
        return True
    result_type = context.type_converter.memref_to_view_type_text(op.result_type())
    context.map_result(op.result(), mapped, result_type)
    context.record_converted(
        op.text,
        f"{context.ssa(mapped)} reused for amdgpu.fat_raw_buffer_cast",
    )
    return True


def _has_optional_operand(operand_segments: tuple[int, ...], index: int) -> bool:
    return len(operand_segments) > index and operand_segments[index] != 0


def convert_lds_barrier(op: SourceOp, context: MlirConversionContext) -> bool:
    """Maps AMDGPU LDS barriers to workgroup-scoped kernel barriers."""

    context.builder.kernel.barrier(
        memory_space="workgroup",
        ordering="acq_rel",
        scope="workgroup",
    )
    context.record_converted(op.text, "kernel.barrier")
    return True


def convert_gather_to_lds(op: SourceOp, context: MlirConversionContext) -> bool:
    """Blocks until Loom has an async gather/copy kernel representation."""

    context.record_blocked(
        op.text,
        "amdgpu.gather_to_lds needs kernel async gather/copy import",
    )
    return True


def convert_memory_counter_wait(op: SourceOp, context: MlirConversionContext) -> bool:
    """Blocks until Loom has an async wait/counter kernel representation."""

    context.record_blocked(
        op.text,
        "amdgpu.memory_counter_wait needs kernel async wait import",
    )
    return True
