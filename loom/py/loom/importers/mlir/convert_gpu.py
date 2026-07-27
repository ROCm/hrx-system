# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR gpu ops at the kernel import boundary."""

from __future__ import annotations

from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp


def register(registry: ConverterRegistry) -> None:
    registry.register("gpu.barrier", convert_barrier)
    registry.register("gpu.thread_id", convert_thread_id)
    registry.register("gpu.shuffle", convert_shuffle)
    registry.register("gpu.subgroup_reduce", convert_subgroup_reduce)


def convert_thread_id(op: SourceOp, context: MlirConversionContext) -> bool:
    axis = gpu_dimension_axis(op.attr("dimension"))
    if axis is None:
        context.record_blocked(
            op.text, "gpu.thread_id dimension attr is not a known axis"
        )
        return True
    mapped = context.mapped(op.result())
    if mapped is None:
        mapped = context.builder.kernel.workitem_id(
            dimension=axis,
            results=[context.type("index")],
            name={"x": "tidx", "y": "tidy", "z": "tidz"}[axis],
        )
        context.map_result(op.result(), mapped, "index")
    context.record_converted(
        op.text,
        f"{context.ssa(mapped)} = kernel.workitem.id<{axis}> prelude",
    )
    return True


def convert_barrier(op: SourceOp, context: MlirConversionContext) -> bool:
    context.builder.kernel.barrier(
        memory_space="workgroup",
        ordering="acq_rel",
        scope="workgroup",
    )
    context.record_converted(op.text, "kernel.barrier")
    return True


def convert_shuffle(op: SourceOp, context: MlirConversionContext) -> bool:
    mode = gpu_shuffle_mode(op.attr("mode")) or "unknown"
    if mode not in {"xor", "up", "down", "index"}:
        context.record_blocked(op.text, f"unknown gpu.shuffle mode `{mode}`")
        return True
    mapped, missing = context.mapped_operands(
        (op.operand(0), op.operand(1), op.operand(2))
    )
    if mapped is None:
        context.record_blocked(
            op.text, f"missing gpu.shuffle operands: {', '.join(missing)}"
        )
        return True
    results = context.builder.kernel.shuffle(
        mode=mode,
        value=mapped[0],
        offset=mapped[1],
        width=mapped[2],
        results=[context.type(op.result_type(0)), context.type(op.result_type(1))],
        names=[context.result_name(result) for result in op.results],
    )
    for source_result, loom_result in zip(op.results, results, strict=True):
        context.map_result(source_result, loom_result, str(source_result.type))
    context.record_converted(
        op.text,
        (
            f"{context.ssa(results[0])}, {context.ssa(results[1])} = "
            f"kernel.subgroup.shuffle<{mode}> ..."
        ),
    )
    return True


def convert_subgroup_reduce(op: SourceOp, context: MlirConversionContext) -> bool:
    mapped = context.mapped(op.operand())
    if mapped is None:
        context.record_blocked(
            op.text, f"missing gpu.subgroup_reduce operand: {op.operand().get_name()}"
        )
        return True
    kind = gpu_all_reduce_kind(op.attr("op"), op.operand_type())
    if kind is None:
        context.record_blocked(op.text, "unknown gpu.subgroup_reduce combining kind")
        return True
    result_type = op.result_type()
    result = context.builder.kernel.subgroup_reduce(
        kind=kind,
        value=mapped,
        cluster_size=context.attr_decoder.integer(op.attr("cluster_size")),
        cluster_stride=context.attr_decoder.integer(op.attr("cluster_stride")),
        results=[context.type(result_type)],
        name=context.result_name(op.result()),
    )
    context.map_result(op.result(), result, result_type)
    context.record_converted(
        op.text,
        f"{context.ssa(result)} = kernel.subgroup.reduce<{kind}> ...",
    )
    return True


def gpu_dimension_axis(attr: object) -> str | None:
    text = str(attr)
    prefix = "#gpu<dim "
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    axis = text[len(prefix) : -1]
    return axis if axis in {"x", "y", "z"} else None


def gpu_shuffle_mode(attr: object) -> str | None:
    text = str(attr)
    prefix = "#gpu<shuffle_mode "
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    return text[len(prefix) : -1]


def gpu_all_reduce_kind(attr: object | None, value_type: str) -> str | None:
    text = str(attr)
    prefix = "#gpu<all_reduce_op "
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    kind = text[len(prefix) : -1]
    if kind == "add":
        return "addf" if _is_float_or_float_vector(value_type) else "addi"
    if kind == "mul":
        return "mulf" if _is_float_or_float_vector(value_type) else "muli"
    if kind == "and":
        return "andi"
    if kind == "or":
        return "ori"
    if kind == "xor":
        return "xori"
    if kind in {
        "minsi",
        "maxsi",
        "minui",
        "maxui",
        "minimumf",
        "maximumf",
        "minnumf",
        "maxnumf",
    }:
        return kind
    return None


def _is_float_or_float_vector(value_type: str) -> bool:
    if value_type.startswith("vector<"):
        return value_type.rstrip(">").split("x")[-1].startswith(("f", "bf"))
    return value_type.startswith(("f", "bf"))
