# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Registry and dispatch for MLIR-to-Loom op converters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import import_module
from inspect import getdoc
from typing import Any, overload

from loom.importers.core import ImportBodyReport
from loom.importers.mlir.model import MlirConversionContext, SourceOp

Handler = Callable[[SourceOp, MlirConversionContext], bool]


@dataclass(frozen=True, slots=True)
class ConverterInfo:
    """Registered MLIR operation converter metadata."""

    op_name: str
    handler: Handler
    summary: str | None


class ConverterRegistry:
    """Maps MLIR operation names to conversion handlers."""

    def __init__(self) -> None:
        self._converters: dict[str, ConverterInfo] = {}

    @overload
    def register(
        self,
        op_name: str,
        handler: Handler,
        *,
        summary: str | None = None,
    ) -> Handler: ...

    @overload
    def register(
        self,
        op_name: str,
        handler: None = None,
        *,
        summary: str | None = None,
    ) -> Callable[[Handler], Handler]: ...

    def register(
        self,
        op_name: str,
        handler: Handler | None = None,
        *,
        summary: str | None = None,
    ) -> Handler | Callable[[Handler], Handler]:
        def bind(bound_handler: Handler) -> Handler:
            if op_name in self._converters:
                raise ValueError(f"converter already registered for {op_name}")
            self._converters[op_name] = ConverterInfo(
                op_name=op_name,
                handler=bound_handler,
                summary=summary or converter_summary(bound_handler),
            )
            return bound_handler

        if handler is None:
            return bind
        return bind(handler)

    def converter(self, op_name: str) -> ConverterInfo | None:
        return self._converters.get(op_name)

    def registered_converters(self) -> tuple[ConverterInfo, ...]:
        return tuple(self._converters[name] for name in sorted(self._converters))

    def convert(self, op: SourceOp, context: MlirConversionContext) -> None:
        converter = self._converters.get(op.op_name)
        if converter is None:
            context.record_unsupported_op(op)
            return
        with context.source_location(op):
            if not converter.handler(op, context):
                context.record_unsupported_op(op)


def converter_summary(handler: Handler) -> str | None:
    doc = getdoc(handler)
    if doc is None:
        return None
    return doc.splitlines()[0]


def build_default_registry() -> ConverterRegistry:
    import loom.importers.mlir.convert_affine as convert_affine
    import loom.importers.mlir.convert_amdgpu as convert_amdgpu
    import loom.importers.mlir.convert_arith as convert_arith
    import loom.importers.mlir.convert_func as convert_func
    import loom.importers.mlir.convert_gpu as convert_gpu
    import loom.importers.mlir.convert_hal as convert_hal
    import loom.importers.mlir.convert_iree_codegen as convert_iree_codegen
    import loom.importers.mlir.convert_memref as convert_memref
    import loom.importers.mlir.convert_scf as convert_scf
    import loom.importers.mlir.convert_vector as convert_vector

    registry = ConverterRegistry()
    convert_affine.register(registry)
    convert_amdgpu.register(registry)
    convert_arith.register(registry)
    convert_func.register(registry)
    convert_gpu.register(registry)
    convert_hal.register(registry)
    convert_iree_codegen.register(registry)
    convert_memref.register(registry)
    convert_scf.register(registry)
    convert_vector.register(registry)
    return registry


def convert_function_body(
    function_operation: Any,
    context: MlirConversionContext,
    registry: ConverterRegistry | None = None,
) -> ImportBodyReport:
    registry = registry or build_default_registry()
    context.registry = registry
    convert_immediate_region_ops(function_operation, context)
    return context.finish()


def convert_immediate_region_ops(
    operation: Any,
    context: MlirConversionContext,
) -> None:
    registry = context.registry
    if not isinstance(registry, ConverterRegistry):
        raise ValueError("conversion context has no registry")
    for op in immediate_region_ops(operation, depth=1):
        registry.convert(op, context)


def immediate_region_ops(
    operation: Any,
    depth: int,
    *,
    region: Any | None = None,
) -> Iterator[SourceOp]:
    """Yields direct child ops using generic MLIR operations, not OpViews."""
    parent = generic_operation(operation)
    for child in walk_operations(parent):
        if child == parent or child.parent != parent:
            continue
        if region is not None and child.block.region != region:
            continue
        yield SourceOp(child, depth)


def walk_operations(operation: Any) -> Iterator[Any]:
    """Walks generic MLIR operations without constructing dialect OpViews."""
    root = generic_operation(operation)
    operations: list[Any] = []
    failure: BaseException | None = None

    # Keep the optional IREE binding import at traversal time so importing this
    # Python package does not require MLIR.
    ir = import_module("iree.compiler.ir")

    def callback(walked_operation: Any) -> Any:
        nonlocal failure
        try:
            operations.append(walked_operation)
        except BaseException as exc:
            failure = exc
            return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    root.walk(callback, ir.WalkOrder.PRE_ORDER)
    if failure is not None:
        raise failure
    yield from operations


def generic_operation(value: Any) -> Any:
    return getattr(value, "operation", value)
