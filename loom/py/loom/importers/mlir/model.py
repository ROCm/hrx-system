# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Data model for the MLIR-to-Loom importer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import loom
from loom.builder import ValueRef
from loom.diagnostics import DiagnosticEngine
from loom.importers.core import (
    ImportBodyReport,
    SourceImportSession,
)
from loom.importers.core import (
    source_key as core_source_key,
)
from loom.importers.mlir.attrs import MlirAttributeDecoder
from loom.importers.mlir.locations import MlirLocationConverter
from loom.importers.mlir.types import MlirTypeConverter
from loom.ir import Type


@dataclass(frozen=True, slots=True)
class SourceOp:
    """MLIR operation plus region-depth context for conversion."""

    source: Any
    depth: int

    @property
    def operation(self) -> Any:
        return getattr(self.source, "operation", self.source)

    @property
    def op_name(self) -> str:
        return str(self.operation.name)

    @property
    def text(self) -> str:
        get_asm = getattr(self.operation, "get_asm", None)
        if get_asm is not None:
            return (
                str(get_asm(enable_debug_info=False, use_local_scope=True))
                .splitlines()[0]
                .strip()
            )
        return str(self.operation).splitlines()[0].strip()

    @property
    def operands(self) -> tuple[Any, ...]:
        return tuple(self.operation.operands)

    @property
    def results(self) -> tuple[Any, ...]:
        return tuple(self.operation.results)

    def attr(self, name: str) -> Any:
        return self.operation.attributes.get(name)

    def operand(self, index: int = 0) -> Any:
        return self.operands[index]

    def result(self, index: int = 0) -> Any:
        return self.results[index]

    def operand_type(self, index: int = 0) -> str:
        return str(self.operand(index).type)

    def result_type(self, index: int = 0) -> str:
        return str(self.result(index).type)


def mlir_source_key(source: object) -> object:
    """Returns a stable key for MLIR value wrappers across API object copies."""

    get_name = getattr(source, "get_name", None)
    if callable(get_name) and hasattr(source, "type"):
        return ("mlir_value", str(get_name()))
    return core_source_key(source)


@dataclass(frozen=True, slots=True)
class Binding:
    """One HAL interface binding converted to a Loom kernel argument."""

    result: str
    binding: int
    alignment: str | None
    offset: str | None
    flags: str | None
    source_type: str
    view_type: str


@dataclass(frozen=True, slots=True)
class KernelFacts:
    """Facts extracted from one HAL executable kernel import."""

    executable_name: str | None
    variant_name: str | None
    target_driver: str | None
    target_format: str | None
    export_name: str | None
    function_name: str
    workgroup_size: tuple[int, int, int] | None
    subgroup_size: int | None
    bindings: tuple[Binding, ...]
    operation_counts: Counter[str]
    converted_body: ImportBodyReport


@dataclass(slots=True)
class MlirConversionContext(SourceImportSession):
    """MLIR-specialized import session using Loom dynamic builders."""

    type_converter: MlirTypeConverter = field(default_factory=MlirTypeConverter)
    attr_decoder: MlirAttributeDecoder = field(default_factory=MlirAttributeDecoder)
    location_converter: MlirLocationConverter | None = None
    source_names: dict[object, str] = field(default_factory=dict)
    binding_args: dict[int, ValueRef] = field(default_factory=dict)
    bindings_by_result: dict[str, Binding] = field(default_factory=dict)

    @classmethod
    def with_prelude(
        cls,
        builder: loom.LoomBuilder,
        values: dict[object, tuple[ValueRef, str | None]],
        *,
        diagnostics: DiagnosticEngine | None = None,
        preview_block: object | None = None,
        type_converter: MlirTypeConverter | None = None,
        source_names: dict[object, str] | None = None,
        binding_args: dict[int, ValueRef] | None = None,
        bindings_by_result: dict[str, Binding] | None = None,
    ) -> MlirConversionContext:
        context = cls(
            builder=builder,
            diagnostics=diagnostics or DiagnosticEngine(),
            preview_block=preview_block,
            type_converter=type_converter or MlirTypeConverter(),
            location_converter=MlirLocationConverter(builder.module),
            source_names={
                context_key: name
                for source, name in (source_names or {}).items()
                for context_key in (mlir_source_key(source),)
            },
            binding_args=binding_args or {},
            bindings_by_result=bindings_by_result or {},
        )
        context.capture_existing_value_names()
        for source, (ref, value_type) in values.items():
            context.map_value(source, ref, value_type)
        return context

    def type(self, value_type: str) -> Type:
        return self.type_converter.map_text(value_type)

    def source_key(self, source: object) -> object:
        return mlir_source_key(source)

    def source_name(self, source: object) -> str:
        return self.source_names.get(self.source_key(source)) or super(
            MlirConversionContext, self
        ).source_name(source)

    def result_name(self, source: object, fallback: str | None = None) -> str:
        existing = self.mapped(source)
        if existing is not None and existing.name:
            return existing.name
        source_name = self.source_names.get(self.source_key(source))
        if source_name is not None:
            return self.names.fresh(source_name)
        return super(MlirConversionContext, self).result_name(source, fallback)

    def build_constant(self, value: Any, value_type: str, name: str) -> ValueRef:
        result_type = self.type(value_type)
        constant_value = self.type_converter.coerce_constant_value(value, value_type)
        if value_type in ("index", "offset"):
            return self.builder.index.constant(
                value=constant_value,
                results=[result_type],
                name=name,
            )
        return self.builder.scalar.constant(
            value=constant_value,
            results=[result_type],
            name=name,
        )

    def require_top_level(self, op: SourceOp, target: str) -> bool:
        if op.depth == 1:
            return True
        self.record_blocked(
            op.text, f"{target} after enclosing region conversion exists"
        )
        return False

    def record_unsupported_op(self, op: SourceOp) -> None:
        self.record_unsupported(op.op_name, op.text)

    @contextmanager
    def source_location(self, op: SourceOp) -> Iterator[None]:
        if self.location_converter is None:
            yield
            return
        location_id = self.location_converter.opaque(
            getattr(op.operation, "location", None)
        )
        with self.builder.location(location_id):
            yield

    def fork(self, *, preview_block: object | None = None) -> MlirConversionContext:
        child = MlirConversionContext(
            builder=self.builder,
            diagnostics=self.diagnostics,
            preview_block=preview_block,
            value_map=dict(self.value_map),
            value_types=dict(self.value_types),
            constants=dict(self.constants),
            registry=self.registry,
            names=self.names,
            type_converter=self.type_converter,
            attr_decoder=self.attr_decoder,
            location_converter=self.location_converter,
            source_names=self.source_names,
            binding_args=self.binding_args,
            bindings_by_result=self.bindings_by_result,
        )
        return child
