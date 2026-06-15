# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared state for importers that build Loom modules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import loom
from loom.builder import ValueRef
from loom.diagnostics import DiagnosticEngine
from loom.format.text.printer import Printer
from loom.importers.core.names import NameAllocator, source_name
from loom.ir import Module, Type
from loom.verify import verify_module


@dataclass(frozen=True, slots=True)
class ConversionRecord:
    """A source-to-target conversion or blocker record."""

    source: str
    target: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImportBodyReport:
    """Summary of the body slice handled by an importer."""

    lines: tuple[str, ...]
    converted: tuple[ConversionRecord, ...]
    blocked: tuple[ConversionRecord, ...]
    region_candidates: tuple[ConversionRecord, ...]
    unsupported_counts: Counter[str]


@dataclass(frozen=True, slots=True)
class _IdentitySourceKey:
    """Identity key for foreign IR nodes."""

    source: object
    identity: int

    def __hash__(self) -> int:
        return self.identity

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdentitySourceKey) and self.source is other.source


_SIMPLE_SOURCE_KEY_TYPES = (str, int, float, bool, bytes, type(None))


def source_key(source: object) -> object:
    """Returns a stable dictionary key for a foreign source object."""
    if isinstance(source, _SIMPLE_SOURCE_KEY_TYPES):
        return source
    return _IdentitySourceKey(source=source, identity=id(source))


@dataclass(slots=True)
class SourceImportSession:
    """Owns mutable state for one whole-module import attempt."""

    builder: loom.LoomBuilder
    diagnostics: DiagnosticEngine = field(default_factory=DiagnosticEngine)
    preview_block: object | None = None
    value_map: dict[object, ValueRef] = field(default_factory=dict)
    value_types: dict[object, str] = field(default_factory=dict)
    constants: dict[tuple[str, str], ValueRef] = field(default_factory=dict)
    converted: list[ConversionRecord] = field(default_factory=list)
    blocked: list[ConversionRecord] = field(default_factory=list)
    region_candidates: list[ConversionRecord] = field(default_factory=list)
    unsupported_counts: Counter[str] = field(default_factory=Counter)
    registry: Any | None = None
    names: NameAllocator = field(default_factory=NameAllocator)

    @classmethod
    def create(
        cls,
        builder: loom.LoomBuilder,
        values: Mapping[object, tuple[ValueRef, str | None]] | None = None,
        *,
        diagnostics: DiagnosticEngine | None = None,
        preview_block: object | None = None,
    ) -> SourceImportSession:
        session = cls(
            builder=builder,
            diagnostics=diagnostics or DiagnosticEngine(),
            preview_block=preview_block,
        )
        session.capture_existing_value_names()
        for source, (ref, value_type) in (values or {}).items():
            session.map_value(source, ref, value_type)
        return session

    @property
    def module(self) -> Module:
        return self.builder.module

    def capture_existing_value_names(self) -> None:
        for value in self.builder.module.values:
            if value.name:
                self.names.capture(value.name)

    def source_key(self, source: object) -> object:
        """Returns a stable dictionary key for a foreign source object."""
        return source_key(source)

    def map_value(
        self,
        source: object,
        ref: ValueRef,
        value_type: str | None = None,
    ) -> None:
        key = self.source_key(source)
        self.value_map[key] = ref
        if ref.name:
            self.names.capture(ref.name)
        self.value_types[key] = value_type if value_type is not None else str(ref.type)

    def map_result(
        self,
        source: object,
        ref: ValueRef,
        value_type: str | None = None,
    ) -> None:
        self.map_value(source, ref, value_type)

    def mapped(self, source: object) -> ValueRef | None:
        return self.value_map.get(self.source_key(source))

    def mapped_value_type(self, source: object) -> str | None:
        return self.value_types.get(self.source_key(source))

    def mapped_operands(
        self,
        operands: tuple[object, ...],
    ) -> tuple[tuple[ValueRef, ...] | None, tuple[str, ...]]:
        mapped: list[ValueRef] = []
        missing: list[str] = []
        for operand in operands:
            ref = self.mapped(operand)
            if ref is None:
                missing.append(self.source_name(operand))
            else:
                mapped.append(ref)
        return (tuple(mapped) if not missing else None, tuple(missing))

    def source_name(self, source: object) -> str:
        return source_name(source)

    def result_name(self, source: object, fallback: str | None = None) -> str:
        existing = self.value_map.get(self.source_key(source))
        if existing is not None and existing.name:
            return existing.name
        return self.names.fresh(
            self.source_name(source) if fallback is None else fallback
        )

    def reserve_name(self, name: str) -> str:
        return self.names.reserve_or_fresh(name)

    def fresh_name(self, base: str) -> str:
        return self.names.fresh(base)

    def ssa(self, ref: ValueRef) -> str:
        return f"%{ref.name}" if ref.name else f"%{ref.id}"

    def type(self, value_type: str) -> Type:
        raise NotImplementedError("import session must provide type conversion")

    def result_type(self, source: Any) -> Type:
        return self.type(str(source.type))

    def build_constant(self, value: Any, value_type: str, name: str) -> ValueRef:
        raise NotImplementedError("import session must provide constant conversion")

    def build_binary(
        self,
        target_op: str,
        lhs: ValueRef,
        rhs: ValueRef,
        result_type: str,
        name: str,
    ) -> ValueRef:
        dialect_name, op_name = target_op.split(".", 1)
        dialect = getattr(self.builder, dialect_name)
        op_builder = getattr(dialect, op_name)
        return cast(
            ValueRef,
            op_builder(
                lhs=lhs,
                rhs=rhs,
                results=[self.type(result_type)],
                name=name,
            ),
        )

    def record_converted(
        self,
        source: str,
        target: list[str] | tuple[str, ...] | str,
    ) -> None:
        targets = (target,) if isinstance(target, str) else tuple(target)
        self.converted.append(ConversionRecord(source, targets))

    def record_blocked(self, source: str, reason: str) -> None:
        self.blocked.append(ConversionRecord(source, (reason,)))
        self.diagnostics.unsupported(source, reason)

    def record_region_candidate(
        self,
        source: str,
        lines: list[str] | tuple[str, ...],
    ) -> None:
        self.region_candidates.append(ConversionRecord(source, tuple(lines)))

    def record_unsupported(self, op_name: str, source: str) -> None:
        self.unsupported_counts[op_name] += 1
        self.record_blocked(source, f"no registered converter for `{op_name}`")

    def ensure_constant(self, value: str, value_type: str, base: str) -> ValueRef:
        key = (value_type, value)
        existing = self.constants.get(key)
        if existing is not None:
            return existing
        result = self.build_constant(value, value_type, self.reserve_name(base))
        self.constants[key] = result
        return result

    def remember_constant(
        self,
        value: str,
        value_type: str,
        result: ValueRef,
    ) -> None:
        self.constants.setdefault((value_type, value), result)

    def fork(self, *, preview_block: object | None = None) -> SourceImportSession:
        child = type(self)(
            builder=self.builder,
            diagnostics=self.diagnostics,
            preview_block=preview_block,
            value_map=dict(self.value_map),
            value_types=dict(self.value_types),
            constants=dict(self.constants),
            registry=self.registry,
            names=self.names,
        )
        return child

    def merge_child_records(self, child: SourceImportSession) -> None:
        self.converted.extend(child.converted)
        self.blocked.extend(child.blocked)
        self.region_candidates.extend(child.region_candidates)
        self.unsupported_counts.update(child.unsupported_counts)

    def finish(self) -> ImportBodyReport:
        return ImportBodyReport(
            lines=tuple(self.preview_lines()),
            converted=tuple(self.converted),
            blocked=tuple(self.blocked),
            region_candidates=tuple(self.region_candidates),
            unsupported_counts=Counter(self.unsupported_counts),
        )

    def verify_structure(self) -> None:
        verify_module(self.module, diagnostics=self.diagnostics)
        self.diagnostics.raise_if_errors()

    def preview_lines(self) -> list[str]:
        block = self.preview_block
        if block is None:
            return []
        ops = getattr(block, "ops", None)
        if ops is None:
            return []
        printer = Printer()
        printer.register_ops(loom.default_ops())
        printer.register_types(loom.default_types())
        lines: list[str] = []
        for op in ops:
            lines.extend(printer.print_operation(op, self.builder.module).splitlines())
        return lines
