# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract case records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loom.dsl import FACT_IDENTITY, MemoryAccessInterface, Op
from loom.target.contracts.descriptors import _require_descriptor
from loom.target.contracts.emits import DescriptorEmitForm, EmitDescriptorOp
from loom.target.contracts.guards import Guard
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.source import ValueRef
from loom.target.low_descriptors import Descriptor, DescriptorSet, OperandRole


@dataclass(frozen=True, slots=True)
class DescriptorRule:
    """Descriptor-rule contract case authored in Python."""

    source_op: Op
    descriptor: Descriptor
    guards: tuple[Guard, ...] = ()
    emit: tuple[EmitDescriptorOp, ...] = ()
    priority: int = 0
    report_key: str = ""

    def __init__(
        self,
        *,
        source_op: Op,
        descriptor: Descriptor,
        guards: Sequence[Guard] = (),
        emit: Sequence[EmitDescriptorOp] = (),
        priority: int = 0,
        report_key: str = "",
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "guards", tuple(guards))
        object.__setattr__(self, "emit", tuple(emit))
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "report_key", report_key)
        if priority < 0:
            raise ValueError("descriptor rule priority must be non-negative")
        _validate_report_key(source_op, report_key)

    @property
    def system(self) -> ContractSystem:
        return ContractSystem.DESCRIPTOR_RULE

    def validate(self, descriptor_set: DescriptorSet) -> None:
        _require_descriptor(descriptor_set, self.descriptor)
        for guard in self.guards:
            guard.validate(self.source_op)
        defined_temporaries = set[str]()
        for emit in self.emit:
            if emit.descriptor != self.descriptor:
                _require_descriptor(descriptor_set, emit.descriptor)
            produced_temporaries = emit.validate(
                self.source_op,
                descriptor_set,
                defined_temporaries,
            )
            defined_temporaries.update(produced_temporaries)
        self._validate_source_memory_index_preservation()
        self._validate_per_lane_sequence()

    def _validate_source_memory_index_preservation(self) -> None:
        source_memories = tuple(
            emit.source_memory for emit in self.emit if emit.source_memory is not None
        )
        if not source_memories:
            return
        memory_access = next(
            (
                interface
                for interface in self.source_op.interfaces
                if isinstance(interface, MemoryAccessInterface)
            ),
            None,
        )
        if memory_access is None or memory_access.indices is None:
            return
        source_index_used = any(
            value_ref.kind == SourceValueKind.OPERAND
            and value_ref.field == memory_access.indices
            for emit in self.emit
            for value_ref in emit.operands.values()
        )
        preserved_source_memory_count = sum(
            source_memory.preserve_source_index for source_memory in source_memories
        )
        if source_index_used and preserved_source_memory_count != len(source_memories):
            raise ValueError(
                f"{self.source_op.name}: source-memory rules that consume the "
                f"original '{memory_access.indices}' operand must preserve the "
                "source index"
            )
        if preserved_source_memory_count != 0 and not source_index_used:
            raise ValueError(
                f"{self.source_op.name}: source-index preservation requires the "
                f"original '{memory_access.indices}' operand"
            )

    def _validate_per_lane_sequence(self) -> None:
        sequence_emit_count = sum(
            emit.form == DescriptorEmitForm.PER_LANE_SEQUENCE for emit in self.emit
        )
        if sequence_emit_count == 0:
            return
        if sequence_emit_count != len(self.emit):
            raise ValueError(
                f"{self.source_op.name}: per-lane-sequence emit programs cannot "
                "mix emission forms"
            )
        if sequence_emit_count < 2:
            raise ValueError(
                f"{self.source_op.name}: per-lane-sequence emit programs need "
                "at least two emits"
            )
        for emit_index, emit in enumerate(self.emit):
            result_bindings = emit.results if emit.results is not None else {}
            result_refs = []
            for descriptor_operand in emit.descriptor.operands:
                if descriptor_operand.role not in (
                    OperandRole.RESULT,
                    OperandRole.OPERAND_RESULT,
                ):
                    continue
                value_ref = result_bindings.get(descriptor_operand.field_name)
                if value_ref is not None:
                    result_refs.append(value_ref)
            if len(result_refs) != 1:
                raise ValueError(
                    f"{self.source_op.name}: per-lane-sequence emit "
                    f"{emit_index} must bind exactly one result"
                )
            result_ref = result_refs[0]
            if emit_index + 1 == len(self.emit):
                if result_ref.kind != SourceValueKind.RESULT:
                    raise ValueError(
                        f"{self.source_op.name}: per-lane-sequence final emit "
                        "must bind a source result"
                    )
            elif result_ref.kind != SourceValueKind.TEMPORARY:
                raise ValueError(
                    f"{self.source_op.name}: per-lane-sequence intermediate emit "
                    "must bind a temporary"
                )
            operand_bindings = emit.operands if emit.operands is not None else {}
            for value_ref in operand_bindings.values():
                if value_ref.materializer is not None:
                    raise ValueError(
                        f"{self.source_op.name}: per-lane-sequence operands "
                        "cannot use value materializers"
                    )


@dataclass(frozen=True, slots=True)
class ValueAliasRule:
    """Contract case that aliases one source value to a source result."""

    source_op: Op
    source: ValueRef
    result: ValueRef
    guards: tuple[Guard, ...] = ()

    def __init__(
        self,
        *,
        source_op: Op,
        source: ValueRef,
        result: ValueRef,
        guards: Sequence[Guard] = (),
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "guards", tuple(guards))

    @property
    def system(self) -> ContractSystem:
        return ContractSystem.VALUE_ALIAS

    def validate(self, descriptor_set: DescriptorSet) -> None:
        del descriptor_set
        if self.source.kind != SourceValueKind.OPERAND:
            raise ValueError(f"{self.source_op.name}: alias source must be an operand")
        if self.result.kind != SourceValueKind.RESULT:
            raise ValueError(f"{self.source_op.name}: alias result must be a result")
        self.source.validate(self.source_op, "alias source")
        self.result.validate(self.source_op, "alias result")
        for guard in self.guards:
            guard.validate(self.source_op)


@dataclass(frozen=True, slots=True)
class OrdinalValueAliasRule:
    """Contract case that aliases operand/result fields by ordinal."""

    source_op: Op
    source: ValueRef
    result: ValueRef
    guards: tuple[Guard, ...] = ()

    def __init__(
        self,
        *,
        source_op: Op,
        source: ValueRef,
        result: ValueRef,
        guards: Sequence[Guard] = (),
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "guards", tuple(guards))

    @property
    def system(self) -> ContractSystem:
        return ContractSystem.VALUE_ALIAS

    def validate(self, descriptor_set: DescriptorSet) -> None:
        del descriptor_set
        if self.source.kind != SourceValueKind.OPERAND:
            raise ValueError(f"{self.source_op.name}: alias source must be an operand")
        if self.result.kind != SourceValueKind.RESULT:
            raise ValueError(f"{self.source_op.name}: alias result must be a result")
        source_operand = self.source_op.operand(self.source.field)
        if source_operand is None:
            raise ValueError(
                f"{self.source_op.name}: alias source field "
                f"'{self.source.field}' is not an operand"
            )
        if not source_operand.variadic:
            raise ValueError(
                f"{self.source_op.name}: ordinal alias source field "
                f"'{self.source.field}' must be variadic"
            )
        result_field = self.source_op.result(self.result.field)
        if result_field is None:
            raise ValueError(
                f"{self.source_op.name}: alias result field "
                f"'{self.result.field}' is not a result"
            )
        if not result_field.variadic:
            raise ValueError(
                f"{self.source_op.name}: ordinal alias result field "
                f"'{self.result.field}' must be variadic"
            )
        self.source.validate(self.source_op, "alias source")
        self.result.validate(self.source_op, "alias result")
        if FACT_IDENTITY not in self.source_op.traits:
            raise ValueError(
                f"{self.source_op.name}: ordinal alias from "
                f"'{self.source.field}' to '{self.result.field}' requires the "
                "FactIdentity trait"
            )
        for guard in self.guards:
            guard.validate(self.source_op)


@dataclass(frozen=True, slots=True)
class ValueElideRule:
    """Contract case that lowers away source results without emitted code."""

    source_op: Op
    values: tuple[ValueRef, ...]
    guards: tuple[Guard, ...] = ()

    def __init__(
        self,
        *,
        source_op: Op,
        values: Sequence[ValueRef],
        guards: Sequence[Guard] = (),
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "guards", tuple(guards))
        if not values:
            raise ValueError(f"{source_op.name}: value-elide rule needs a result")

    @property
    def system(self) -> ContractSystem:
        return ContractSystem.VALUE_ELIDE

    def validate(self, descriptor_set: DescriptorSet) -> None:
        del descriptor_set
        for value in self.values:
            if value.kind != SourceValueKind.RESULT:
                raise ValueError(
                    f"{self.source_op.name}: elided values must be results"
                )
            value.validate(self.source_op, "elided value")
        for guard in self.guards:
            guard.validate(self.source_op)


@dataclass(frozen=True, slots=True)
class RecipeRule:
    """Contract case for a bounded non-descriptor target recipe."""

    source_op: Op
    guards: tuple[Guard, ...] = ()

    def __init__(
        self,
        *,
        source_op: Op,
        guards: Sequence[Guard] = (),
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "guards", tuple(guards))

    @property
    def system(self) -> ContractSystem:
        return ContractSystem.RECIPE_RULE

    def validate(self, descriptor_set: DescriptorSet) -> None:
        del descriptor_set
        for guard in self.guards:
            guard.validate(self.source_op)


@dataclass(frozen=True, slots=True)
class DescriptorMatrixRule:
    """Contract case handled by the shared descriptor-matrix system."""

    source_op: Op
    source: str

    def __init__(
        self,
        *,
        source_op: Op,
        source: str,
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "source", source)
        if not source:
            raise ValueError(
                f"{source_op.name}: descriptor-matrix source must be non-empty"
            )

    @property
    def system(self) -> ContractSystem:
        return ContractSystem.DESCRIPTOR_MATRIX

    def validate(self, descriptor_set: DescriptorSet) -> None:
        del descriptor_set


type ContractCase = (
    DescriptorRule
    | ValueAliasRule
    | OrdinalValueAliasRule
    | ValueElideRule
    | RecipeRule
    | DescriptorMatrixRule
)


def _validate_report_key(source_op: Op, report_key: str) -> None:
    if not report_key:
        return
    if any(char.isspace() for char in report_key):
        raise ValueError(f"{source_op.name}: report key must not contain whitespace")
    if report_key != report_key.strip("."):
        raise ValueError(f"{source_op.name}: report key must not have empty segments")
    if ".." in report_key:
        raise ValueError(f"{source_op.name}: report key must not have empty segments")
