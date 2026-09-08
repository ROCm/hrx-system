# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract case records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

from loom.dsl import FACT_IDENTITY, PURE, MemoryAccessInterface, Op
from loom.target.contracts.descriptors import _require_descriptor
from loom.target.contracts.emits import (
    ContractEmit,
    DescriptorEmitForm,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterSlice,
)
from loom.target.contracts.guards import Guard
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.source import ValueRef
from loom.target.low_descriptors import Descriptor, DescriptorSet, OperandRole

MAX_SOURCE_NODES = 8
SOURCE_NODE_COUNT_BITS = (MAX_SOURCE_NODES - 1).bit_length()


@unique
class SourceNodeRelation(Enum):
    """SSA relation used to find one source op adjacent to another."""

    ADJACENT_UNIQUE_USER = "adjacent_unique_user"
    ADJACENT_DEFINITION = "adjacent_definition"


@dataclass(frozen=True, slots=True)
class SourceNode:
    """Named source op joined to the root of a descriptor rule."""

    name: str
    source_op: Op
    relation: SourceNodeRelation
    parent_value: ValueRef
    node_value: ValueRef
    parent: str = ""
    guards: tuple[Guard, ...] = ()

    def __init__(
        self,
        *,
        name: str,
        source_op: Op,
        relation: SourceNodeRelation,
        parent_value: ValueRef,
        node_value: ValueRef,
        parent: str = "",
        guards: Sequence[Guard] = (),
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "parent_value", parent_value)
        object.__setattr__(self, "node_value", node_value)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "guards", tuple(guards))

    @classmethod
    def adjacent_unique_user(
        cls,
        name: str,
        *,
        source_op: Op,
        parent_result: ValueRef,
        node_operand: ValueRef,
        parent: str = "",
        guards: Sequence[Guard] = (),
    ) -> Self:
        """Finds the adjacent sole user of a parent result."""

        return cls(
            name=name,
            source_op=source_op,
            relation=SourceNodeRelation.ADJACENT_UNIQUE_USER,
            parent_value=parent_result,
            node_value=node_operand,
            parent=parent,
            guards=guards,
        )

    @classmethod
    def adjacent_definition(
        cls,
        name: str,
        *,
        source_op: Op,
        parent_operand: ValueRef,
        node_result: ValueRef,
        parent: str = "",
        guards: Sequence[Guard] = (),
    ) -> Self:
        """Finds the adjacent sole-use definition of a parent operand."""

        return cls(
            name=name,
            source_op=source_op,
            relation=SourceNodeRelation.ADJACENT_DEFINITION,
            parent_value=parent_operand,
            node_value=node_result,
            parent=parent,
            guards=guards,
        )

    def validate(self, source_ops: dict[str, Op]) -> None:
        if not self.name:
            raise ValueError("descriptor-rule source node name must be non-empty")
        if self.name in source_ops:
            raise ValueError(f"duplicate descriptor-rule source node '{self.name}'")
        parent_op = source_ops.get(self.parent)
        if parent_op is None:
            raise ValueError(
                f"{self.source_op.name}: source node '{self.name}' references "
                f"unknown parent '{self.parent}'"
            )
        if not isinstance(self.relation, SourceNodeRelation):
            raise ValueError(
                f"{self.source_op.name}: source node '{self.name}' has an "
                "unknown relation"
            )
        if self.source_op.regions or PURE not in self.source_op.traits:
            raise ValueError(
                f"{self.source_op.name}: related source nodes must be pure and "
                "regionless"
            )
        for subject, value_ref in (
            ("parent connection", self.parent_value),
            ("node connection", self.node_value),
        ):
            if value_ref.source_node:
                raise ValueError(
                    f"{self.source_op.name}: source node '{self.name}' {subject} "
                    "must be local to its endpoint"
                )
            if value_ref.materializer is not None:
                raise ValueError(
                    f"{self.source_op.name}: source node '{self.name}' {subject} "
                    "cannot use a materializer"
                )
        if self.relation is SourceNodeRelation.ADJACENT_UNIQUE_USER:
            expected_parent_kind = SourceValueKind.RESULT
            expected_node_kind = SourceValueKind.OPERAND
        else:
            expected_parent_kind = SourceValueKind.OPERAND
            expected_node_kind = SourceValueKind.RESULT
        if self.parent_value.kind is not expected_parent_kind:
            raise ValueError(
                f"{self.source_op.name}: source node '{self.name}' parent "
                f"connection must be a {expected_parent_kind.value}"
            )
        if self.node_value.kind is not expected_node_kind:
            raise ValueError(
                f"{self.source_op.name}: source node '{self.name}' node "
                f"connection must be a {expected_node_kind.value}"
            )
        self.parent_value.validate(parent_op, "source-node parent connection")
        self.node_value.validate(self.source_op, "source-node local connection")
        for guard in self.guards:
            guard.validate(self.source_op)
        source_ops[self.name] = self.source_op


@dataclass(frozen=True, slots=True)
class DescriptorRule:
    """Source-to-Low rule contract case authored in Python.

    Cases for the same source operation are selected by descending priority,
    preserving authored order among cases with equal priority.
    """

    source_op: Op
    descriptor: Descriptor | None
    source_nodes: tuple[SourceNode, ...] = ()
    guards: tuple[Guard, ...] = ()
    emit: tuple[ContractEmit, ...] = ()
    priority: int = 0
    report_key: str = ""

    def __init__(
        self,
        *,
        source_op: Op,
        descriptor: Descriptor | None = None,
        source_nodes: Sequence[SourceNode] = (),
        guards: Sequence[Guard] = (),
        emit: Sequence[ContractEmit] = (),
        priority: int = 0,
        report_key: str = "",
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "source_nodes", tuple(source_nodes))
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
        if self.descriptor is not None:
            _require_descriptor(descriptor_set, self.descriptor)
        if len(self.source_nodes) + 1 > MAX_SOURCE_NODES:
            raise ValueError(
                f"{self.source_op.name}: descriptor rules support at most "
                f"{MAX_SOURCE_NODES} source nodes"
            )
        source_ops = {"": self.source_op}
        for source_node in self.source_nodes:
            source_node.validate(source_ops)
        for guard in self.guards:
            guard.validate(self.source_op)
        defined_temporaries = set[str]()
        for emit in self.emit:
            if (
                isinstance(emit, EmitDescriptorOp)
                and emit.descriptor != self.descriptor
            ):
                _require_descriptor(descriptor_set, emit.descriptor)
            produced_temporaries = emit.validate(
                self.source_op,
                descriptor_set,
                defined_temporaries,
                source_ops=source_ops,
            )
            defined_temporaries.update(produced_temporaries)
        self._validate_related_results()
        self._validate_source_memory_index_preservation()
        self._validate_per_lane_sequence()

    def _validate_related_results(self) -> None:
        covered_results = set[tuple[str, str, int]]()
        for source_node in self.source_nodes:
            if source_node.node_value.kind is SourceValueKind.RESULT:
                covered_results.add(
                    (
                        source_node.name,
                        source_node.node_value.field,
                        source_node.node_value.element,
                    )
                )
            if source_node.parent and (
                source_node.parent_value.kind is SourceValueKind.RESULT
            ):
                covered_results.add(
                    (
                        source_node.parent,
                        source_node.parent_value.field,
                        source_node.parent_value.element,
                    )
                )
        for emit in self.emit:
            if isinstance(emit, EmitDescriptorOp):
                result_refs = tuple(emit.results.values())
            elif isinstance(
                emit,
                (EmitRegisterConcat, EmitRegisterCopy, EmitRegisterSlice),
            ):
                result_refs = (emit.result,)
            else:
                result_refs = ()
            covered_results.update(
                (value_ref.source_node, value_ref.field, value_ref.element)
                for value_ref in result_refs
                if value_ref.kind is SourceValueKind.RESULT and value_ref.source_node
            )
        for source_node in self.source_nodes:
            for result in source_node.source_op.results:
                if result.variadic:
                    raise ValueError(
                        f"{source_node.source_op.name}: related source node "
                        f"'{source_node.name}' cannot have variadic results"
                    )
                if (source_node.name, result.name, 0) not in covered_results:
                    raise ValueError(
                        f"{source_node.source_op.name}: related source node "
                        f"'{source_node.name}' result '{result.name}' is neither "
                        "internal nor bound by the emit program"
                    )

    def _validate_source_memory_index_preservation(self) -> None:
        source_memories = tuple(
            emit.source_memory
            for emit in self.emit
            if isinstance(emit, EmitDescriptorOp)
            if emit.source_memory is not None
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
            and not value_ref.source_node
            and value_ref.field == memory_access.indices
            for emit in self.emit
            if isinstance(emit, EmitDescriptorOp)
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
        sequence_start = next(
            (
                emit_index
                for emit_index, emit in enumerate(self.emit)
                if isinstance(emit, EmitDescriptorOp)
                and emit.form == DescriptorEmitForm.PER_LANE_SEQUENCE
            ),
            None,
        )
        if sequence_start is None:
            return
        sequence_emits = self.emit[sequence_start:]
        if any(
            not isinstance(emit, EmitDescriptorOp)
            or emit.form != DescriptorEmitForm.PER_LANE_SEQUENCE
            for emit in sequence_emits
        ):
            raise ValueError(
                f"{self.source_op.name}: per-lane-sequence emits must form the "
                "final contiguous emit-program tail"
            )
        if len(sequence_emits) < 2:
            raise ValueError(
                f"{self.source_op.name}: per-lane-sequence emit programs need "
                "at least two lane emits"
            )
        for sequence_index, emit in enumerate(sequence_emits):
            emit_index = sequence_start + sequence_index
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
            if sequence_index + 1 == len(sequence_emits):
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


def contract_case_priority(contract_case: ContractCase) -> int:
    """Returns the selection priority for a contract case."""

    if isinstance(contract_case, DescriptorRule):
        return contract_case.priority
    return 0


def _validate_report_key(source_op: Op, report_key: str) -> None:
    if not report_key:
        return
    if any(char.isspace() for char in report_key):
        raise ValueError(f"{source_op.name}: report key must not contain whitespace")
    if report_key != report_key.strip("."):
        raise ValueError(f"{source_op.name}: report key must not have empty segments")
    if ".." in report_key:
        raise ValueError(f"{source_op.name}: report key must not have empty segments")
