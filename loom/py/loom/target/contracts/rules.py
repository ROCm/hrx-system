# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract case records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

from loom.dsl import (
    CONTEXTUAL,
    CONVERGENT,
    FACT_IDENTITY,
    PURE,
    MemoryAccessInterface,
    Op,
    TiedResult,
)
from loom.target.contracts.descriptors import _require_descriptor
from loom.target.contracts.emits import (
    ContractEmit,
    DescriptorResultType,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterMove,
    EmitRegisterSlice,
    is_per_lane_sequence,
)
from loom.target.contracts.guards import Guard, GuardKind
from loom.target.contracts.immediates import OperandSegmentProject
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.source import ValueRef
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorOpKind,
    DescriptorSet,
    OperandRole,
)

MAX_SOURCE_NODES = 8
SOURCE_NODE_COUNT_BITS = (MAX_SOURCE_NODES - 1).bit_length()
MAX_RULE_PRIORITY = 15


@unique
class SourceNodeRelation(Enum):
    """SSA relation used to find one source op from another."""

    ADJACENT_UNIQUE_USER = "adjacent_unique_user"
    ADJACENT_DEFINITION = "adjacent_definition"
    EXCLUSIVE_USER = "exclusive_user"
    EXCLUSIVE_DEFINITION = "exclusive_definition"


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

    @classmethod
    def exclusive_user(
        cls,
        name: str,
        *,
        source_op: Op,
        parent_result: ValueRef,
        node_operand: ValueRef,
        parent: str = "",
        guards: Sequence[Guard] = (),
    ) -> Self:
        """Finds the sole user of a parent result in the same block."""

        return cls(
            name=name,
            source_op=source_op,
            relation=SourceNodeRelation.EXCLUSIVE_USER,
            parent_value=parent_result,
            node_value=node_operand,
            parent=parent,
            guards=guards,
        )

    @classmethod
    def exclusive_definition(
        cls,
        name: str,
        *,
        source_op: Op,
        parent_operand: ValueRef,
        node_result: ValueRef,
        parent: str = "",
        guards: Sequence[Guard] = (),
    ) -> Self:
        """Finds a same-block definition whose result has one use."""

        return cls(
            name=name,
            source_op=source_op,
            relation=SourceNodeRelation.EXCLUSIVE_DEFINITION,
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
        if any(
            trait in self.source_op.traits for trait in (CONVERGENT, CONTEXTUAL)
        ) or any(isinstance(result, TiedResult) for result in self.source_op.results):
            raise ValueError(
                f"{self.source_op.name}: related source nodes must be relocatable"
            )
        if not self.source_op.results:
            raise ValueError(
                f"{self.source_op.name}: related source nodes must produce a result"
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
        if self.relation in (
            SourceNodeRelation.ADJACENT_UNIQUE_USER,
            SourceNodeRelation.EXCLUSIVE_USER,
        ):
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
    preserving authored order among cases with equal priority. Related source
    graphs also use the priority to resolve ownership across root operations.
    Rules that only combine operations already covered by singleton rules set
    target_query to false so they participate in lowering without independently
    establishing target legality.
    """

    source_op: Op
    descriptor: Descriptor | None
    source_nodes: tuple[SourceNode, ...] = ()
    guards: tuple[Guard, ...] = ()
    emit: tuple[ContractEmit, ...] = ()
    priority: int = 0
    report_key: str = ""
    target_query: bool = True
    # Setup temporaries updated by the repeated tail and carried to its next lane.
    sequence_carried: tuple[str, ...] = ()

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
        target_query: bool = True,
        sequence_carried: Sequence[str] = (),
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "source_nodes", tuple(source_nodes))
        object.__setattr__(self, "guards", tuple(guards))
        object.__setattr__(self, "emit", tuple(emit))
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "report_key", report_key)
        object.__setattr__(self, "target_query", target_query)
        object.__setattr__(self, "sequence_carried", tuple(sequence_carried))
        if priority < 0:
            raise ValueError("descriptor rule priority must be non-negative")
        if priority > MAX_RULE_PRIORITY:
            raise ValueError(
                f"descriptor rule priority must not exceed {MAX_RULE_PRIORITY}"
            )
        _validate_report_key(source_op, report_key)

    @property
    def system(self) -> ContractSystem:
        return (
            ContractSystem.DESCRIPTOR_RULE
            if self.target_query
            else ContractSystem.DESCRIPTOR_RULE_LOWERING_ONLY
        )

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
        self._validate_source_graph_forest(source_ops)
        for guard in self.guards:
            guard.validate(self.source_op)
        self._validate_per_lane_sequence()
        defined_temporaries = set[str]()
        for emit in self.emit:
            if (
                isinstance(emit, EmitDescriptorOp)
                and emit.descriptor != self.descriptor
            ):
                _require_descriptor(descriptor_set, emit.descriptor)
            if isinstance(emit, EmitDescriptorOp):
                produced_temporaries = emit.validate(
                    self.source_op,
                    descriptor_set,
                    defined_temporaries,
                    source_ops=source_ops,
                    sequence_carried=self.sequence_carried,
                )
            else:
                produced_temporaries = emit.validate(
                    self.source_op,
                    descriptor_set,
                    defined_temporaries,
                    source_ops=source_ops,
                )
            defined_temporaries.update(produced_temporaries)
        self._validate_operand_segment_projections()
        self._validate_source_memory_preservation()
        self._validate_source_results(source_ops)
        self._validate_terminator_emit()

    def _validate_operand_segment_projections(self) -> None:
        exact_segment_guards = {
            (guard.field, guard.bit_width)
            for guard in self.guards
            if guard.kind is GuardKind.OPERAND_SEGMENT_EXACT_BITS
        }
        for emit in self.emit:
            if not isinstance(emit, EmitDescriptorOp) or not isinstance(
                emit.immediates, Mapping
            ):
                continue
            for projection in emit.immediates.values():
                if not isinstance(projection, OperandSegmentProject):
                    continue
                required_guard = (projection.source_operand, projection.bit_width)
                if required_guard not in exact_segment_guards:
                    raise ValueError(
                        f"{self.source_op.name}: operand-segment projection for "
                        f"'{projection.source_operand}' needs a matching "
                        "operand_segment_exact_bits guard"
                    )

    def _validate_source_graph_forest(self, source_ops: dict[str, Op]) -> None:
        endpoint_guards = {"": self.guards}
        endpoint_guards.update(
            (source_node.name, source_node.guards) for source_node in self.source_nodes
        )
        connected_results = set[tuple[str, str, int]]()
        for source_node in self.source_nodes:
            if source_node.relation in (
                SourceNodeRelation.ADJACENT_UNIQUE_USER,
                SourceNodeRelation.EXCLUSIVE_USER,
            ):
                definition_name = source_node.parent
                definition_value = source_node.parent_value
            else:
                definition_name = source_node.name
                definition_value = source_node.node_value
            result_key = (
                definition_name,
                definition_value.field,
                definition_value.element,
            )
            if result_key in connected_results:
                endpoint = definition_name or "root"
                raise ValueError(
                    f"{self.source_op.name}: source graph result "
                    f"'{endpoint}.{definition_value.field}' has multiple users"
                )
            connected_results.add(result_key)

            definition_op = source_ops[definition_name]
            if any(result.variadic for result in definition_op.results):
                endpoint = definition_name or "root"
                raise ValueError(
                    f"{self.source_op.name}: source graph definition '{endpoint}' "
                    "cannot have variadic results"
                )
            if len(definition_op.results) <= 1:
                continue
            dead_result_fields = {
                guard.field
                for guard in endpoint_guards[definition_name]
                if guard.kind is GuardKind.VALUE_NO_USES
            }
            live_sibling_fields = tuple(
                result.name
                for result in definition_op.results
                if result.name != definition_value.field
                and result.name not in dead_result_fields
            )
            if live_sibling_fields:
                endpoint = definition_name or "root"
                raise ValueError(
                    f"{self.source_op.name}: source graph definition '{endpoint}' "
                    "must guard sibling results as unused: "
                    + ", ".join(live_sibling_fields)
                )

    def _validate_source_results(self, source_ops: Mapping[str, Op]) -> None:
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
            if source_node.parent_value.kind is SourceValueKind.RESULT:
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
                (
                    EmitRegisterConcat,
                    EmitRegisterCopy,
                    EmitRegisterMove,
                    EmitRegisterSlice,
                ),
            ):
                result_refs = (emit.result,)
            else:
                result_refs = ()
            covered_results.update(
                (value_ref.source_node, value_ref.field, value_ref.element)
                for value_ref in result_refs
                if value_ref.kind is SourceValueKind.RESULT
            )
        for source_node_name, source_op in source_ops.items():
            endpoint = source_node_name or "root"
            endpoint_guards = (
                self.guards
                if not source_node_name
                else next(
                    node.guards
                    for node in self.source_nodes
                    if node.name == source_node_name
                )
            )
            for guard in endpoint_guards:
                if (
                    guard.kind is GuardKind.VALUE_NO_USES
                    and source_op.result(guard.field) is not None
                ):
                    covered_results.add(
                        (
                            source_node_name,
                            guard.field,
                            guard.element if guard.element is not None else 0,
                        )
                    )
            for result in source_op.results:
                if result.variadic:
                    if source_node_name:
                        raise ValueError(
                            f"{source_op.name}: source node '{endpoint}' cannot "
                            "have variadic results"
                        )
                    continue
                if (source_node_name, result.name, 0) not in covered_results:
                    rule_key = self.descriptor.key if self.descriptor else "<none>"
                    raise ValueError(
                        f"{source_op.name}: descriptor rule '{rule_key}' source "
                        f"node '{endpoint}' result '{result.name}' is neither "
                        "internal nor bound by the emit program"
                    )

    def _validate_source_memory_preservation(self) -> None:
        source_memories = tuple(
            emit.source_memory
            for emit in self.emit
            if isinstance(emit, (EmitDescriptorOp, EmitRegisterCopy))
            if emit.source_memory is not None
        )
        if not source_memories:
            return
        if any(memory.reuse_view_base for memory in source_memories):
            # One rule shares its materialized offsets across emit rows. Every
            # row must use the same dynamic/static decomposition.
            if not all(memory.reuse_view_base for memory in source_memories):
                raise ValueError(
                    f"{self.source_op.name}: memory emits must agree on view-base reuse"
                )
            offset_used = False
            for emit in self.emit:
                if not isinstance(emit, EmitDescriptorOp):
                    continue
                for value_ref in emit.operands.values():
                    if value_ref.kind == SourceValueKind.SOURCE_MEMORY_ADDRESS:
                        raise ValueError(
                            f"{self.source_op.name}: view-base reuse requires "
                            "byte-offset materialization"
                        )
                    offset_used |= value_ref.kind in (
                        SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET,
                        SourceValueKind.SOURCE_MEMORY_BYTE_OFFSET,
                    )
            if not offset_used:
                raise ValueError(
                    f"{self.source_op.name}: view-base reuse requires a byte offset"
                )
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
                index
                for index, emit in enumerate(self.emit)
                if is_per_lane_sequence(emit)
            ),
            None,
        )
        if sequence_start is None:
            if self.sequence_carried:
                raise ValueError(
                    f"{self.source_op.name}: "
                    "carried temporaries require a per-lane sequence"
                )
            return
        sequence_emits = self.emit[sequence_start:]
        if not all(is_per_lane_sequence(emit) for emit in sequence_emits):
            raise ValueError(
                f"{self.source_op.name}: per-lane-sequence emits must form the "
                "final contiguous emit-program tail"
            )
        if len(sequence_emits) == 1 and sequence_start == 0:
            raise ValueError(
                f"{self.source_op.name}: a single per-lane-sequence emit "
                "requires shared setup"
            )
        carried = set(self.sequence_carried)
        if len(carried) != len(self.sequence_carried) or any(
            not name for name in carried
        ):
            raise ValueError(
                f"{self.source_op.name}: "
                "carried temporary names must be unique and nonempty"
            )
        setup = {
            ref.field
            for emit in self.emit[:sequence_start]
            for ref in (
                tuple(emit.results.values())
                if isinstance(emit, EmitDescriptorOp)
                else (emit.result,)
            )
            if ref.kind is SourceValueKind.TEMPORARY
        }
        if not carried <= setup:
            raise ValueError(
                f"{self.source_op.name}: carried temporaries require a setup definition"
            )
        updated = set[str]()
        for sequence_index, emit in enumerate(sequence_emits):
            final = sequence_index + 1 == len(sequence_emits)
            if isinstance(emit, EmitRegisterConcat):
                if not final:
                    raise ValueError(
                        f"{self.source_op.name}: "
                        "per-lane-sequence concat must be the final emit"
                    )
                result_refs = (emit.result,)
                operand_refs = emit.sources
            else:
                for operand in emit.descriptor.operands:
                    if (
                        operand.role is not OperandRole.IMPLICIT
                        and operand.unit_count == 0
                    ):
                        raise ValueError(
                            f"{self.source_op.name}: per-lane-sequence descriptor "
                            f"'{emit.descriptor.key}' field '{operand.field_name}' "
                            "requires a fixed register width"
                        )
                result_refs = tuple(emit.results.values())
                operand_refs = tuple(emit.operands.values())
                if emit.skip_last_lane and (
                    final
                    or any(
                        ref.kind is not SourceValueKind.TEMPORARY
                        or ref.field not in carried
                        for ref in result_refs
                    )
                ):
                    raise ValueError(
                        f"{self.source_op.name}: "
                        "skipped sequence results must update carried temporaries"
                    )
            if final and self.source_op.results:
                if len(result_refs) != 1:
                    raise ValueError(
                        f"{self.source_op.name}: "
                        "per-lane-sequence final emit must bind exactly one result"
                    )
                if result_refs[0].kind is not SourceValueKind.RESULT:
                    raise ValueError(
                        f"{self.source_op.name}: "
                        "per-lane-sequence final emit must bind a source result"
                    )
            elif any(ref.kind is not SourceValueKind.TEMPORARY for ref in result_refs):
                raise ValueError(
                    f"{self.source_op.name}: "
                    "per-lane-sequence non-collecting emit must bind a temporary"
                )
            updated.update(
                ref.field
                for ref in result_refs
                if ref.kind is SourceValueKind.TEMPORARY
            )
            if any(ref.materializer is not None for ref in operand_refs):
                raise ValueError(
                    f"{self.source_op.name}: "
                    "per-lane-sequence operands cannot use value materializers"
                )
            if any(
                ref.kind
                in (
                    SourceValueKind.SOURCE_MEMORY_ROOT,
                    SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM,
                    SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET,
                    SourceValueKind.SOURCE_MEMORY_BYTE_OFFSET,
                    SourceValueKind.SOURCE_MEMORY_ADDRESS,
                    SourceValueKind.SOURCE_MEMORY_ELEMENT_STRIDE,
                )
                for ref in operand_refs
            ):
                raise ValueError(
                    f"{self.source_op.name}: per-lane-sequence address geometry "
                    "must be materialized in shared setup"
                )
        if not carried <= updated:
            raise ValueError(
                f"{self.source_op.name}: carried temporaries require a sequence update"
            )
        self._validate_sequence_carriers(carried)

    def _validate_sequence_carriers(self, carried: set[str]) -> None:
        # Compare symbolic type identities at generation time. A reference to
        # the incoming temporary preserves that type directly; native descriptor
        # bindings preserve it when their register class and width agree.
        if not carried:
            return
        types: dict[str, object] = {}
        for emit in self.emit:
            if isinstance(emit, EmitDescriptorOp):
                definitions = []
                for operand in emit.descriptor.operands:
                    ref = emit.results.get(operand.field_name)
                    if ref is None or ref.kind is not SourceValueKind.TEMPORARY:
                        continue
                    binding = (emit.result_types or {}).get(operand.field_name, ref)
                    if isinstance(binding, DescriptorResultType):
                        binding = (operand.reg_alts[0].reg_class, operand.unit_count)
                    definitions.append((ref, binding))
            else:
                source = (
                    emit.sources[0]
                    if isinstance(emit, EmitRegisterConcat)
                    else emit.source
                )
                definitions = [(emit.result, emit.result_type or source)]
            resolved = []
            for ref, binding in definitions:
                if (
                    isinstance(binding, ValueRef)
                    and binding.kind is SourceValueKind.TEMPORARY
                ):
                    binding = types.get(binding.field, binding)
                if (
                    isinstance(emit, (EmitRegisterSlice, EmitRegisterConcat))
                    and emit.unit_count is not None
                ):
                    if (
                        isinstance(emit, EmitRegisterSlice)
                        and emit.result_register_class
                    ):
                        binding = (emit.result_register_class, emit.unit_count)
                    elif isinstance(binding, tuple):
                        binding = (binding[0], emit.unit_count)
                    else:
                        binding = (binding, emit.unit_count)
                if (
                    is_per_lane_sequence(emit)
                    and ref.field in carried
                    and binding != types.get(ref.field)
                ):
                    raise ValueError(
                        f"{self.source_op.name}: carried temporary '{ref.field}' "
                        "must preserve its incoming result type"
                    )
                resolved.append((ref.field, binding))
            types.update(resolved)

    def _validate_terminator_emit(self) -> None:
        for emit_index, emit in enumerate(self.emit):
            if (
                not isinstance(emit, EmitDescriptorOp)
                or emit.descriptor.op_kind is not DescriptorOpKind.COND_BR
            ):
                continue
            if emit_index + 1 != len(self.emit):
                raise ValueError(
                    f"{self.source_op.name}: low.op.cond_br must be the final "
                    "emit in a descriptor rule"
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
class UnsupportedRule:
    """Contract case explicitly rejecting unmatched forms of an operation."""

    source_op: Op

    @property
    def system(self) -> ContractSystem:
        return ContractSystem.UNSUPPORTED

    def validate(self, descriptor_set: DescriptorSet) -> None:
        del descriptor_set


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
    | UnsupportedRule
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
