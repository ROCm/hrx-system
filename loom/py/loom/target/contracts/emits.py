# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-Low contract emission forms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import Op
from loom.target.contracts.descriptors import (
    _immediate_has_default,
    _require_descriptor,
    _require_descriptor_operand,
    _require_immediate,
    _require_input_descriptor_role,
    _require_output_descriptor_role,
    _validate_immediate_literal,
    _validate_required_descriptor_operands,
)
from loom.target.contracts.immediates import (
    AttrProject,
    SourceMemoryProject,
    SourceMemoryProjectKind,
    SourceOpProject,
    ValueProject,
)
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source import ValueRef
from loom.target.contracts.source_memory import (
    SourceMemoryAddressMaterializer,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorOpKind,
    DescriptorSet,
    ImmediateKind,
    Operand,
    OperandRole,
    RegClassAltFlag,
)


@unique
class DescriptorEmitForm(Enum):
    """Descriptor emission form used by the target-low lowering interpreter."""

    AUTO = "auto"
    OP = "op"
    CONST = "const"
    FIRST_LANE = "first_lane"
    PER_LANE = "per_lane"
    PER_LANE_SEQUENCE = "per_lane_sequence"
    ACCUMULATE_LANES = "accumulate_lanes"


@unique
class DescriptorAccumulatorSeed(Enum):
    """Initial accumulator source for lane-accumulating descriptor emits."""

    OPERAND = "operand"
    FIRST_LANE = "first_lane"


@unique
class DescriptorAccumulatorTree(Enum):
    """Lane-combining tree shape for lane-accumulating descriptor emits."""

    CHAIN = "chain"
    BALANCED = "balanced"


@dataclass(frozen=True, slots=True)
class DescriptorResultType:
    """Uses the descriptor result operand's concrete low register type."""


type ResultTypeBinding = ValueRef | TypePattern | DescriptorResultType


@dataclass(frozen=True, slots=True)
class EmitRegisterSlice:
    """Projects consecutive register units from one source value."""

    source: ValueRef
    result: ValueRef
    unit_offset: int = 0
    result_type: ResultTypeBinding | None = None

    def __post_init__(self) -> None:
        if self.unit_offset < 0 or self.unit_offset > 0xFFFF:
            raise ValueError("register slice unit offset must fit u16")

    def validate(
        self,
        source_op: Op,
        descriptor_set: DescriptorSet,
        defined_temporaries: set[str],
    ) -> tuple[str, ...]:
        del descriptor_set
        _validate_structural_source(
            source_op,
            self.source,
            "register slice source",
            defined_temporaries,
        )
        return _validate_structural_result(
            source_op,
            self.result,
            self.result_type,
            "register slice result",
            defined_temporaries,
        )


@dataclass(frozen=True, slots=True)
class EmitRegisterConcat:
    """Concatenates register-unit sources into one aggregate value."""

    sources: tuple[ValueRef, ...]
    result: ValueRef
    result_type: ResultTypeBinding | None = None

    def __init__(
        self,
        *,
        sources: Sequence[ValueRef],
        result: ValueRef,
        result_type: ResultTypeBinding | None = None,
    ) -> None:
        object.__setattr__(self, "sources", tuple(sources))
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "result_type", result_type)
        if not sources:
            raise ValueError("register concat needs at least one source")

    def validate(
        self,
        source_op: Op,
        descriptor_set: DescriptorSet,
        defined_temporaries: set[str],
    ) -> tuple[str, ...]:
        del descriptor_set
        for source_index, source in enumerate(self.sources):
            _validate_structural_source(
                source_op,
                source,
                f"register concat source {source_index}",
                defined_temporaries,
            )
        return _validate_structural_result(
            source_op,
            self.result,
            self.result_type,
            "register concat result",
            defined_temporaries,
        )


type ContractEmit = EmitDescriptorOp | EmitRegisterSlice | EmitRegisterConcat


@dataclass(frozen=True, slots=True)
class EmitDescriptorOp:
    """Emits one descriptor-backed low op from source fields."""

    descriptor: Descriptor
    operands: Mapping[str, ValueRef] | None = None
    results: Mapping[str, ValueRef] | None = None
    result_types: Mapping[str, ResultTypeBinding] | None = None
    immediates: (
        Mapping[
            str,
            AttrProject | SourceOpProject | ValueProject | SourceMemoryProject | int,
        ]
        | Sequence[AttrProject]
    ) = ()
    form: DescriptorEmitForm = DescriptorEmitForm.AUTO
    swap_first_two_operands: bool = False
    copy_operands: Sequence[str] = ()
    accumulator: str | None = None
    accumulator_seed: DescriptorAccumulatorSeed = DescriptorAccumulatorSeed.OPERAND
    accumulator_tree: DescriptorAccumulatorTree = DescriptorAccumulatorTree.CHAIN
    skip_first_lane: bool = False
    source_memory: SourceMemoryConstraint | None = None
    source_memory_byte_offset_materializer: (
        SourceMemoryByteOffsetMaterializer | None
    ) = None
    source_memory_address_materializer: SourceMemoryAddressMaterializer | None = None

    def __post_init__(self) -> None:
        operand_bindings = self.operands if self.operands is not None else {}
        result_bindings = self.results if self.results is not None else {}
        result_type_bindings = (
            self.result_types if self.result_types is not None else None
        )
        object.__setattr__(self, "operands", dict(operand_bindings))
        object.__setattr__(self, "results", dict(result_bindings))
        object.__setattr__(
            self,
            "result_types",
            None if result_type_bindings is None else dict(result_type_bindings),
        )
        object.__setattr__(self, "copy_operands", tuple(self.copy_operands))
        for operand in self.copy_operands:
            if not operand:
                raise ValueError("copied descriptor operand name must be non-empty")
        if self.accumulator is not None and not self.accumulator:
            raise ValueError("descriptor accumulator field must be non-empty")

    def validate(
        self,
        source_op: Op,
        descriptor_set: DescriptorSet,
        defined_temporaries: set[str],
    ) -> tuple[str, ...]:
        _require_descriptor(descriptor_set, self.descriptor)
        if (
            self.form == DescriptorEmitForm.CONST
            and self.descriptor.op_kind is not DescriptorOpKind.CONST
        ):
            raise ValueError(
                f"{source_op.name}: descriptor '{self.descriptor.key}' uses "
                "low.op but the contract requests low.const"
            )
        if (
            self.form
            in (
                DescriptorEmitForm.OP,
                DescriptorEmitForm.FIRST_LANE,
                DescriptorEmitForm.PER_LANE,
                DescriptorEmitForm.PER_LANE_SEQUENCE,
                DescriptorEmitForm.ACCUMULATE_LANES,
            )
            and self.descriptor.op_kind is not DescriptorOpKind.OP
        ):
            raise ValueError(
                f"{source_op.name}: descriptor '{self.descriptor.key}' uses "
                "low.const but the contract requests a low.op emission form"
            )
        operand_bindings = dict(self.operands) if self.operands is not None else {}
        result_bindings = dict(self.results) if self.results is not None else {}
        result_type_bindings = (
            dict(self.result_types) if self.result_types is not None else {}
        )
        for descriptor_field, value_ref in operand_bindings.items():
            operand = _require_descriptor_operand(
                self.descriptor, descriptor_field, "descriptor operand binding"
            )
            _require_input_descriptor_role(self.descriptor, operand)
            value_ref.validate(
                source_op,
                f"descriptor '{self.descriptor.key}' operand '{descriptor_field}'",
                defined_temporaries=defined_temporaries,
            )
        produced_temporaries = []
        for descriptor_field, value_ref in result_bindings.items():
            operand = _require_descriptor_operand(
                self.descriptor, descriptor_field, "descriptor result binding"
            )
            _require_output_descriptor_role(self.descriptor, operand)
            if value_ref.kind not in (
                SourceValueKind.RESULT,
                SourceValueKind.TEMPORARY,
            ):
                raise ValueError(
                    f"{source_op.name}: descriptor result '{descriptor_field}' "
                    "must bind a source result or temporary"
                )
            if value_ref.kind == SourceValueKind.TEMPORARY:
                if (
                    value_ref.field in defined_temporaries
                    or value_ref.field in produced_temporaries
                ):
                    raise ValueError(
                        f"{source_op.name}: descriptor result '{descriptor_field}' "
                        f"redefines temporary '{value_ref.field}'"
                    )
                result_type_binding = result_type_bindings.get(descriptor_field)
                if result_type_binding is None:
                    raise ValueError(
                        f"{source_op.name}: descriptor result "
                        f"'{descriptor_field}' temporary '{value_ref.field}' "
                        "needs an explicit result type binding"
                    )
                if isinstance(result_type_binding, DescriptorResultType):
                    _require_descriptor_result_type_operand(
                        source_op,
                        operand,
                    )
                produced_temporaries.append(value_ref.field)
            else:
                value_ref.validate(
                    source_op,
                    f"descriptor '{self.descriptor.key}' result '{descriptor_field}'",
                )
        for descriptor_field, binding in result_type_bindings.items():
            operand = _require_descriptor_operand(
                self.descriptor, descriptor_field, "descriptor result type binding"
            )
            if isinstance(binding, DescriptorResultType):
                _require_output_descriptor_role(self.descriptor, operand)
                _require_descriptor_result_type_operand(source_op, operand)
                continue
            if isinstance(binding, ValueRef):
                if binding.kind in (
                    SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM,
                    SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET,
                    SourceValueKind.SOURCE_MEMORY_ADDRESS,
                ):
                    raise ValueError(
                        f"{source_op.name}: descriptor result type "
                        f"'{descriptor_field}' cannot bind a source-memory value"
                    )
                binding.validate(
                    source_op,
                    f"descriptor result type '{descriptor_field}'",
                    defined_temporaries=defined_temporaries,
                )
        _validate_required_descriptor_operands(
            source_op,
            self.descriptor,
            operand_bindings.keys(),
            result_bindings.keys(),
        )
        if self.swap_first_two_operands and len(operand_bindings) < 2:
            raise ValueError(
                f"{source_op.name}: descriptor operand swap needs at least two operands"
            )
        for descriptor_field in self.copy_operands:
            if descriptor_field not in operand_bindings:
                raise ValueError(
                    f"{source_op.name}: copied descriptor operand "
                    f"'{descriptor_field}' is not an operand binding"
                )
            _require_descriptor_copy_operand(
                source_op,
                self.descriptor,
                descriptor_field,
            )
        if self.form == DescriptorEmitForm.ACCUMULATE_LANES:
            if self.accumulator is None:
                raise ValueError(
                    f"{source_op.name}: accumulate-lanes emit needs an accumulator"
                )
            if self.accumulator not in operand_bindings:
                raise ValueError(
                    f"{source_op.name}: accumulator '{self.accumulator}' "
                    "is not a descriptor operand binding"
                )
        elif self.accumulator is not None:
            raise ValueError(
                f"{source_op.name}: accumulator is only valid for "
                "accumulate-lanes emits"
            )
        if (
            self.accumulator_seed != DescriptorAccumulatorSeed.OPERAND
            and self.form != DescriptorEmitForm.ACCUMULATE_LANES
        ):
            raise ValueError(
                f"{source_op.name}: accumulator seed is only valid for "
                "accumulate-lanes emits"
            )
        if (
            self.accumulator_tree != DescriptorAccumulatorTree.CHAIN
            and self.form != DescriptorEmitForm.ACCUMULATE_LANES
        ):
            raise ValueError(
                f"{source_op.name}: accumulator tree is only valid for "
                "accumulate-lanes emits"
            )
        if self.skip_first_lane and self.form != DescriptorEmitForm.ACCUMULATE_LANES:
            raise ValueError(
                f"{source_op.name}: skip-first-lane is only valid for "
                "accumulate-lanes emits"
            )
        if (
            self.skip_first_lane
            and self.accumulator_seed == DescriptorAccumulatorSeed.FIRST_LANE
        ):
            raise ValueError(
                f"{source_op.name}: skip-first-lane cannot be combined with "
                "first-lane accumulator seeding"
            )
        self._validate_source_memory_emit(
            source_op,
            descriptor_set,
            operand_bindings,
        )
        self._validate_immediates(source_op, descriptor_set)
        return tuple(produced_temporaries)

    def _validate_source_memory_emit(
        self,
        source_op: Op,
        descriptor_set: DescriptorSet,
        operand_bindings: Mapping[str, ValueRef],
    ) -> None:
        if self.source_memory is not None and self.form not in (
            DescriptorEmitForm.AUTO,
            DescriptorEmitForm.OP,
        ):
            raise ValueError(f"{source_op.name}: source memory requires an op emit")
        if self.source_memory is not None:
            self.source_memory.validate(source_op)
        if self.source_memory_byte_offset_materializer is not None:
            if self.source_memory is None:
                raise ValueError(
                    f"{source_op.name}: source-memory byte-offset materializer "
                    "needs a source-memory emit"
                )
            materializer = self.source_memory_byte_offset_materializer
            for descriptor in (
                materializer.const_i64,
                materializer.add_i64,
                materializer.mul_i64,
                materializer.shl_i64,
            ):
                if descriptor is not None:
                    _require_descriptor(descriptor_set, descriptor)
            _validate_byte_offset_materializer(source_op, materializer)
        if self.source_memory_address_materializer is not None:
            if self.source_memory is None:
                raise ValueError(
                    f"{source_op.name}: source-memory address materializer needs "
                    "a source-memory emit"
                )
            materializer = self.source_memory_address_materializer
            for descriptor in (
                materializer.const_coordinate,
                materializer.add_coordinate,
                materializer.mul_coordinate,
                materializer.shl_coordinate,
                materializer.index_to_coordinate_input,
                materializer.index_to_coordinate,
                materializer.address,
            ):
                if descriptor is not None:
                    _require_descriptor(descriptor_set, descriptor)
            _validate_address_materializer(source_op, materializer)
        for descriptor_field, value_ref in operand_bindings.items():
            if value_ref.kind != SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM:
                continue
            if self.source_memory is None:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' needs a source-memory emit"
                )
            if self.source_memory.dynamic_term_count is None:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' needs a fixed source-memory "
                    "dynamic term count"
                )
            if value_ref.element >= self.source_memory.dynamic_term_count:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' references dynamic term "
                    f"{value_ref.element}, but the source-memory constraint only "
                    f"selects {self.source_memory.dynamic_term_count}"
                )
        for descriptor_field, value_ref in operand_bindings.items():
            if value_ref.kind != SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
                continue
            if self.source_memory is None:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' needs a source-memory emit"
                )
            if self.source_memory.dynamic_term_count is None:
                if self.source_memory.dynamic_term_count_minimum == 0:
                    raise ValueError(
                        f"{source_op.name}: descriptor '{self.descriptor.key}' "
                        f"operand '{descriptor_field}' needs dynamic source "
                        "memory"
                    )
            elif self.source_memory.dynamic_term_count == 0:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' needs dynamic source memory"
                )
            if self.source_memory_byte_offset_materializer is None:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' needs a source-memory byte "
                    "offset materializer"
                )
        for descriptor_field, value_ref in operand_bindings.items():
            if value_ref.kind != SourceValueKind.SOURCE_MEMORY_ADDRESS:
                continue
            if self.source_memory is None:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' needs a source-memory emit"
                )
            if self.source_memory_address_materializer is None:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"operand '{descriptor_field}' needs a source-memory address "
                    "materializer"
                )

    def _validate_immediates(
        self,
        source_op: Op,
        descriptor_set: DescriptorSet,
    ) -> None:
        if isinstance(self.immediates, Mapping):
            bound_names = set[str]()
            for immediate_name, binding in self.immediates.items():
                immediate = _require_immediate(
                    self.descriptor,
                    immediate_name,
                    "descriptor immediate binding",
                )
                if isinstance(binding, AttrProject | SourceOpProject | ValueProject):
                    binding.validate(source_op, self.descriptor, immediate_name)
                elif isinstance(binding, SourceMemoryProject):
                    if self.source_memory is None:
                        raise ValueError(
                            f"{source_op.name}: source-memory immediate "
                            f"'{immediate_name}' needs a source-memory emit"
                        )
                    binding.validate(source_op, self.descriptor, immediate_name)
                    if (
                        binding.kind
                        == SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL
                    ):
                        projected_minimum = (
                            self.source_memory.static_byte_offset_minimum
                            + binding.literal_i64
                        )
                        projected_maximum = (
                            self.source_memory.static_byte_offset_maximum
                            + binding.literal_i64
                        )
                        if not (
                            -(2**63)
                            <= projected_minimum
                            <= projected_maximum
                            <= (2**63) - 1
                        ):
                            raise ValueError(
                                f"{source_op.name}: source-memory immediate "
                                f"'{immediate_name}' static byte offset range plus "
                                f"{binding.literal_i64} must fit in signed i64"
                            )
                    if binding.kind == SourceMemoryProjectKind.DYNAMIC_BYTE_STRIDE and (
                        self.source_memory.dynamic_term_count is None
                        or binding.dynamic_term_index
                        >= self.source_memory.dynamic_term_count
                    ):
                        raise ValueError(
                            f"{source_op.name}: source-memory immediate "
                            f"'{immediate_name}' references dynamic term "
                            f"{binding.dynamic_term_index}, but the source-memory "
                            "constraint only selects "
                            f"{self.source_memory.dynamic_term_count}"
                        )
                else:
                    _validate_immediate_literal(
                        source_op,
                        descriptor_set,
                        self.descriptor,
                        immediate,
                        binding,
                    )
                bound_names.add(immediate_name)
        else:
            bound_names = set[str]()
            for projection in self.immediates:
                projection.validate(source_op, self.descriptor, None)
                bound_names.update(projection.target_names)
        for immediate in self.descriptor.immediates:
            if _immediate_has_default(immediate):
                continue
            if immediate.field_name not in bound_names:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"immediate '{immediate.field_name}' is not bound"
                )


def _validate_structural_source(
    source_op: Op,
    source: ValueRef,
    subject: str,
    defined_temporaries: set[str],
) -> None:
    if source.kind not in (SourceValueKind.OPERAND, SourceValueKind.TEMPORARY):
        raise ValueError(
            f"{source_op.name}: {subject} must bind an operand or temporary"
        )
    source.validate(
        source_op,
        subject,
        defined_temporaries=defined_temporaries,
    )


def _validate_structural_result(
    source_op: Op,
    result: ValueRef,
    result_type: ResultTypeBinding | None,
    subject: str,
    defined_temporaries: set[str],
) -> tuple[str, ...]:
    if result.kind not in (SourceValueKind.RESULT, SourceValueKind.TEMPORARY):
        raise ValueError(
            f"{source_op.name}: {subject} must bind a source result or temporary"
        )
    produced_temporaries: tuple[str, ...] = ()
    if result.kind == SourceValueKind.TEMPORARY:
        if result.field in defined_temporaries:
            raise ValueError(
                f"{source_op.name}: {subject} redefines temporary '{result.field}'"
            )
        if result_type is None:
            raise ValueError(
                f"{source_op.name}: {subject} temporary '{result.field}' "
                "needs an explicit result type binding"
            )
        produced_temporaries = (result.field,)
    else:
        result.validate(source_op, subject)

    if isinstance(result_type, DescriptorResultType):
        raise ValueError(f"{source_op.name}: {subject} has no descriptor result type")
    if isinstance(result_type, ValueRef):
        if result_type.kind in (
            SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM,
            SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET,
            SourceValueKind.SOURCE_MEMORY_ADDRESS,
        ):
            raise ValueError(
                f"{source_op.name}: {subject} type cannot bind source memory"
            )
        if result_type.kind == SourceValueKind.TEMPORARY:
            raise ValueError(
                f"{source_op.name}: {subject} type cannot bind a temporary"
            )
        result_type.validate(
            source_op,
            f"{subject} type",
            defined_temporaries=defined_temporaries,
        )
    return produced_temporaries


def _require_descriptor_result_type_operand(
    source_op: Op,
    operand: Operand,
) -> None:
    if len(operand.reg_alts) != 1:
        raise ValueError(
            f"{source_op.name}: descriptor result '{operand.field_name}' cannot "
            "infer a descriptor result type from multiple register alternatives"
        )
    reg_alt = operand.reg_alts[0]
    if reg_alt.reg_class is None or RegClassAltFlag.IMMEDIATE in reg_alt.flags:
        raise ValueError(
            f"{source_op.name}: descriptor result '{operand.field_name}' cannot "
            "infer a descriptor result type from an immediate alternative"
        )
    if operand.unit_count == 0:
        raise ValueError(
            f"{source_op.name}: descriptor result '{operand.field_name}' cannot "
            "infer a descriptor result type with zero register units"
        )


def _require_descriptor_copy_operand(
    source_op: Op,
    descriptor: Descriptor,
    descriptor_field: str,
) -> None:
    operand = _require_descriptor_operand(
        descriptor,
        descriptor_field,
        "copied descriptor operand",
    )
    register_alternatives = tuple(
        alternative
        for alternative in operand.reg_alts
        if alternative.reg_class is not None
        and RegClassAltFlag.IMMEDIATE not in alternative.flags
    )
    if not register_alternatives:
        raise ValueError(
            f"{source_op.name}: copied descriptor operand "
            f"'{descriptor_field}' has no register alternative"
        )
    if operand.unit_count == 0:
        raise ValueError(
            f"{source_op.name}: copied descriptor operand "
            f"'{descriptor_field}' has zero register units"
        )


type _MaterializerCarrier = tuple[str, int]


_MATERIALIZER_PACKET_ROLES = frozenset(
    (OperandRole.OPERAND, OperandRole.PREDICATE, OperandRole.RESOURCE)
)


def _validate_materializer_descriptor(
    source_op: Op,
    descriptor: Descriptor,
    *,
    subject: str,
    op_kind: DescriptorOpKind,
    input_carriers: tuple[_MaterializerCarrier | None, ...],
    result_carrier: _MaterializerCarrier | None = None,
    bound_immediate: str | None = None,
) -> _MaterializerCarrier:
    if descriptor.op_kind is not op_kind:
        expected_kind = "low.const" if op_kind is DescriptorOpKind.CONST else "low.op"
        raise ValueError(
            f"{source_op.name}: {subject} descriptor '{descriptor.key}' must use "
            f"{expected_kind}"
        )

    results = tuple(
        operand for operand in descriptor.operands if operand.role is OperandRole.RESULT
    )
    if (
        len(results) != 1
        or not descriptor.operands
        or descriptor.operands[0] is not results[0]
    ):
        raise ValueError(
            f"{source_op.name}: {subject} descriptor '{descriptor.key}' must "
            "declare exactly one leading result"
        )
    unsupported = tuple(
        operand
        for operand in descriptor.operands
        if operand.role not in _MATERIALIZER_PACKET_ROLES
        and operand.role not in (OperandRole.RESULT, OperandRole.IMPLICIT)
    )
    if unsupported:
        raise ValueError(
            f"{source_op.name}: {subject} descriptor '{descriptor.key}' uses "
            "an unsupported operand-result role"
        )
    inputs = tuple(
        operand
        for operand in descriptor.operands
        if operand.role in _MATERIALIZER_PACKET_ROLES
    )
    if len(inputs) != len(input_carriers):
        raise ValueError(
            f"{source_op.name}: {subject} descriptor '{descriptor.key}' must "
            f"declare exactly {len(input_carriers)} packet inputs"
        )

    result = results[0]
    _require_descriptor_result_type_operand(source_op, result)
    actual_result_carrier = (result.reg_alts[0].reg_class, result.unit_count)
    if result_carrier is not None and actual_result_carrier != result_carrier:
        raise ValueError(
            f"{source_op.name}: {subject} descriptor '{descriptor.key}' result "
            "does not use the materializer carrier"
        )
    for operand, input_carrier in zip(inputs, input_carriers, strict=True):
        if input_carrier is None:
            continue
        reg_class, unit_count = input_carrier
        accepts_carrier = operand.unit_count == unit_count and any(
            alternative.reg_class == reg_class
            and RegClassAltFlag.IMMEDIATE not in alternative.flags
            for alternative in operand.reg_alts
        )
        if not accepts_carrier:
            raise ValueError(
                f"{source_op.name}: {subject} descriptor '{descriptor.key}' "
                f"operand '{operand.field_name}' does not accept the "
                "materializer carrier"
            )

    if bound_immediate is None:
        unbound_immediates = tuple(
            immediate.field_name
            for immediate in descriptor.immediates
            if not _immediate_has_default(immediate)
        )
    else:
        immediate = _require_immediate(
            descriptor,
            bound_immediate,
            f"{subject} immediate",
        )
        if immediate.kind not in (ImmediateKind.SIGNED, ImmediateKind.UNSIGNED):
            raise ValueError(
                f"{source_op.name}: {subject} descriptor '{descriptor.key}' "
                f"immediate '{bound_immediate}' must be an integer"
            )
        unbound_immediates = tuple(
            other.field_name
            for other in descriptor.immediates
            if other.field_name != bound_immediate and not _immediate_has_default(other)
        )
    if unbound_immediates:
        raise ValueError(
            f"{source_op.name}: {subject} descriptor '{descriptor.key}' has "
            "additional required immediates"
        )
    return actual_result_carrier


def _validate_byte_offset_materializer(
    source_op: Op,
    materializer: SourceMemoryByteOffsetMaterializer,
) -> None:
    carrier = _validate_materializer_descriptor(
        source_op,
        materializer.const_i64,
        subject="source-memory byte-offset constant",
        op_kind=DescriptorOpKind.CONST,
        input_carriers=(),
        bound_immediate=materializer.const_i64_immediate,
    )
    for subject, descriptor in (
        ("source-memory byte-offset add", materializer.add_i64),
        ("source-memory byte-offset multiply", materializer.mul_i64),
        ("source-memory byte-offset shift", materializer.shl_i64),
    ):
        if descriptor is not None:
            _validate_materializer_descriptor(
                source_op,
                descriptor,
                subject=subject,
                op_kind=DescriptorOpKind.OP,
                input_carriers=(carrier, carrier),
                result_carrier=carrier,
            )


def _validate_address_materializer(
    source_op: Op,
    materializer: SourceMemoryAddressMaterializer,
) -> None:
    carrier = _validate_materializer_descriptor(
        source_op,
        materializer.const_coordinate,
        subject="source-memory address-coordinate constant",
        op_kind=DescriptorOpKind.CONST,
        input_carriers=(),
        bound_immediate=materializer.const_coordinate_immediate,
    )
    for subject, descriptor in (
        ("source-memory address-coordinate add", materializer.add_coordinate),
        (
            "source-memory address-coordinate multiply",
            materializer.mul_coordinate,
        ),
        ("source-memory address-coordinate shift", materializer.shl_coordinate),
    ):
        if descriptor is not None:
            _validate_materializer_descriptor(
                source_op,
                descriptor,
                subject=subject,
                op_kind=DescriptorOpKind.OP,
                input_carriers=(carrier, carrier),
                result_carrier=carrier,
            )

    index_input_carrier = None
    if materializer.index_to_coordinate_input is not None:
        index_input_carrier = _validate_materializer_descriptor(
            source_op,
            materializer.index_to_coordinate_input,
            subject="source-memory address-coordinate index-input conversion",
            op_kind=DescriptorOpKind.OP,
            input_carriers=(None,),
        )
    if materializer.index_to_coordinate is not None:
        _validate_materializer_descriptor(
            source_op,
            materializer.index_to_coordinate,
            subject="source-memory address-coordinate index conversion",
            op_kind=DescriptorOpKind.OP,
            input_carriers=(index_input_carrier,),
            result_carrier=carrier,
        )
    elif index_input_carrier is not None:
        raise ValueError(
            f"{source_op.name}: source-memory address-coordinate input "
            "conversion needs a final index conversion descriptor"
        )
    _validate_materializer_descriptor(
        source_op,
        materializer.address,
        subject="source-memory address",
        op_kind=DescriptorOpKind.OP,
        input_carriers=(None, carrier),
    )
