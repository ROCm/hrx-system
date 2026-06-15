# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract case records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loom.dsl import Op
from loom.target.contracts.descriptors import _require_descriptor
from loom.target.contracts.emits import EmitDescriptorOp
from loom.target.contracts.guards import Guard
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.source import ValueRef
from loom.target.low_descriptors import Descriptor, DescriptorSet


@dataclass(frozen=True, slots=True)
class DescriptorRule:
    """Descriptor-rule contract case authored in Python."""

    source_op: Op
    descriptor: Descriptor
    guards: tuple[Guard, ...] = ()
    emit: tuple[EmitDescriptorOp, ...] = ()
    priority: int = 0

    def __init__(
        self,
        *,
        source_op: Op,
        descriptor: Descriptor,
        guards: Sequence[Guard] = (),
        emit: Sequence[EmitDescriptorOp] = (),
        priority: int = 0,
    ) -> None:
        object.__setattr__(self, "source_op", source_op)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "guards", tuple(guards))
        object.__setattr__(self, "emit", tuple(emit))
        object.__setattr__(self, "priority", priority)
        if priority < 0:
            raise ValueError("descriptor rule priority must be non-negative")

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
    DescriptorRule | ValueAliasRule | ValueElideRule | DescriptorMatrixRule
)
