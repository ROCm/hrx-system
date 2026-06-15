# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-memory constraints attached to generated descriptor emits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import Op
from loom.target.contracts.guards import GuardDiagnostic
from loom.target.low_descriptors import Descriptor

_MEMORY_SPACE_NAMES = frozenset(
    (
        "unknown",
        "global",
        "workgroup",
        "private",
        "constant",
        "host",
        "descriptor",
        "generic",
    )
)

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_U32_MAX = 2**32 - 1
_U8_MAX = 2**8 - 1


@unique
class SourceMemoryOperation(Enum):
    """Target-independent source memory operation kind."""

    LOAD = "load"
    STORE = "store"
    PREFETCH = "prefetch"
    ATOMIC_REDUCE = "atomic_reduce"
    ATOMIC_RMW = "atomic_rmw"
    ATOMIC_CMPXCHG = "atomic_cmpxchg"


@unique
class SourceMemoryDynamicIndexSource(Enum):
    """Source provenance required for a dynamic memory index term."""

    NONE = "none"
    VALUE = "value"
    WORKITEM_ID = "workitem_id"
    WORKGROUP_ID = "workgroup_id"


@unique
class SourceMemoryRootKind(Enum):
    """Source provenance required for the root memory value."""

    ANY = "any"
    BLOCK_ARGUMENT = "block_argument"


@dataclass(frozen=True, slots=True)
class SourceMemoryByteOffsetMaterializer:
    """Low descriptors used to materialize a dynamic byte offset value."""

    const_i64: Descriptor
    add_i64: Descriptor
    mul_i64: Descriptor
    shl_i64: Descriptor


@dataclass(frozen=True, slots=True)
class SourceMemoryConstraint:
    """Target-independent memory-access shape required by an emit row."""

    operation: SourceMemoryOperation
    root_kind: SourceMemoryRootKind
    memory_spaces: tuple[str, ...]
    element_byte_count: int
    vector_lane_count: int
    vector_lane_byte_stride: int
    static_byte_offset_minimum: int
    static_byte_offset_maximum: int
    minimum_alignment: int = 0
    dynamic_term_count: int | None = 0
    dynamic_index_source: SourceMemoryDynamicIndexSource = (
        SourceMemoryDynamicIndexSource.NONE
    )
    dynamic_byte_stride: int | None = 0
    allow_dynamic_stride_values: bool = False
    dynamic_offset_unsigned_bit_count: int = 0
    dynamic_offset_diagnostic: GuardDiagnostic | None = None
    cache_policy_build_flags: int = 0
    diagnostic: GuardDiagnostic | None = None

    def __init__(
        self,
        *,
        operation: SourceMemoryOperation,
        root_kind: SourceMemoryRootKind = SourceMemoryRootKind.ANY,
        memory_spaces: Sequence[str],
        element_byte_count: int,
        vector_lane_count: int,
        vector_lane_byte_stride: int,
        static_byte_offset: int | None = None,
        static_byte_offset_minimum: int | None = None,
        static_byte_offset_maximum: int | None = None,
        minimum_alignment: int = 0,
        dynamic_term_count: int | None = 0,
        dynamic_index_source: SourceMemoryDynamicIndexSource = (
            SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride: int | None = 0,
        allow_dynamic_stride_values: bool = False,
        dynamic_offset_unsigned_bit_count: int = 0,
        dynamic_offset_diagnostic: GuardDiagnostic | None = None,
        cache_policy_build_flags: int = 0,
        diagnostic: GuardDiagnostic | None = None,
    ) -> None:
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "root_kind", root_kind)
        object.__setattr__(self, "memory_spaces", tuple(memory_spaces))
        object.__setattr__(self, "element_byte_count", element_byte_count)
        object.__setattr__(self, "vector_lane_count", vector_lane_count)
        object.__setattr__(self, "vector_lane_byte_stride", vector_lane_byte_stride)
        minimum, maximum = _resolve_static_byte_offset_range(
            static_byte_offset,
            static_byte_offset_minimum,
            static_byte_offset_maximum,
        )
        object.__setattr__(self, "static_byte_offset_minimum", minimum)
        object.__setattr__(self, "static_byte_offset_maximum", maximum)
        object.__setattr__(self, "minimum_alignment", minimum_alignment)
        object.__setattr__(self, "dynamic_term_count", dynamic_term_count)
        object.__setattr__(self, "dynamic_index_source", dynamic_index_source)
        object.__setattr__(self, "dynamic_byte_stride", dynamic_byte_stride)
        object.__setattr__(
            self,
            "allow_dynamic_stride_values",
            allow_dynamic_stride_values,
        )
        object.__setattr__(
            self,
            "dynamic_offset_unsigned_bit_count",
            dynamic_offset_unsigned_bit_count,
        )
        object.__setattr__(
            self,
            "dynamic_offset_diagnostic",
            dynamic_offset_diagnostic,
        )
        object.__setattr__(self, "cache_policy_build_flags", cache_policy_build_flags)
        object.__setattr__(self, "diagnostic", diagnostic)
        self._validate_shape()

    def _validate_shape(self) -> None:
        if not isinstance(self.root_kind, SourceMemoryRootKind):
            raise ValueError("source memory root kind must be a SourceMemoryRootKind")
        if not self.memory_spaces:
            raise ValueError("source memory constraint needs a memory space")
        for memory_space in self.memory_spaces:
            if memory_space not in _MEMORY_SPACE_NAMES:
                raise ValueError(f"unknown source memory space '{memory_space}'")
        if not 0 < self.element_byte_count <= _U32_MAX:
            raise ValueError(
                "source memory element byte count must fit in a positive u32"
            )
        if not 0 < self.vector_lane_count <= _U32_MAX:
            raise ValueError(
                "source memory vector lane count must fit in a positive u32"
            )
        if not _I64_MIN <= self.vector_lane_byte_stride <= _I64_MAX:
            raise ValueError("source memory vector lane byte stride must fit in i64")
        if self.vector_lane_byte_stride == 0:
            raise ValueError("source memory vector lane byte stride must be non-zero")
        if not _I64_MIN <= self.static_byte_offset_minimum <= _I64_MAX:
            raise ValueError("source memory minimum static byte offset must fit in i64")
        if not _I64_MIN <= self.static_byte_offset_maximum <= _I64_MAX:
            raise ValueError("source memory maximum static byte offset must fit in i64")
        if self.static_byte_offset_minimum > self.static_byte_offset_maximum:
            raise ValueError("source memory static byte offset range is empty")
        if not 0 <= self.minimum_alignment <= _U32_MAX:
            raise ValueError("source memory minimum alignment must fit in u32")
        if self.minimum_alignment != 0 and (
            self.minimum_alignment & (self.minimum_alignment - 1)
        ):
            raise ValueError("source memory minimum alignment must be a power of two")
        dynamic_term_count = self.dynamic_term_count
        if dynamic_term_count is None:
            if self.dynamic_index_source != SourceMemoryDynamicIndexSource.NONE:
                raise ValueError(
                    "source memory with any dynamic term count cannot require "
                    "a dynamic index source"
                )
            if self.dynamic_byte_stride != 0:
                raise ValueError(
                    "source memory with any dynamic term count cannot require "
                    "a dynamic stride"
                )
        elif not 0 <= dynamic_term_count < _U8_MAX:
            raise ValueError("source memory dynamic term count must be non-negative")
        elif dynamic_term_count == 0:
            if self.dynamic_index_source != SourceMemoryDynamicIndexSource.NONE:
                raise ValueError("static source memory cannot require a dynamic index")
            if self.dynamic_byte_stride != 0:
                raise ValueError("static source memory cannot require a dynamic stride")
        elif self.dynamic_index_source == SourceMemoryDynamicIndexSource.NONE:
            raise ValueError("dynamic source memory needs an index source")
        elif self.dynamic_byte_stride == 0:
            raise ValueError("dynamic source memory stride must be non-zero")
        if self.allow_dynamic_stride_values and (
            dynamic_term_count is None or dynamic_term_count == 0
        ):
            raise ValueError(
                "dynamic source memory stride values require a fixed nonzero "
                "dynamic term count"
            )
        if self.dynamic_byte_stride is not None and not (
            _I64_MIN <= self.dynamic_byte_stride <= _I64_MAX
        ):
            raise ValueError("source memory dynamic byte stride must fit in i64")
        if not 0 <= self.dynamic_offset_unsigned_bit_count <= 64:
            raise ValueError(
                "source memory dynamic offset unsigned bit count must fit in u8"
            )
        if not 0 <= self.cache_policy_build_flags <= _U32_MAX:
            raise ValueError("source memory cache policy flags must fit in u32")

    def validate(self, source_op: Op) -> None:
        del source_op


def _resolve_static_byte_offset_range(
    exact: int | None,
    minimum: int | None,
    maximum: int | None,
) -> tuple[int, int]:
    if exact is not None:
        if minimum is not None or maximum is not None:
            raise ValueError(
                "source memory static byte offset cannot mix exact and range"
            )
        return exact, exact
    if minimum is None or maximum is None:
        raise ValueError("source memory static byte offset needs exact or range")
    return minimum, maximum
