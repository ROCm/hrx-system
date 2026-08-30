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

from loom.dsl import MemoryAccessInterface, Op
from loom.target.contracts.guards import GuardDiagnostic
from loom.target.contracts.memory_spaces import MEMORY_SPACE_NAMES
from loom.target.low_descriptors import Descriptor

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
    ALLOCA = "alloca"


@unique
class SourceMemoryAddressLayout(Enum):
    """Address-layout classification required by a source-memory row."""

    ANY = "any"
    COMPACT_ROW_MAJOR = "compact_row_major"


@unique
class SourceMemoryAddressBase(Enum):
    """Source-memory value used as the base of a complete target address."""

    ROOT = "root"
    BASE_VIEW = "base_view"


@unique
class SourceMemoryAddressCoordinateType(Enum):
    """Semantic source type carried by a materialized address coordinate."""

    OFFSET = "offset"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class SourceMemoryByteOffsetMaterializer:
    """Low descriptors used to materialize a dynamic byte offset value."""

    const_i64: Descriptor
    add_i64: Descriptor
    mul_i64: Descriptor
    shl_i64: Descriptor | None
    const_i64_immediate: str = "value"


@dataclass(frozen=True, slots=True)
class SourceMemoryAddressMaterializer:
    """Low descriptors and policies for one complete source-memory address."""

    const_coordinate: Descriptor
    add_coordinate: Descriptor
    mul_coordinate: Descriptor
    address: Descriptor
    base: SourceMemoryAddressBase = SourceMemoryAddressBase.ROOT
    coordinate_type: SourceMemoryAddressCoordinateType = (
        SourceMemoryAddressCoordinateType.OFFSET
    )
    coordinate_unit_byte_count: int = 1
    coordinate_minimum: int = _I64_MIN
    coordinate_maximum: int = _I64_MAX
    shl_coordinate: Descriptor | None = None
    index_to_coordinate_input: Descriptor | None = None
    index_to_coordinate: Descriptor | None = None
    const_coordinate_immediate: str = "value"
    diagnostic: GuardDiagnostic | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base, SourceMemoryAddressBase):
            raise ValueError(
                "source-memory address base must be a SourceMemoryAddressBase"
            )
        if not isinstance(
            self.coordinate_type,
            SourceMemoryAddressCoordinateType,
        ):
            raise ValueError(
                "source-memory address coordinate type must be a "
                "SourceMemoryAddressCoordinateType"
            )
        if not 0 < self.coordinate_unit_byte_count <= _U32_MAX:
            raise ValueError(
                "source-memory address coordinate unit byte count must fit in a "
                "positive u32"
            )
        if not _I64_MIN <= self.coordinate_minimum <= _I64_MAX:
            raise ValueError("source-memory address minimum coordinate must fit in i64")
        if not _I64_MIN <= self.coordinate_maximum <= _I64_MAX:
            raise ValueError("source-memory address maximum coordinate must fit in i64")
        if self.coordinate_minimum > self.coordinate_maximum:
            raise ValueError("source-memory address coordinate range is empty")
        if (
            self.coordinate_type == SourceMemoryAddressCoordinateType.OFFSET
            and self.coordinate_unit_byte_count != 1
        ):
            raise ValueError("offset-coordinate source-memory addresses use byte units")
        if self.coordinate_type == SourceMemoryAddressCoordinateType.INDEX and (
            self.index_to_coordinate_input is not None
            or self.index_to_coordinate is not None
        ):
            raise ValueError(
                "index-coordinate source-memory addresses use the mapped index "
                "carrier directly"
            )
        if (
            self.coordinate_type == SourceMemoryAddressCoordinateType.OFFSET
            and self.index_to_coordinate is None
        ):
            raise ValueError(
                "offset-coordinate source-memory addresses need an index "
                "conversion descriptor"
            )


@dataclass(frozen=True, slots=True)
class SourceMemoryConstraint:
    """Target-independent memory-access shape required by an emit row."""

    operation: SourceMemoryOperation
    root_kind: SourceMemoryRootKind
    address_layout: SourceMemoryAddressLayout
    memory_spaces: tuple[str, ...]
    element_byte_count: int
    vector_lane_count: int
    vector_lane_byte_stride: int
    static_byte_offset_minimum: int
    static_byte_offset_maximum: int
    minimum_alignment: int = 0
    dynamic_term_count: int | None = 0
    dynamic_term_count_minimum: int = 0
    dynamic_view_base_term_count: int | None = None
    dynamic_index_source: SourceMemoryDynamicIndexSource = (
        SourceMemoryDynamicIndexSource.NONE
    )
    dynamic_byte_stride: int | None = 0
    allow_dynamic_stride_values: bool = False
    preserve_source_index: bool = False
    dynamic_offset_unsigned_bit_count: int = 0
    dynamic_offset_diagnostic: GuardDiagnostic | None = None
    address_layout_diagnostic: GuardDiagnostic | None = None
    cache_policy_build_flags: int = 0
    diagnostic: GuardDiagnostic | None = None

    def __init__(
        self,
        *,
        operation: SourceMemoryOperation,
        root_kind: SourceMemoryRootKind = SourceMemoryRootKind.ANY,
        address_layout: SourceMemoryAddressLayout = SourceMemoryAddressLayout.ANY,
        memory_spaces: Sequence[str],
        element_byte_count: int,
        vector_lane_count: int,
        vector_lane_byte_stride: int,
        static_byte_offset: int | None = None,
        static_byte_offset_minimum: int | None = None,
        static_byte_offset_maximum: int | None = None,
        minimum_alignment: int = 0,
        dynamic_term_count: int | None = 0,
        dynamic_term_count_minimum: int = 0,
        dynamic_view_base_term_count: int | None = None,
        dynamic_index_source: SourceMemoryDynamicIndexSource = (
            SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride: int | None = 0,
        allow_dynamic_stride_values: bool = False,
        preserve_source_index: bool = False,
        dynamic_offset_unsigned_bit_count: int = 0,
        dynamic_offset_diagnostic: GuardDiagnostic | None = None,
        address_layout_diagnostic: GuardDiagnostic | None = None,
        cache_policy_build_flags: int = 0,
        diagnostic: GuardDiagnostic | None = None,
    ) -> None:
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "root_kind", root_kind)
        object.__setattr__(self, "address_layout", address_layout)
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
        object.__setattr__(
            self,
            "dynamic_term_count_minimum",
            dynamic_term_count_minimum,
        )
        object.__setattr__(
            self,
            "dynamic_view_base_term_count",
            dynamic_view_base_term_count,
        )
        object.__setattr__(self, "dynamic_index_source", dynamic_index_source)
        object.__setattr__(self, "dynamic_byte_stride", dynamic_byte_stride)
        object.__setattr__(
            self,
            "allow_dynamic_stride_values",
            allow_dynamic_stride_values,
        )
        object.__setattr__(self, "preserve_source_index", preserve_source_index)
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
        object.__setattr__(
            self,
            "address_layout_diagnostic",
            address_layout_diagnostic,
        )
        object.__setattr__(self, "cache_policy_build_flags", cache_policy_build_flags)
        object.__setattr__(self, "diagnostic", diagnostic)
        self._validate_shape()

    def _validate_shape(self) -> None:
        if not isinstance(self.root_kind, SourceMemoryRootKind):
            raise ValueError("source memory root kind must be a SourceMemoryRootKind")
        if not isinstance(self.address_layout, SourceMemoryAddressLayout):
            raise ValueError(
                "source memory address layout must be a SourceMemoryAddressLayout"
            )
        if (
            self.address_layout == SourceMemoryAddressLayout.ANY
            and self.address_layout_diagnostic is not None
        ):
            raise ValueError(
                "unconstrained source memory cannot have an address-layout diagnostic"
            )
        if not self.memory_spaces:
            raise ValueError("source memory constraint needs a memory space")
        for memory_space in self.memory_spaces:
            if memory_space not in MEMORY_SPACE_NAMES:
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
            if not 0 <= self.dynamic_term_count_minimum < _U8_MAX:
                raise ValueError(
                    "source memory minimum dynamic term count must fit in u8"
                )
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
        elif self.dynamic_term_count_minimum != 0:
            raise ValueError(
                "source memory with a fixed dynamic term count cannot also "
                "require a minimum"
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
        if self.dynamic_view_base_term_count is not None and not (
            0 <= self.dynamic_view_base_term_count < _U8_MAX
        ):
            raise ValueError(
                "source memory dynamic view-base term count must fit in u8"
            )
        if self.allow_dynamic_stride_values and (
            dynamic_term_count == 0
            or (dynamic_term_count is None and self.dynamic_term_count_minimum == 0)
        ):
            raise ValueError(
                "dynamic source memory stride values require at least one dynamic term"
            )
        if self.preserve_source_index:
            if dynamic_term_count == 0 or (
                dynamic_term_count is None and self.dynamic_term_count_minimum == 0
            ):
                raise ValueError(
                    "source-index preservation requires dynamic source memory"
                )
            if self.dynamic_view_base_term_count != 0:
                raise ValueError(
                    "source-index preservation requires zero dynamic view-base terms"
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
        if not self.preserve_source_index:
            return
        memory_access = next(
            (
                interface
                for interface in source_op.interfaces
                if isinstance(interface, MemoryAccessInterface)
            ),
            None,
        )
        if memory_access is None or memory_access.indices is None:
            raise ValueError(
                f"{source_op.name}: source-index preservation requires a "
                "memory-access index operand"
            )


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
