# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor immediate projections for target contract rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

from loom.dsl import ATTR_TYPE_ENUM, ATTR_TYPE_I64, ATTR_TYPE_I64_ARRAY, Op
from loom.target.contracts.descriptors import _require_immediate
from loom.target.contracts.source import _require_attr, _require_value
from loom.target.low_descriptors import Descriptor, ImmediateKind


@unique
class AttrProjectKind(Enum):
    """Projection from source op attributes to descriptor immediates."""

    DIRECT = "direct"
    ENUM_ORDINAL = "enum_ordinal"
    I64_ARRAY_ELEMENT = "i64_array_element"
    I64_ARRAY_PACK_ELEMENTS = "i64_array_pack_elements"
    I64_ATTRS_PACK_CONSECUTIVE = "i64_attrs_pack_consecutive"
    I64_LOW_BIT_MASK = "i64_low_bit_mask"
    I64_SHIFTED_LOW_BIT_MASK = "i64_shifted_low_bit_mask"
    I64_SHIFTED_LOW_BIT_CLEAR_MASK = "i64_shifted_low_bit_clear_mask"
    I64_LITERAL_MINUS_ATTR = "i64_literal_minus_attr"
    I64_LITERAL_MINUS_ATTRS = "i64_literal_minus_attrs"
    EXPAND_LANE_I64_ARRAY_TO_BYTE_LANES = "expand_lane_i64_array_to_byte_lanes"


@unique
class ValueProjectKind(Enum):
    """Projection from source value facts to descriptor immediates."""

    EXACT_I64 = "exact_i64"
    EXACT_I64_NEGATE = "exact_i64_negate"
    EXACT_I64_LOG2 = "exact_i64_log2"
    EXACT_I64_MINUS_ONE = "exact_i64_minus_one"
    U32_DIVISOR_MAGIC_MULTIPLIER = "u32_divisor_magic_multiplier"
    U32_DIVISOR_MAGIC_SHIFT = "u32_divisor_magic_shift"
    I32_AS_U32_BITS = "i32_as_u32_bits"
    F64_AS_F16_BITS = "f64_as_f16_bits"
    F64_AS_BF16_BITS = "f64_as_bf16_bits"
    F64_AS_F32_BITS = "f64_as_f32_bits"
    F64_AS_F64_BITS = "f64_as_f64_bits"


@unique
class SourceMemoryProjectKind(Enum):
    """Projection from a selected source-memory access to descriptor immediates."""

    STATIC_BYTE_OFFSET = "static_byte_offset"
    STATIC_BYTE_OFFSET_QUOTIENT = "static_byte_offset_quotient"
    STATIC_BYTE_OFFSET_REMAINDER = "static_byte_offset_remainder"
    DYNAMIC_BYTE_STRIDE = "dynamic_byte_stride"


@unique
class SourceOpProjectKind(Enum):
    """Projection from source op state to descriptor immediates."""

    INSTANCE_FLAGS = "instance_flags"


@dataclass(frozen=True, slots=True)
class AttrProject:
    """Descriptor immediate projection from a source attribute."""

    kind: AttrProjectKind
    source_attr: str
    other_source_attr: str = ""
    literal_i64: int = 0
    element: int | None = None
    count: int | None = None
    bit_width: int | None = None
    target_bit_offset: int = 0
    source_lane_count: int | None = None
    bytes_per_lane: int | None = None
    target_names: tuple[str, ...] = ()

    @classmethod
    def direct(cls, source_attr: str) -> Self:
        return cls(kind=AttrProjectKind.DIRECT, source_attr=source_attr)

    @classmethod
    def enum_ordinal(cls, source_attr: str) -> Self:
        return cls(kind=AttrProjectKind.ENUM_ORDINAL, source_attr=source_attr)

    @classmethod
    def i64_array_element(
        cls,
        source_attr: str,
        *,
        element: int,
        target_bit_offset: int = 0,
    ) -> Self:
        return cls(
            kind=AttrProjectKind.I64_ARRAY_ELEMENT,
            source_attr=source_attr,
            element=element,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def i64_array_pack_elements(
        cls,
        source_attr: str,
        *,
        element: int,
        count: int,
        bit_width: int,
        target_bit_offset: int = 0,
    ) -> Self:
        return cls(
            kind=AttrProjectKind.I64_ARRAY_PACK_ELEMENTS,
            source_attr=source_attr,
            element=element,
            count=count,
            bit_width=bit_width,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def i64_attrs_pack_consecutive(
        cls,
        source_attr: str,
        *,
        count: int,
        bit_width: int,
        target_bit_offset: int = 0,
    ) -> Self:
        return cls(
            kind=AttrProjectKind.I64_ATTRS_PACK_CONSECUTIVE,
            source_attr=source_attr,
            count=count,
            bit_width=bit_width,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def i64_low_bit_mask(cls, width_attr: str) -> Self:
        return cls(kind=AttrProjectKind.I64_LOW_BIT_MASK, source_attr=width_attr)

    @classmethod
    def i64_shifted_low_bit_mask(cls, width_attr: str, *, offset_attr: str) -> Self:
        return cls(
            kind=AttrProjectKind.I64_SHIFTED_LOW_BIT_MASK,
            source_attr=width_attr,
            other_source_attr=offset_attr,
        )

    @classmethod
    def i64_shifted_low_bit_clear_mask(
        cls,
        width_attr: str,
        *,
        offset_attr: str,
    ) -> Self:
        return cls(
            kind=AttrProjectKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK,
            source_attr=width_attr,
            other_source_attr=offset_attr,
        )

    @classmethod
    def i64_literal_minus_attr(cls, source_attr: str, *, literal: int) -> Self:
        return cls(
            kind=AttrProjectKind.I64_LITERAL_MINUS_ATTR,
            source_attr=source_attr,
            literal_i64=literal,
        )

    @classmethod
    def i64_literal_minus_attrs(
        cls,
        source_attr: str,
        *,
        other_source_attr: str,
        literal: int,
    ) -> Self:
        return cls(
            kind=AttrProjectKind.I64_LITERAL_MINUS_ATTRS,
            source_attr=source_attr,
            other_source_attr=other_source_attr,
            literal_i64=literal,
        )

    @classmethod
    def expand_lane_i64_array_to_byte_lanes(
        cls,
        *,
        source_attr: str,
        source_lane_count: int,
        bytes_per_lane: int,
        target_names: Sequence[str],
    ) -> Self:
        return cls(
            kind=AttrProjectKind.EXPAND_LANE_I64_ARRAY_TO_BYTE_LANES,
            source_attr=source_attr,
            source_lane_count=source_lane_count,
            bytes_per_lane=bytes_per_lane,
            target_names=tuple(target_names),
        )

    def __post_init__(self) -> None:
        if not self.source_attr:
            raise ValueError(f"{self.kind.value} projection requires a source attr")
        shifted_mask_kinds = (
            AttrProjectKind.I64_SHIFTED_LOW_BIT_MASK,
            AttrProjectKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK,
        )
        two_attr_kinds = (
            *shifted_mask_kinds,
            AttrProjectKind.I64_LITERAL_MINUS_ATTRS,
        )
        if self.kind in two_attr_kinds and not self.other_source_attr:
            raise ValueError(
                f"{self.kind.value} projection requires another source attr"
            )
        if self.kind not in two_attr_kinds and self.other_source_attr:
            raise ValueError(
                f"{self.kind.value} projection must not name another source attr"
            )
        literal_kinds = (
            AttrProjectKind.I64_LITERAL_MINUS_ATTR,
            AttrProjectKind.I64_LITERAL_MINUS_ATTRS,
        )
        if self.kind not in literal_kinds and self.literal_i64 != 0:
            raise ValueError(f"{self.kind.value} projection must not name a literal")
        if self.element is not None and self.element < 0:
            raise ValueError(f"{self.kind.value} element must be non-negative")
        if self.count is not None and self.count <= 0:
            raise ValueError(f"{self.kind.value} count must be positive")
        if self.bit_width is not None and self.bit_width <= 0:
            raise ValueError(f"{self.kind.value} bit width must be positive")
        if self.target_bit_offset < 0:
            raise ValueError(
                f"{self.kind.value} target bit offset must be non-negative"
            )
        mask_kinds = (
            AttrProjectKind.I64_LOW_BIT_MASK,
            AttrProjectKind.I64_SHIFTED_LOW_BIT_MASK,
            AttrProjectKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK,
        )
        if self.kind in (*mask_kinds, *literal_kinds) and self.target_bit_offset != 0:
            raise ValueError(
                f"{self.kind.value} projection must not use target bit offset"
            )
        if self.count is not None and self.bit_width is not None:
            packed_bit_count = self.count * self.bit_width
            if packed_bit_count + self.target_bit_offset > 63:
                raise ValueError(
                    f"{self.kind.value} packed bit count must fit in signed i64"
                )
        if self.source_lane_count is not None and self.source_lane_count < 0:
            raise ValueError(
                f"{self.kind.value} source lane count must be non-negative"
            )
        if self.bytes_per_lane is not None and self.bytes_per_lane <= 0:
            raise ValueError(f"{self.kind.value} bytes per lane must be positive")

    def validate(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
    ) -> None:
        subject = f"immediate projection {self.kind.value}"
        attr = _require_attr(source_op, self.source_attr, subject)
        if self.kind == AttrProjectKind.DIRECT:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            _require_immediate(descriptor, bound_immediate_name, subject)
            return
        if self.kind == AttrProjectKind.ENUM_ORDINAL:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            immediate = _require_immediate(descriptor, bound_immediate_name, subject)
            if attr.attr_type != ATTR_TYPE_ENUM:
                raise ValueError(
                    f"{source_op.name}: {subject} source attr "
                    f"'{self.source_attr}' must be an enum attr"
                )
            if immediate.kind != ImmediateKind.ENUM:
                raise ValueError(
                    f"{source_op.name}: {subject} descriptor immediate "
                    f"'{bound_immediate_name}' must be an enum immediate"
                )
            return
        if self.kind == AttrProjectKind.I64_ATTRS_PACK_CONSECUTIVE:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            _require_immediate(descriptor, bound_immediate_name, subject)
            if self.count is None or self.bit_width is None:
                raise ValueError(f"{source_op.name}: {subject} needs count/bit_width")
            if attr.attr_type != ATTR_TYPE_I64:
                raise ValueError(
                    f"{source_op.name}: {subject} source attr '{self.source_attr}' "
                    "must be an i64 attr"
                )
            attr_index = source_op.attrs.index(attr)
            if attr_index + self.count > len(source_op.attrs):
                raise ValueError(
                    f"{source_op.name}: {subject} source attr '{self.source_attr}' "
                    "does not have enough following attrs"
                )
            for element_attr in source_op.attrs[attr_index : attr_index + self.count]:
                if element_attr.attr_type != ATTR_TYPE_I64:
                    raise ValueError(
                        f"{source_op.name}: {subject} source attr "
                        f"'{element_attr.name}' must be an i64 attr"
                    )
            return
        mask_kinds = (
            AttrProjectKind.I64_LOW_BIT_MASK,
            AttrProjectKind.I64_SHIFTED_LOW_BIT_MASK,
            AttrProjectKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK,
        )
        if self.kind in mask_kinds:
            self._validate_i64_attr_projection(
                source_op,
                descriptor,
                bound_immediate_name,
                subject,
                require_unsigned_immediate=True,
            )
            return
        literal_minus_kinds = (
            AttrProjectKind.I64_LITERAL_MINUS_ATTR,
            AttrProjectKind.I64_LITERAL_MINUS_ATTRS,
        )
        if self.kind in literal_minus_kinds:
            self._validate_i64_attr_projection(
                source_op,
                descriptor,
                bound_immediate_name,
                subject,
            )
            return
        if attr.attr_type != ATTR_TYPE_I64_ARRAY:
            raise ValueError(
                f"{source_op.name}: {subject} source attr '{self.source_attr}' "
                "must be an i64_array attr"
            )
        if self.kind == AttrProjectKind.I64_ARRAY_ELEMENT:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            _require_immediate(descriptor, bound_immediate_name, subject)
            if self.element is None:
                raise ValueError(f"{source_op.name}: {subject} needs an element")
            return
        if self.kind == AttrProjectKind.I64_ARRAY_PACK_ELEMENTS:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            _require_immediate(descriptor, bound_immediate_name, subject)
            if self.element is None or self.count is None or self.bit_width is None:
                raise ValueError(
                    f"{source_op.name}: {subject} needs element/count/bit_width"
                )
            return
        if bound_immediate_name is not None:
            raise ValueError(
                f"{source_op.name}: {subject} expands multiple immediates and "
                "must not be bound through a single immediate name"
            )
        if self.source_lane_count is None or self.bytes_per_lane is None:
            raise ValueError(
                f"{source_op.name}: {subject} needs source_lane_count and "
                "bytes_per_lane"
            )
        expected_count = self.source_lane_count * self.bytes_per_lane
        if len(self.target_names) != expected_count:
            raise ValueError(
                f"{source_op.name}: {subject} produces {expected_count} "
                f"immediates, got {len(self.target_names)}"
            )
        for name in self.target_names:
            _require_immediate(descriptor, name, subject)

    def _validate_i64_attr_projection(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
        subject: str,
        *,
        require_unsigned_immediate: bool = False,
    ) -> None:
        if bound_immediate_name is None:
            raise ValueError(
                f"{source_op.name}: {subject} must bind one descriptor immediate"
            )
        immediate = _require_immediate(descriptor, bound_immediate_name, subject)
        if require_unsigned_immediate and immediate.kind != ImmediateKind.UNSIGNED:
            raise ValueError(
                f"{source_op.name}: {subject} descriptor immediate "
                f"'{bound_immediate_name}' must be an unsigned immediate"
            )
        attr = _require_attr(source_op, self.source_attr, subject)
        if attr.attr_type != ATTR_TYPE_I64:
            raise ValueError(
                f"{source_op.name}: {subject} source attr '{self.source_attr}' "
                "must be an i64 attr"
            )
        if not self.other_source_attr:
            return
        other_attr = _require_attr(source_op, self.other_source_attr, subject)
        if other_attr.attr_type != ATTR_TYPE_I64:
            raise ValueError(
                f"{source_op.name}: {subject} source attr "
                f"'{self.other_source_attr}' must be an i64 attr"
            )


@dataclass(frozen=True, slots=True)
class SourceOpProject:
    """Descriptor immediate projection from source operation state."""

    kind: SourceOpProjectKind

    @classmethod
    def instance_flags(cls) -> Self:
        return cls(kind=SourceOpProjectKind.INSTANCE_FLAGS)

    def validate(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
    ) -> None:
        subject = f"immediate projection {self.kind.value}"
        if self.kind != SourceOpProjectKind.INSTANCE_FLAGS:
            raise ValueError(
                f"{source_op.name}: {subject} is not representable by "
                "generated lower rules yet"
            )
        if bound_immediate_name is None:
            raise ValueError(
                f"{source_op.name}: {subject} must bind one descriptor immediate"
            )
        immediate = _require_immediate(descriptor, bound_immediate_name, subject)
        if immediate.kind != ImmediateKind.UNSIGNED:
            raise ValueError(
                f"{source_op.name}: {subject} descriptor immediate "
                f"'{bound_immediate_name}' must be an unsigned immediate"
            )


@dataclass(frozen=True, slots=True)
class ValueProject:
    """Descriptor immediate projection from source value facts."""

    kind: ValueProjectKind
    source_value: str
    target_bit_offset: int = 0

    @classmethod
    def exact_i64(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def exact_i64_negate(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64_NEGATE,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def exact_i64_log2(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64_LOG2,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def exact_i64_minus_one(
        cls, source_value: str, *, target_bit_offset: int = 0
    ) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64_MINUS_ONE,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def u32_divisor_magic_multiplier(
        cls, source_value: str, *, target_bit_offset: int = 0
    ) -> Self:
        return cls(
            kind=ValueProjectKind.U32_DIVISOR_MAGIC_MULTIPLIER,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def u32_divisor_magic_shift(
        cls, source_value: str, *, target_bit_offset: int = 0
    ) -> Self:
        return cls(
            kind=ValueProjectKind.U32_DIVISOR_MAGIC_SHIFT,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def i32_as_u32_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.I32_AS_U32_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_f16_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_F16_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_bf16_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_BF16_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_f32_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_F32_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_f64_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_F64_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    def __post_init__(self) -> None:
        if not self.source_value:
            raise ValueError(f"{self.kind.value} projection requires a source value")
        if self.target_bit_offset < 0:
            raise ValueError(
                f"{self.kind.value} target bit offset must be non-negative"
            )

    def validate(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
    ) -> None:
        subject = f"immediate projection {self.kind.value}"
        _require_value(source_op, self.source_value, subject)
        if bound_immediate_name is None:
            raise ValueError(
                f"{source_op.name}: {subject} must bind one descriptor immediate"
            )
        _require_immediate(descriptor, bound_immediate_name, subject)


@dataclass(frozen=True, slots=True)
class SourceMemoryProject:
    """Descriptor immediate projection from the selected source-memory access."""

    kind: SourceMemoryProjectKind
    dynamic_term_index: int = 0
    divisor: int = 1

    @classmethod
    def static_byte_offset(cls) -> Self:
        return cls(kind=SourceMemoryProjectKind.STATIC_BYTE_OFFSET)

    @classmethod
    def static_byte_offset_quotient(cls, divisor: int) -> Self:
        return cls(
            kind=SourceMemoryProjectKind.STATIC_BYTE_OFFSET_QUOTIENT,
            divisor=divisor,
        )

    @classmethod
    def static_byte_offset_remainder(cls, divisor: int) -> Self:
        return cls(
            kind=SourceMemoryProjectKind.STATIC_BYTE_OFFSET_REMAINDER,
            divisor=divisor,
        )

    @classmethod
    def dynamic_byte_stride(cls, *, term: int = 0) -> Self:
        return cls(
            kind=SourceMemoryProjectKind.DYNAMIC_BYTE_STRIDE,
            dynamic_term_index=term,
        )

    def __post_init__(self) -> None:
        if self.dynamic_term_index < 0:
            raise ValueError(
                f"{self.kind.value} dynamic term index must be non-negative"
            )
        if (
            self.kind
            in (
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_QUOTIENT,
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_REMAINDER,
            )
            and self.dynamic_term_index != 0
        ):
            raise ValueError("static byte offset projections must not name a term")
        if self.divisor <= 0:
            raise ValueError(f"{self.kind.value} divisor must be positive")
        if (
            self.kind
            not in (
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_QUOTIENT,
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_REMAINDER,
            )
            and self.divisor != 1
        ):
            raise ValueError(f"{self.kind.value} projection must not set divisor")

    def validate(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
    ) -> None:
        subject = f"immediate projection {self.kind.value}"
        if bound_immediate_name is None:
            raise ValueError(
                f"{source_op.name}: {subject} must bind one descriptor immediate"
            )
        _require_immediate(descriptor, bound_immediate_name, subject)
