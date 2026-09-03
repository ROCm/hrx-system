# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selection guard schema for target contract rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

from loom.dsl import (
    ATTR_TYPE_ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    EnumCase,
    Op,
)
from loom.error.target import ERR_TARGET_003
from loom.target.contracts.diagnostics import (
    DiagnosticRef,
    string_param,
    target_diagnostic,
)
from loom.target.contracts.memory_spaces import MEMORY_SPACE_NAMES
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source import (
    _require_attr,
    _require_operand,
    _require_value,
)
from loom.target.low_descriptors import Descriptor

_MAX_U32 = 0xFFFFFFFF


@unique
class GuardKind(Enum):
    """Selection guard kind for descriptor-rule contracts."""

    VALUE_TYPE = "value_type"
    ATTR_KIND = "attr_kind"
    ENUM_ATTR_EQUALS = "enum_attr_equals"
    I64_RANGE = "i64_range"
    DESCRIPTOR_AVAILABLE = "descriptor_available"
    VALUE_MATERIALIZABLE = "value_materializable"
    LOW_VALUE_REGISTER_CLASS = "low_value_register_class"
    LOW_VALUE_REGISTER_UNIT_COUNT = "low_value_register_unit_count"
    VALUE_STATIC_DIM0_MULTIPLE = "value_static_dim0_multiple"
    LOW_VALUE_REGISTER_UNIT_COUNT_EQ = "low_value_register_unit_count_eq"
    OPERAND_SEGMENT_COUNT = "operand_segment_count"
    I64_ARRAY_COUNT = "i64_array_count"
    I64_ARRAY_ELEMENT_RANGE = "i64_array_element_range"
    I64_ARRAY_ELEMENTS_RANGE = "i64_array_elements_range"
    VALUE_SIGNED_BIT_COUNT = "value_signed_bit_count"
    VALUE_UNSIGNED_BIT_COUNT = "value_unsigned_bit_count"
    VALUE_EXACT_I64 = "value_exact_i64"
    VALUE_EXACT_POWER_OF_TWO_I64 = "value_exact_power_of_two_i64"
    VALUE_U32_DIVISOR_MAGIC_IS_ADD = "value_u32_divisor_magic_is_add"
    VALUE_EXACT_FLOAT = "value_exact_float"
    VALUE_I64_RANGE = "value_i64_range"
    VALUE_I64_RANGE_LE = "value_i64_range_le"
    VALUE_I64_RANGE_GE = "value_i64_range_ge"
    VALUE_FLOAT_EQUALS = "value_float_equals"
    VALUE_STORAGE_ELEMENT_FORMAT = "value_storage_element_format"
    VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES = "value_packed_integer_payload_from_lanes"
    VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD = "value_packed_integer_lanes_from_payload"
    VALUE_NO_USES = "value_no_uses"
    INSTANCE_FLAGS_HAS_ALL = "instance_flags_has_all"
    VECTOR_EXTRACT_SHAPE = "vector_extract_shape"
    VALUE_STATIC_ELEMENT_COUNT_EQ = "value_static_element_count_eq"
    VALUE_MEMORY_SPACE = "value_memory_space"
    SOURCE_REPRESENTATION_GROUP = "source_representation_group"
    SOURCE_REPRESENTATION_CANDIDATE = "source_representation_candidate"


_LOW_VALUE_GUARD_KINDS = (
    GuardKind.LOW_VALUE_REGISTER_CLASS,
    GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT,
    GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
    GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
)


@dataclass(frozen=True, slots=True)
class GuardDiagnostic:
    """Authored structured diagnostic for a guard failure."""

    ref: DiagnosticRef | None = None
    subject_role: str = ""
    subject_name: str = ""
    constraint_key: str = ""

    def __post_init__(self) -> None:
        if self.ref is not None:
            return
        if not self.subject_role:
            raise ValueError("guard diagnostic subject kind must be non-empty")
        if not self.subject_name:
            raise ValueError("guard diagnostic subject name must be non-empty")
        if not self.constraint_key:
            raise ValueError("guard diagnostic constraint key must be non-empty")
        object.__setattr__(
            self,
            "ref",
            target_diagnostic(
                ERR_TARGET_003,
                string_param("subject_role", self.subject_role),
                string_param("subject_name", self.subject_name),
                string_param("constraint_key", self.constraint_key),
            ),
        )


@dataclass(frozen=True, slots=True)
class Guard:
    """Selection predicate evaluated before a contract row can match."""

    kind: GuardKind
    field: str
    other_field: str | None = None
    attr_field: str | None = None
    type_pattern: TypePattern | None = None
    attr_type: str | None = None
    enum_keyword: str | None = None
    count: int | None = None
    element: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    f64_value: float | None = None
    numeric_format_c_expression: str | None = None
    descriptor: Descriptor | None = None
    register_class: str | None = None
    materializer: str | None = None
    memory_spaces: tuple[str, ...] = ()
    diagnostic: GuardDiagnostic | None = None

    @classmethod
    def value_type(
        cls,
        field: str,
        type_pattern: TypePattern,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_TYPE,
            field=field,
            type_pattern=type_pattern,
            diagnostic=diagnostic,
        )

    @classmethod
    def enum_attr_equals(
        cls,
        field: str,
        enum_case: str | EnumCase,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        keyword = enum_case.keyword if isinstance(enum_case, EnumCase) else enum_case
        return cls(
            kind=GuardKind.ENUM_ATTR_EQUALS,
            field=field,
            enum_keyword=keyword,
            diagnostic=diagnostic,
        )

    @classmethod
    def attr_kind(
        cls,
        field: str,
        attr_type: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.ATTR_KIND,
            field=field,
            attr_type=attr_type,
            diagnostic=diagnostic,
        )

    @classmethod
    def i64_range(
        cls,
        field: str,
        minimum: int,
        maximum: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.I64_RANGE,
            field=field,
            minimum=minimum,
            maximum=maximum,
            diagnostic=diagnostic,
        )

    @classmethod
    def descriptor_available(
        cls,
        descriptor: Descriptor,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.DESCRIPTOR_AVAILABLE,
            field=descriptor.key,
            descriptor=descriptor,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_materializable(
        cls,
        field: str,
        materializer: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_MATERIALIZABLE,
            field=field,
            materializer=materializer,
            diagnostic=diagnostic,
        )

    @classmethod
    def low_value_register_class(
        cls,
        field: str,
        register_class: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.LOW_VALUE_REGISTER_CLASS,
            field=field,
            register_class=register_class,
            diagnostic=diagnostic,
        )

    @classmethod
    def low_value_register_unit_count(
        cls,
        field: str,
        count: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT,
            field=field,
            count=count,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_static_dim0_multiple(
        cls,
        field: str,
        multiple: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
            field=field,
            count=multiple,
            diagnostic=diagnostic,
        )

    @classmethod
    def low_value_register_unit_count_eq(
        cls,
        field: str,
        other_field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
            field=field,
            other_field=other_field,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_static_element_count_eq(
        cls,
        field: str,
        other_field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ,
            field=field,
            other_field=other_field,
            diagnostic=diagnostic,
        )

    @classmethod
    def operand_segment_count(cls, field: str, count: int) -> Self:
        return cls(kind=GuardKind.OPERAND_SEGMENT_COUNT, field=field, count=count)

    @classmethod
    def i64_array_count(cls, field: str, count: int) -> Self:
        return cls(kind=GuardKind.I64_ARRAY_COUNT, field=field, count=count)

    @classmethod
    def i64_array_element_range(
        cls,
        field: str,
        element: int,
        minimum: int,
        maximum: int,
    ) -> Self:
        return cls(
            kind=GuardKind.I64_ARRAY_ELEMENT_RANGE,
            field=field,
            element=element,
            minimum=minimum,
            maximum=maximum,
        )

    @classmethod
    def i64_array_elements_range(
        cls,
        field: str,
        minimum: int,
        maximum: int,
    ) -> Self:
        return cls(
            kind=GuardKind.I64_ARRAY_ELEMENTS_RANGE,
            field=field,
            minimum=minimum,
            maximum=maximum,
        )

    @classmethod
    def value_signed_bit_count(
        cls,
        field: str,
        bit_count: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_SIGNED_BIT_COUNT,
            field=field,
            count=bit_count,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_unsigned_bit_count(
        cls,
        field: str,
        bit_count: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_UNSIGNED_BIT_COUNT,
            field=field,
            count=bit_count,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_exact_i64(
        cls,
        field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_EXACT_I64,
            field=field,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_exact_power_of_two_i64(
        cls,
        field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_EXACT_POWER_OF_TWO_I64,
            field=field,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_u32_divisor_magic_is_add(
        cls,
        field: str,
        is_add: bool,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD,
            field=field,
            count=1 if is_add else 0,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_exact_float(
        cls,
        field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_EXACT_FLOAT,
            field=field,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_i64_range(
        cls,
        field: str,
        minimum: int,
        maximum: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_I64_RANGE,
            field=field,
            minimum=minimum,
            maximum=maximum,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_i64_range_le(
        cls,
        field: str,
        other_field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_I64_RANGE_LE,
            field=field,
            other_field=other_field,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_i64_range_ge(
        cls,
        field: str,
        other_field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_I64_RANGE_GE,
            field=field,
            other_field=other_field,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_float_equals(
        cls,
        field: str,
        value: float,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_FLOAT_EQUALS,
            field=field,
            f64_value=value,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_storage_element_format(
        cls,
        field: str,
        numeric_format_c_expression: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_STORAGE_ELEMENT_FORMAT,
            field=field,
            numeric_format_c_expression=numeric_format_c_expression,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_memory_space(
        cls,
        field: str,
        memory_spaces: Sequence[str],
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_MEMORY_SPACE,
            field=field,
            memory_spaces=tuple(memory_spaces),
            diagnostic=diagnostic,
        )

    @classmethod
    def value_packed_integer_payload_from_lanes(
        cls,
        lane_field: str,
        storage_field: str,
        width_attr: str,
        *,
        storage_unit_bit_count: int,
        storage_payload_multiple: int,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
            field=lane_field,
            other_field=storage_field,
            attr_field=width_attr,
            count=storage_payload_multiple,
            minimum=storage_unit_bit_count,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_packed_integer_lanes_from_payload(
        cls,
        storage_field: str,
        lane_field: str,
        width_attr: str,
        *,
        storage_unit_bit_count: int,
        maximum_storage_unit_count: int,
        maximum_lane_count: int,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
            field=storage_field,
            other_field=lane_field,
            attr_field=width_attr,
            count=maximum_storage_unit_count,
            minimum=storage_unit_bit_count,
            maximum=maximum_lane_count,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_no_uses(
        cls,
        field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_NO_USES,
            field=field,
            diagnostic=diagnostic,
        )

    @classmethod
    def vector_extract_shape(
        cls,
        source_field: str,
        result_field: str,
        static_indices_attr: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VECTOR_EXTRACT_SHAPE,
            field=source_field,
            other_field=result_field,
            attr_field=static_indices_attr,
            diagnostic=diagnostic,
        )

    @classmethod
    def instance_flags_has_all(
        cls,
        field: str,
        enum_case: str | EnumCase,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        keyword = enum_case.keyword if isinstance(enum_case, EnumCase) else enum_case
        return cls(
            kind=GuardKind.INSTANCE_FLAGS_HAS_ALL,
            field=field,
            enum_keyword=keyword,
            diagnostic=diagnostic,
        )

    @classmethod
    def source_representation_group(
        cls,
        group_key: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        if not group_key:
            raise ValueError("source representation group key must be non-empty")
        return cls(
            kind=GuardKind.SOURCE_REPRESENTATION_GROUP,
            field=group_key,
            diagnostic=diagnostic,
        )

    @classmethod
    def source_representation_candidate(
        cls,
        group_key: str,
        candidate_key: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        if not group_key:
            raise ValueError("source representation group key must be non-empty")
        if not candidate_key:
            raise ValueError("source representation candidate key must be non-empty")
        return cls(
            kind=GuardKind.SOURCE_REPRESENTATION_CANDIDATE,
            field=group_key,
            other_field=candidate_key,
            diagnostic=diagnostic,
        )

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError(f"{self.kind.value} guard requires a field")
        if self.other_field is not None and not self.other_field:
            raise ValueError(f"{self.kind.value} other field must be non-empty")
        if self.attr_field is not None and not self.attr_field:
            raise ValueError(f"{self.kind.value} attr field must be non-empty")
        if self.attr_type is not None and not self.attr_type:
            raise ValueError(f"{self.kind.value} attr type must be non-empty")
        if self.enum_keyword is not None and not self.enum_keyword:
            raise ValueError(f"{self.kind.value} enum keyword must be non-empty")
        if self.count is not None and self.count < 0:
            raise ValueError(f"{self.kind.value} count must be non-negative")
        if self.element is not None and self.element < 0:
            raise ValueError(f"{self.kind.value} element must be non-negative")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.kind not in (GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,)
            and self.minimum > self.maximum
        ):
            raise ValueError(f"{self.kind.value} range minimum exceeds maximum")
        if self.register_class is not None and not self.register_class:
            raise ValueError(f"{self.kind.value} register class must be non-empty")
        if self.materializer is not None and not self.materializer:
            raise ValueError(f"{self.kind.value} materializer must be non-empty")
        if self.kind == GuardKind.VALUE_MEMORY_SPACE:
            if not self.memory_spaces:
                raise ValueError(f"{self.kind.value} guard needs a memory space")
            seen_memory_spaces: set[str] = set()
            for memory_space in self.memory_spaces:
                if memory_space not in MEMORY_SPACE_NAMES:
                    raise ValueError(f"unknown value memory space '{memory_space}'")
                if memory_space in seen_memory_spaces:
                    raise ValueError(
                        f"{self.kind.value} guard repeats memory space '{memory_space}'"
                    )
                seen_memory_spaces.add(memory_space)
        elif self.memory_spaces:
            raise ValueError(f"{self.kind.value} guard cannot carry a memory-space set")
        if self.kind == GuardKind.VALUE_FLOAT_EQUALS and self.f64_value is None:
            raise ValueError(f"{self.kind.value} guard needs an f64 value")
        if (
            self.kind == GuardKind.VALUE_STORAGE_ELEMENT_FORMAT
            and not self.numeric_format_c_expression
        ):
            raise ValueError(
                f"{self.kind.value} guard needs a numeric format C expression"
            )

    def validate(self, source_op: Op) -> None:
        subject = f"guard {self.kind.value}"
        if self.kind == GuardKind.VALUE_TYPE:
            _require_value(source_op, self.field, subject)
            if self.type_pattern is None:
                raise ValueError(f"{source_op.name}: {subject} needs a type pattern")
            return
        if self.kind == GuardKind.ATTR_KIND:
            _require_attr(source_op, self.field, subject)
            if self.attr_type is None:
                raise ValueError(f"{source_op.name}: {subject} needs an attr type")
            return
        if self.kind == GuardKind.ENUM_ATTR_EQUALS:
            attr = _require_attr(source_op, self.field, subject)
            if attr.attr_type != ATTR_TYPE_ENUM:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "must be an enum attr"
                )
            if self.enum_keyword is None:
                raise ValueError(f"{source_op.name}: {subject} needs an enum keyword")
            enum_def = attr.enum_def
            if enum_def is None:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "has no enum definition"
                )
            if self.enum_keyword not in enum_def.keywords:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    f"has no enum case '{self.enum_keyword}'"
                )
            return
        if self.kind == GuardKind.I64_RANGE:
            attr = _require_attr(source_op, self.field, subject)
            if attr.attr_type not in (ATTR_TYPE_I64, ATTR_TYPE_ANY):
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "must be an i64 or any attr"
                )
            if self.minimum is None or self.maximum is None:
                raise ValueError(f"{source_op.name}: {subject} needs minimum/maximum")
            return
        if self.kind == GuardKind.DESCRIPTOR_AVAILABLE:
            if self.descriptor is None:
                raise ValueError(f"{source_op.name}: {subject} needs a descriptor")
            return
        if self.kind == GuardKind.VALUE_MATERIALIZABLE:
            _require_operand(source_op, self.field, subject)
            if self.materializer is None:
                raise ValueError(f"{source_op.name}: {subject} needs a materializer")
            return
        if self.kind in _LOW_VALUE_GUARD_KINDS:
            _validate_low_value_guard(self, source_op, subject)
            return
        if self.kind == GuardKind.OPERAND_SEGMENT_COUNT:
            operand = _require_operand(source_op, self.field, subject)
            if not operand.variadic:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "must be a variadic operand"
                )
            if self.count is None:
                raise ValueError(f"{source_op.name}: {subject} needs a count")
            return
        if self.kind in (
            GuardKind.VALUE_SIGNED_BIT_COUNT,
            GuardKind.VALUE_UNSIGNED_BIT_COUNT,
            GuardKind.VALUE_EXACT_I64,
            GuardKind.VALUE_EXACT_POWER_OF_TWO_I64,
            GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD,
            GuardKind.VALUE_EXACT_FLOAT,
            GuardKind.VALUE_I64_RANGE,
            GuardKind.VALUE_I64_RANGE_LE,
            GuardKind.VALUE_I64_RANGE_GE,
            GuardKind.VALUE_FLOAT_EQUALS,
            GuardKind.VALUE_STORAGE_ELEMENT_FORMAT,
            GuardKind.VALUE_MEMORY_SPACE,
            GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
            GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
            GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ,
        ):
            _validate_value_fact_guard(self, source_op, subject)
            return
        if self.kind == GuardKind.VALUE_NO_USES:
            _require_value(source_op, self.field, subject)
            return
        if self.kind == GuardKind.VECTOR_EXTRACT_SHAPE:
            _require_operand(source_op, self.field, subject)
            if self.other_field is None or self.attr_field is None:
                raise ValueError(
                    f"{source_op.name}: {subject} needs source, result, and "
                    "static_indices fields"
                )
            _require_value(source_op, self.other_field, subject)
            attr = _require_attr(source_op, self.attr_field, subject)
            if attr.attr_type != ATTR_TYPE_I64_ARRAY:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.attr_field}' "
                    "must be an i64_array attr"
                )
            if source_op.name != "vector.extract":
                raise ValueError(
                    f"{source_op.name}: {subject} is only valid for vector.extract"
                )
            return
        if self.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL:
            attr = _require_attr(source_op, self.field, subject)
            if attr.attr_type != ATTR_TYPE_FLAGS:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "must be a flags attr"
                )
            if self.enum_keyword is None:
                raise ValueError(f"{source_op.name}: {subject} needs a flag keyword")
            enum_def = attr.enum_def
            if enum_def is None:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "has no enum definition"
                )
            if self.enum_keyword not in enum_def.keywords:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    f"has no enum case '{self.enum_keyword}'"
                )
            return
        if self.kind == GuardKind.SOURCE_REPRESENTATION_GROUP:
            return
        if self.kind == GuardKind.SOURCE_REPRESENTATION_CANDIDATE:
            if self.other_field is None:
                raise ValueError(f"{source_op.name}: {subject} needs a candidate key")
            return
        _validate_i64_array_guard(self, source_op, subject)


def _validate_low_value_guard(
    guard: Guard,
    source_op: Op,
    subject: str,
) -> None:
    _require_value(source_op, guard.field, subject)
    if guard.kind == GuardKind.LOW_VALUE_REGISTER_CLASS:
        if guard.register_class is None:
            raise ValueError(f"{source_op.name}: {subject} needs a register class")
        return
    if guard.kind == GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT:
        if guard.count is None or guard.count <= 0:
            raise ValueError(f"{source_op.name}: {subject} needs a positive unit count")
        return
    if guard.kind == GuardKind.VALUE_STATIC_DIM0_MULTIPLE:
        if guard.count is None or guard.count <= 0:
            raise ValueError(f"{source_op.name}: {subject} needs a positive multiple")
        return
    if guard.other_field is None:
        raise ValueError(f"{source_op.name}: {subject} needs another value")
    _require_value(source_op, guard.other_field, subject)


def _require_positive_u32(
    value: int | None,
    source_op: Op,
    subject: str,
    name: str,
) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{source_op.name}: {subject} needs a positive {name}")
    if value > _MAX_U32:
        raise ValueError(f"{source_op.name}: {subject} {name} must fit in u32")


def _validate_value_fact_guard(
    guard: Guard,
    source_op: Op,
    subject: str,
) -> None:
    _require_value(source_op, guard.field, subject)
    if guard.kind in (
        GuardKind.VALUE_I64_RANGE_LE,
        GuardKind.VALUE_I64_RANGE_GE,
        GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ,
    ):
        if guard.other_field is None:
            raise ValueError(f"{source_op.name}: {subject} needs another value")
        _require_value(source_op, guard.other_field, subject)
        return
    if guard.kind in (
        GuardKind.VALUE_SIGNED_BIT_COUNT,
        GuardKind.VALUE_UNSIGNED_BIT_COUNT,
    ):
        if guard.count is None or guard.count <= 0:
            raise ValueError(f"{source_op.name}: {subject} needs a positive bit count")
        return
    if guard.kind == GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD:
        if guard.count not in (0, 1):
            raise ValueError(
                f"{source_op.name}: {subject} needs an expected add indicator"
            )
        return
    if guard.kind == GuardKind.VALUE_I64_RANGE and (
        guard.minimum is None or guard.maximum is None
    ):
        raise ValueError(f"{source_op.name}: {subject} needs minimum/maximum")
    if guard.kind == GuardKind.VALUE_FLOAT_EQUALS and guard.f64_value is None:
        raise ValueError(f"{source_op.name}: {subject} needs an f64 value")
    if guard.kind in (
        GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
        GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
    ):
        if guard.other_field is None:
            raise ValueError(f"{source_op.name}: {subject} needs another value")
        _require_value(source_op, guard.other_field, subject)
        if guard.attr_field is None:
            raise ValueError(f"{source_op.name}: {subject} needs an attr field")
        attr = _require_attr(source_op, guard.attr_field, subject)
        if attr.attr_type != ATTR_TYPE_I64:
            raise ValueError(
                f"{source_op.name}: {subject} attr field "
                f"'{guard.attr_field}' must be an i64 attr"
            )
        _require_positive_u32(
            guard.minimum,
            source_op,
            subject,
            "storage unit bit count",
        )
        _require_positive_u32(
            guard.count,
            source_op,
            subject,
            (
                "storage payload multiple"
                if guard.kind == GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES
                else "maximum storage unit count"
            ),
        )
        if guard.kind == GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD:
            _require_positive_u32(
                guard.maximum,
                source_op,
                subject,
                "maximum lane count",
            )


def _validate_i64_array_guard(
    guard: Guard,
    source_op: Op,
    subject: str,
) -> None:
    attr = _require_attr(source_op, guard.field, subject)
    if attr.attr_type != ATTR_TYPE_I64_ARRAY:
        raise ValueError(
            f"{source_op.name}: {subject} field '{guard.field}' "
            "must be an i64_array attr"
        )
    if guard.kind == GuardKind.I64_ARRAY_COUNT:
        if guard.count is None:
            raise ValueError(f"{source_op.name}: {subject} needs a count")
        return
    if guard.kind == GuardKind.I64_ARRAY_ELEMENT_RANGE:
        if guard.element is None or guard.minimum is None or guard.maximum is None:
            raise ValueError(
                f"{source_op.name}: {subject} needs element/minimum/maximum"
            )
        return
    if guard.minimum is None or guard.maximum is None:
        raise ValueError(f"{source_op.name}: {subject} needs minimum/maximum")
