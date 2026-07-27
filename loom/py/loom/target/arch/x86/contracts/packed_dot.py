# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 packed-dot source-to-low contract fragment."""

from __future__ import annotations

from dataclasses import dataclass

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.x86.descriptors import X86_PACKED_DOT_DESCRIPTOR_SET
from loom.target.arch.x86.packed_dot_data import (
    CONTRACT_FLAG_SATURATING,
    NUMERIC_BF16,
    NUMERIC_F16,
    NUMERIC_F32,
    NUMERIC_I8,
    NUMERIC_I32,
    NUMERIC_U8,
    X86_PACKED_DOT_DESCRIPTORS,
    PackedDotDescriptor,
)
from loom.target.contracts import (
    ContractFragment,
    DescriptorRule,
    DotDescriptorCase,
    GuardDiagnostic,
    TypePattern,
    Vector,
    descriptor_by_key,
    dot_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor

_DOT_TYPE_FIELD_NAMES = {
    "acc": "accumulator",
}

_DOT_KIND_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="kind",
    constraint_key="x86.packed_dot.kind",
)

_DOT_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="descriptor",
    subject_name="x86.packed_dot",
    constraint_key="x86.packed_dot.descriptor",
)

_DOT4I_KIND_BY_NUMERIC = {
    (NUMERIC_I8, NUMERIC_I8): "s8s8",
    (NUMERIC_U8, NUMERIC_I8): "u8s8",
    (NUMERIC_I8, NUMERIC_U8): "s8u8",
    (NUMERIC_U8, NUMERIC_U8): "u8u8",
}

_VECTOR_ELEMENT_BY_NUMERIC = {
    NUMERIC_BF16: "bf16",
    NUMERIC_F16: "f16",
    NUMERIC_F32: "f32",
    NUMERIC_I8: "i8",
    NUMERIC_I32: "i32",
    NUMERIC_U8: "i8",
}


@dataclass(frozen=True, slots=True)
class _PackedDotSourceCase:
    source_op: Op
    kind: str | None
    input_numeric_type: str


type _PackedDotSignature = tuple[int, str, str, str, str]

_PACKED_DOT_SOURCE_CASES = {
    **{
        (2, numeric_type, numeric_type, NUMERIC_F32, NUMERIC_F32): (
            _PackedDotSourceCase(
                source_op=vector.vector_dot2f,
                kind=None,
                input_numeric_type=numeric_type,
            )
        )
        for numeric_type in (NUMERIC_F16, NUMERIC_BF16)
    },
    **{
        (4, lhs_numeric_type, rhs_numeric_type, NUMERIC_I32, NUMERIC_I32): (
            _PackedDotSourceCase(
                source_op=vector.vector_dot4i,
                kind=kind,
                input_numeric_type=NUMERIC_I8,
            )
        )
        for (lhs_numeric_type, rhs_numeric_type), kind in _DOT4I_KIND_BY_NUMERIC.items()
    },
}


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_PACKED_DOT_DESCRIPTOR_SET, key)


def _vector_type(numeric_type: str, lane_count: int) -> TypePattern:
    return Vector(_VECTOR_ELEMENT_BY_NUMERIC[numeric_type], lanes=lane_count)


def _type_diagnostic(field: str) -> GuardDiagnostic:
    field_name = _DOT_TYPE_FIELD_NAMES.get(field, field)
    return GuardDiagnostic(
        subject_role="type",
        subject_name=field,
        constraint_key=f"x86.packed_dot.{field_name}_type",
    )


def _source_signature(descriptor_data: PackedDotDescriptor) -> _PackedDotSignature:
    return (
        descriptor_data.reduction_group_size,
        descriptor_data.lhs_numeric_type,
        descriptor_data.rhs_numeric_type,
        descriptor_data.accumulator_numeric_type,
        descriptor_data.result_numeric_type,
    )


def _source_case(descriptor_data: PackedDotDescriptor) -> _PackedDotSourceCase | None:
    return _PACKED_DOT_SOURCE_CASES.get(_source_signature(descriptor_data))


def _packed_dot_case(descriptor_data: PackedDotDescriptor) -> DotDescriptorCase | None:
    source_case = _source_case(descriptor_data)
    if source_case is None:
        return None
    descriptor = _descriptor(descriptor_data.key)
    source_type = _vector_type(
        source_case.input_numeric_type,
        descriptor_data.input_lane_count,
    )
    result_type = _vector_type(
        descriptor_data.result_numeric_type,
        descriptor_data.result_lane_count,
    )
    return DotDescriptorCase(
        source_op=source_case.source_op,
        descriptor=descriptor,
        lhs_type=source_type,
        rhs_type=source_type,
        accumulator_type=result_type,
        result_type=result_type,
        kind=source_case.kind,
    )


def _packed_dot_rules() -> tuple[DescriptorRule, ...]:
    cases: list[DotDescriptorCase] = []
    for descriptor_data in X86_PACKED_DOT_DESCRIPTORS:
        if descriptor_data.flags & CONTRACT_FLAG_SATURATING:
            continue
        case = _packed_dot_case(descriptor_data)
        if case is not None:
            cases.append(case)
    return dot_descriptor_rules(
        cases,
        kind_diagnostic=_DOT_KIND_DIAGNOSTIC,
        type_diagnostics={
            field: _type_diagnostic(field) for field in ("lhs", "rhs", "acc", "result")
        },
        descriptor_diagnostic=_DOT_DESCRIPTOR_DIAGNOSTIC,
    )


X86_PACKED_DOT_CONTRACT_DIALECT_OPS = {
    "vector": ALL_VECTOR_OPS,
}

X86_PACKED_DOT_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.packed_dot",
    descriptor_set=X86_PACKED_DOT_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/packed_dot.h",
    cases=_packed_dot_rules(),
)
