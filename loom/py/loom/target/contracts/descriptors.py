# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor lookup and validation helpers for target contracts."""

from __future__ import annotations

from collections.abc import Iterable

from loom.dsl import Op
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    OperandFlag,
    OperandRole,
)
from loom.target.low_descriptors import (
    Operand as DescriptorOperand,
)


def descriptor_by_key(descriptor_set: DescriptorSet, key: str) -> Descriptor:
    """Returns the descriptor with the stable key, failing on missing keys."""

    for descriptor in descriptor_set.descriptors:
        if descriptor.key == key:
            return descriptor
    raise ValueError(f"descriptor set '{descriptor_set.key}' has no descriptor '{key}'")


def descriptor_by_semantic_tag(
    descriptor_set: DescriptorSet,
    semantic_tag: str,
) -> Descriptor:
    """Returns the unique descriptor with the semantic tag."""

    matches = [
        descriptor
        for descriptor in descriptor_set.descriptors
        if descriptor.semantic_tag == semantic_tag
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"descriptor set '{descriptor_set.key}' has no descriptor with "
            f"semantic tag '{semantic_tag}'"
        )
    raise ValueError(
        f"descriptor set '{descriptor_set.key}' has {len(matches)} descriptors "
        f"with semantic tag '{semantic_tag}'"
    )


def _validate_descriptor_set_keys(descriptor_set: DescriptorSet) -> None:
    seen = set[str]()
    for descriptor in descriptor_set.descriptors:
        if descriptor.key in seen:
            raise ValueError(
                f"descriptor set '{descriptor_set.key}' has duplicate "
                f"descriptor key '{descriptor.key}'"
            )
        seen.add(descriptor.key)


def _require_descriptor(
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
) -> Descriptor:
    selected = descriptor_by_key(descriptor_set, descriptor.key)
    if selected != descriptor:
        raise ValueError(
            f"descriptor '{descriptor.key}' does not match the descriptor "
            f"registered in set '{descriptor_set.key}'"
        )
    return selected


def _descriptor_operand_by_name(
    descriptor: Descriptor,
) -> dict[str, DescriptorOperand]:
    result = {}
    for operand in descriptor.operands:
        if operand.field_name in result:
            raise ValueError(
                f"descriptor '{descriptor.key}' has duplicate operand field "
                f"'{operand.field_name}'"
            )
        result[operand.field_name] = operand
    return result


def _require_descriptor_operand(
    descriptor: Descriptor,
    field: str,
    subject: str,
) -> DescriptorOperand:
    operand = _descriptor_operand_by_name(descriptor).get(field)
    if operand is None:
        raise ValueError(
            f"descriptor '{descriptor.key}': {subject} field '{field}' "
            "is not a descriptor operand"
        )
    return operand


def _require_input_descriptor_role(
    descriptor: Descriptor,
    operand: DescriptorOperand,
) -> None:
    if operand.role == OperandRole.RESULT:
        raise ValueError(
            f"descriptor '{descriptor.key}': operand field "
            f"'{operand.field_name}' is a result, not an input"
        )


def _require_output_descriptor_role(
    descriptor: Descriptor,
    operand: DescriptorOperand,
) -> None:
    if operand.role not in (OperandRole.RESULT, OperandRole.OPERAND_RESULT):
        raise ValueError(
            f"descriptor '{descriptor.key}': result field "
            f"'{operand.field_name}' is not a descriptor result"
        )


def _validate_required_descriptor_operands(
    source_op: Op,
    descriptor: Descriptor,
    bound_operand_names: Iterable[str],
    bound_result_names: Iterable[str],
) -> None:
    bound_operands = set(bound_operand_names)
    bound_results = set(bound_result_names)
    for operand in descriptor.operands:
        if _descriptor_operand_optional(operand):
            continue
        if operand.role == OperandRole.RESULT:
            if operand.field_name not in bound_results:
                raise ValueError(
                    f"{source_op.name}: descriptor '{descriptor.key}' result "
                    f"'{operand.field_name}' is not bound"
                )
            continue
        if operand.role == OperandRole.OPERAND_RESULT:
            if (
                operand.field_name not in bound_operands
                and operand.field_name not in bound_results
            ):
                raise ValueError(
                    f"{source_op.name}: descriptor '{descriptor.key}' "
                    f"operand-result '{operand.field_name}' is not bound"
                )
            continue
        if operand.field_name not in bound_operands:
            raise ValueError(
                f"{source_op.name}: descriptor '{descriptor.key}' operand "
                f"'{operand.field_name}' is not bound"
            )


def _descriptor_operand_optional(operand: DescriptorOperand) -> bool:
    return (
        OperandFlag.OPTIONAL in operand.flags or OperandFlag.IMPLICIT in operand.flags
    )


def _immediate_by_name(descriptor: Descriptor) -> dict[str, Immediate]:
    result = {}
    for immediate in descriptor.immediates:
        if immediate.field_name in result:
            raise ValueError(
                f"descriptor '{descriptor.key}' has duplicate immediate "
                f"'{immediate.field_name}'"
            )
        result[immediate.field_name] = immediate
    return result


def _require_immediate(
    descriptor: Descriptor,
    field: str,
    subject: str,
) -> Immediate:
    immediate = _immediate_by_name(descriptor).get(field)
    if immediate is None:
        raise ValueError(
            f"descriptor '{descriptor.key}': {subject} field '{field}' "
            "is not a descriptor immediate"
        )
    return immediate


def _immediate_has_default(immediate: Immediate) -> bool:
    return ImmediateFlag.DEFAULT_VALUE in immediate.flags


def _enum_domain_values(
    descriptor_set: DescriptorSet,
    enum_domain_name: str,
) -> tuple[int, ...]:
    for enum_domain in descriptor_set.enum_domains:
        if enum_domain.name == enum_domain_name:
            return tuple(enum_value.value for enum_value in enum_domain.values)
    raise ValueError(
        f"descriptor set '{descriptor_set.key}' has no enum domain '{enum_domain_name}'"
    )


def _validate_immediate_literal(
    source_op: Op,
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
    immediate: Immediate,
    value: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{source_op.name}: descriptor '{descriptor.key}' immediate "
            f"'{immediate.field_name}' literal must be an integer"
        )
    if immediate.kind == ImmediateKind.ENUM:
        if immediate.enum_domain is None:
            raise ValueError(
                f"{source_op.name}: descriptor '{descriptor.key}' immediate "
                f"'{immediate.field_name}' has enum kind without enum domain"
            )
        enum_values = _enum_domain_values(descriptor_set, immediate.enum_domain)
        if value in enum_values:
            return
        raise ValueError(
            f"{source_op.name}: descriptor '{descriptor.key}' immediate "
            f"'{immediate.field_name}' literal {value} is not in enum domain "
            f"'{immediate.enum_domain}'"
        )
    if immediate.kind == ImmediateKind.SIGNED:
        if value < immediate.signed_min or value > immediate.unsigned_max:
            raise ValueError(
                f"{source_op.name}: descriptor '{descriptor.key}' immediate "
                f"'{immediate.field_name}' literal {value} is out of range"
            )
        return
    if value < 0 or value > immediate.unsigned_max:
        raise ValueError(
            f"{source_op.name}: descriptor '{descriptor.key}' immediate "
            f"'{immediate.field_name}' literal {value} is out of range"
        )
