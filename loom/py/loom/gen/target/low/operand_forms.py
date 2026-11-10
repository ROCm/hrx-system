# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile operand forms and legal commutative alternatives into retained maps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from loom.gen.target.low import validation
from loom.gen.target.low.compiled import CompiledOperandForm, CompiledOperandFormMatch
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_SET_ORDINAL_NONE,
    ConstraintKind,
    Descriptor,
    OperandForm,
    OperandFormImmediateAction,
    OperandFormMatchKind,
    RegClassAltFlag,
)


def _compile_operand_form(
    descriptor: Descriptor,
    descriptor_ordinals: dict[str, int],
    selected_descriptors: Sequence[Descriptor],
    operand_form: OperandForm,
    permutation: tuple[int, ...],
) -> (
    tuple[
        CompiledOperandForm,
        tuple[CompiledOperandFormMatch, ...],
        tuple[int, ...],
    ]
    | None
):
    is_commuted = any(index != source for index, source in enumerate(permutation))
    operand_indices, immediate_indices = validation.index_descriptor_fields(descriptor)
    source_packet_indices = validation.descriptor_packet_operand_indices(descriptor)
    source_packet_index_by_operand_index = {operand_index: packet_index for packet_index, operand_index in enumerate(source_packet_indices)}
    compiled_matches: list[CompiledOperandFormMatch] = []
    match_index_by_source_operand: dict[str, int] = {}
    for match in operand_form.matches:
        if match.match_kind is OperandFormMatchKind.ACCEPTED_REGISTER_CLASS:
            raise ValueError("receiving register-class predicates are generated from operand maps")
        source_operand_index = operand_indices.get(match.source_operand)
        if source_operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form references unknown source operand '{match.source_operand}'")
        source_operand_index = permutation[source_operand_index]
        source_packet_operand_index = source_packet_index_by_operand_index.get(source_operand_index)
        if source_packet_operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form source '{match.source_operand}' is not a packet operand")
        validation.validate_i64(
            match.match_i64,
            f"descriptor '{descriptor.key}' operand form match value",
        )
        match_index_by_source_operand[match.source_operand] = len(compiled_matches)
        compiled_matches.append(
            CompiledOperandFormMatch(
                source_operand_index=source_operand_index,
                source_packet_operand_index=source_packet_operand_index,
                match_kind=match.match_kind,
                match_i64=match.match_i64,
            )
        )

    replacement_ordinal = descriptor_ordinals[operand_form.replacement_descriptor]
    replacement = selected_descriptors[replacement_ordinal]

    # Replacements preserve the packet's retained memory proof at each effect
    # ordinal. Timing events may change with the encoding; semantic effects may
    # not change identity, width, scope, or ordering.
    def semantic_effects(value: Descriptor):
        return tuple((effect.kind, effect.memory_space, effect.scope_id, effect.flags, effect.width_bits) for effect in value.effects)

    if semantic_effects(descriptor) != semantic_effects(replacement):
        raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' must preserve semantic effect ordinals")
    _replacement_operand_indices, replacement_immediate_indices = validation.index_descriptor_fields(replacement)
    source_result_count = validation.validate_descriptor_operands(descriptor).result_count
    replacement_result_count = validation.validate_descriptor_operands(replacement).result_count
    if replacement_result_count != source_result_count:
        raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' must have the same result count")
    for i in range(source_result_count):
        source_result = descriptor.operands[i].field_name
        replacement_result = replacement.operands[i].field_name
        if replacement_result != source_result:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' result {i} is '{replacement_result}' instead of '{source_result}'")
    source_immediates = tuple(immediate.field_name for immediate in descriptor.immediates)
    replacement_immediates = tuple(immediate.field_name for immediate in replacement.immediates)
    source_immediate_indices = {field_name: i for i, field_name in enumerate(source_immediates)}
    replacement_immediate_index = LOW_DESCRIPTOR_SET_ORDINAL_NONE
    source_immediate_index = LOW_DESCRIPTOR_SET_ORDINAL_NONE
    immediate_match_index = LOW_DESCRIPTOR_SET_ORDINAL_NONE
    if operand_form.immediate_action is OperandFormImmediateAction.NONE:
        if operand_form.immediate_field is not None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form without an immediate action names immediate field '{operand_form.immediate_field}'")
        expected_replacement_immediates = source_immediates
    elif operand_form.immediate_action is OperandFormImmediateAction.SET_MATCHED_I64:
        if operand_form.immediate_field is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate field")
        if operand_form.immediate_source_operand is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate source")
        immediate_match_index = match_index_by_source_operand[operand_form.immediate_source_operand]
        replacement_immediate_index = replacement_immediate_indices.get(operand_form.immediate_field, LOW_DESCRIPTOR_SET_ORDINAL_NONE)
        if replacement_immediate_index == LOW_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' has no immediate field '{operand_form.immediate_field}'")
        if not validation.immediate_accepts_i64_assignment(replacement.immediates[replacement_immediate_index]):
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' immediate field '{operand_form.immediate_field}' cannot hold an i64 value")
        if operand_form.immediate_field in immediate_indices:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement immediate '{operand_form.immediate_field}' already exists on the source descriptor")
        expected_replacement_immediates = tuple((*source_immediates, operand_form.immediate_field))
    elif operand_form.immediate_action is OperandFormImmediateAction.ADD_MATCHED_I64:
        if operand_form.immediate_field is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate field")
        if operand_form.immediate_source_operand is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate source")
        immediate_match_index = match_index_by_source_operand[operand_form.immediate_source_operand]
        source_immediate_index = source_immediate_indices.get(operand_form.immediate_field, LOW_DESCRIPTOR_SET_ORDINAL_NONE)
        replacement_immediate_index = replacement_immediate_indices.get(operand_form.immediate_field, LOW_DESCRIPTOR_SET_ORDINAL_NONE)
        if source_immediate_index == LOW_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' operand form source has no immediate field '{operand_form.immediate_field}'")
        if replacement_immediate_index == LOW_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' has no immediate field '{operand_form.immediate_field}'")
        if descriptor.immediates[source_immediate_index] != replacement.immediates[replacement_immediate_index]:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' immediate field '{operand_form.immediate_field}' has different metadata")
        if not validation.immediate_accepts_i64_arithmetic(replacement.immediates[replacement_immediate_index]):
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' immediate field '{operand_form.immediate_field}' cannot hold an i64 value")
        expected_replacement_immediates = source_immediates
    else:
        raise ValueError(f"descriptor '{descriptor.key}' operand form has unsupported immediate action {operand_form.immediate_action}")
    if replacement_immediates != expected_replacement_immediates:
        raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' has immediate fields {replacement_immediates!r}, expected {expected_replacement_immediates!r}")

    replacement_packet_indices = validation.descriptor_packet_operand_indices(replacement)
    source_packet_index_by_field = {descriptor.operands[operand_index].field_name: packet_index for packet_index, operand_index in enumerate(source_packet_indices)}
    operand_map: list[int] = []
    matched_source_operands = {match.source_operand for match in operand_form.matches}
    for replacement_operand_index in replacement_packet_indices:
        field_name = replacement.operands[replacement_operand_index].field_name
        if field_name in matched_source_operands:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' still consumes source operand '{field_name}'")
        source_packet_index = source_packet_index_by_field.get(field_name)
        if source_packet_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' operand '{field_name}' does not exist in the source packet")
        source_operand_index = permutation[source_packet_indices[source_packet_index]]
        source_operand = descriptor.operands[source_operand_index]
        receiving_operand = replacement.operands[replacement_operand_index]
        if source_operand.unit_count != receiving_operand.unit_count:
            if is_commuted:
                return None
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' operand '{field_name}' changes register unit count")
        source_classes = {alt.reg_class: alt for alt in source_operand.reg_alts if RegClassAltFlag.IMMEDIATE not in alt.flags}
        receiving_classes = {alt.reg_class: alt for alt in receiving_operand.reg_alts if RegClassAltFlag.IMMEDIATE not in alt.flags}
        if not source_classes.keys() & receiving_classes.keys():
            if is_commuted:
                return None
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' operand '{field_name}' has no compatible register class")
        for name in source_classes.keys() & receiving_classes.keys():
            if source_classes[name].register_part != receiving_classes[name].register_part:
                if is_commuted:
                    return None
                raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' operand '{field_name}' changes register part")
        source_packet_index = source_packet_index_by_operand_index[source_operand_index]
        operand_map.append(source_packet_index)
        if not source_classes.keys() <= receiving_classes.keys():
            compiled_matches.append(
                CompiledOperandFormMatch(
                    source_operand_index=source_operand_index,
                    source_packet_operand_index=source_packet_index,
                    match_kind=OperandFormMatchKind.ACCEPTED_REGISTER_CLASS,
                    match_i64=replacement_operand_index,
                )
            )

    return (
        CompiledOperandForm(
            replacement_descriptor_ordinal=replacement_ordinal,
            source_immediate_index=source_immediate_index,
            replacement_immediate_index=replacement_immediate_index,
            immediate_match_index=immediate_match_index,
            immediate_action=operand_form.immediate_action,
            match_start=0,
            match_count=len(compiled_matches),
            operand_map_start=0,
            operand_map_count=len(operand_map),
        ),
        tuple(compiled_matches),
        tuple(operand_map),
    )


def compile_operand_forms(
    descriptor: Descriptor,
    descriptor_ordinals: dict[str, int],
    selected_descriptors: Sequence[Descriptor],
) -> Iterator[tuple[CompiledOperandForm, tuple[CompiledOperandFormMatch, ...], tuple[int, ...]]]:
    """Preserves authored priority, trying legal commutations within each form.

    COMMUTABLE exchanges raw operand values with attributes and effects unchanged.
    It does not grant a register conversion. Receiving-class guards retain the
    narrower type requirements in the same predicate plan as exact-value matches.
    Automatic exchanges require a known, non-increasing native encoding size.
    Each declared pair contributes one alternative; no permutation closure is
    needed by the JIT or constructed here.
    """
    identity = tuple(range(len(descriptor.operands)))
    for operand_form in descriptor.operand_forms:
        if operand_form.replacement_descriptor not in descriptor_ordinals:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{operand_form.replacement_descriptor}' was not selected")
        original = _compile_operand_form(descriptor, descriptor_ordinals, selected_descriptors, operand_form, identity)
        assert original is not None
        yield original
        replacement = selected_descriptors[descriptor_ordinals[operand_form.replacement_descriptor]]
        # Semantic commutativity does not decide whether another encoding is
        # profitable. In particular, duplicating a shared constant into literal
        # words can grow code and perturb allocation. Automatic exchanges only
        # use replacements with a proven non-increasing instruction footprint.
        if descriptor.encoding_byte_count is None or replacement.encoding_byte_count is None or replacement.encoding_byte_count > descriptor.encoding_byte_count:
            continue
        matched_fields = {match.source_operand for match in operand_form.matches}
        for constraint in descriptor.constraints:
            if constraint.kind is not ConstraintKind.COMMUTABLE:
                continue
            left_index, right_index = constraint.lhs_operand_index, constraint.rhs_operand_index
            left, right = descriptor.operands[left_index], descriptor.operands[right_index]
            if not matched_fields.intersection((left.field_name, right.field_name)):
                continue
            permutation = list(identity)
            permutation[left_index], permutation[right_index] = right_index, left_index
            commuted = _compile_operand_form(descriptor, descriptor_ordinals, selected_descriptors, operand_form, tuple(permutation))
            if commuted is not None:
                yield commuted
