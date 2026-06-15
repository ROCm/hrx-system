# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler from low descriptor inputs to the dense intermediate model."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from loom.gen.support.string_pool import CStringPool
from loom.gen.target.low import validation
from loom.gen.target.low.compiled import (
    CompiledAsmForm,
    CompiledAsmImmediate,
    CompiledDescriptorSet,
    CompiledNativeAsmValue,
    CompiledOperandForm,
    CompiledOperandFormMatch,
    DescriptorAllowlist,
)
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    LOW_DESCRIPTOR_SET_ORDINAL_NONE,
    AsmForm,
    Constraint,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EncodingFieldValue,
    EnumValue,
    Hazard,
    Immediate,
    ImmediateEncodingSlice,
    ImmediateKind,
    IssueUse,
    NativeAsmValue,
    NativeAsmValueKind,
    Operand,
    OperandForm,
    OperandFormImmediateAction,
    OperandRole,
    PressureDelta,
    RegClassAltFlag,
    StorageLease,
    descriptor_stable_id,
)


def _dedupe_by_name[T](items: Sequence[T], get_name: Callable[[T], str]) -> dict[str, T]:
    result: dict[str, T] = {}
    for item in items:
        name = get_name(item)
        if name in result:
            raise ValueError(f"duplicate low descriptor input name '{name}'")
        result[name] = item
    return result


def _select_descriptors(spec: DescriptorSet, allowlist: DescriptorAllowlist | None) -> list[Descriptor]:
    key_map = {descriptor.key: descriptor for descriptor in spec.descriptors}
    semantic_map: dict[str, list[Descriptor]] = {}
    mnemonic_map: dict[str, list[Descriptor]] = {}
    for descriptor in spec.descriptors:
        if descriptor.semantic_tag is not None:
            semantic_map.setdefault(descriptor.semantic_tag, []).append(descriptor)
        if descriptor.mnemonic is not None:
            mnemonic_map.setdefault(descriptor.mnemonic, []).append(descriptor)

    if allowlist is None or allowlist.is_empty():
        return list(spec.descriptors)

    selected: dict[str, Descriptor] = {}
    for key in allowlist.keys:
        selected_descriptor = key_map.get(key)
        if selected_descriptor is None:
            raise ValueError(f"allowlist references unknown descriptor key '{key}'")
        selected[selected_descriptor.key] = selected_descriptor
    for semantic_tag in allowlist.semantic_tags:
        descriptors = semantic_map.get(semantic_tag)
        if descriptors is None:
            raise ValueError(f"allowlist references unknown semantic tag '{semantic_tag}'")
        for descriptor in descriptors:
            selected[descriptor.key] = descriptor
    for mnemonic in allowlist.mnemonics:
        descriptors = mnemonic_map.get(mnemonic)
        if descriptors is None:
            raise ValueError(f"allowlist references unknown mnemonic '{mnemonic}'")
        for descriptor in descriptors:
            selected[descriptor.key] = descriptor

    changed = True
    while changed:
        changed = False
        for descriptor in tuple(selected.values()):
            for operand_form in descriptor.operand_forms:
                replacement = key_map.get(operand_form.replacement_descriptor)
                if replacement is None:
                    raise ValueError(f"descriptor '{descriptor.key}' operand form references unknown replacement descriptor '{operand_form.replacement_descriptor}'")
                if replacement.key not in selected:
                    selected[replacement.key] = replacement
                    changed = True

    return [descriptor for descriptor in spec.descriptors if descriptor.key in selected]


def _index_descriptor_fields(
    descriptor: Descriptor,
) -> tuple[dict[str, int], dict[str, int]]:
    operand_indices: dict[str, int] = {}
    for i, operand in enumerate(descriptor.operands):
        if operand.field_name in operand_indices:
            raise ValueError(f"descriptor '{descriptor.key}' operand field '{operand.field_name}' is duplicated and cannot be used by asm forms")
        operand_indices[operand.field_name] = i

    immediate_indices: dict[str, int] = {}
    for i, immediate in enumerate(descriptor.immediates):
        if immediate.field_name in immediate_indices:
            raise ValueError(f"descriptor '{descriptor.key}' immediate field '{immediate.field_name}' is duplicated and cannot be used by asm forms")
        immediate_indices[immediate.field_name] = i
    return operand_indices, immediate_indices


def _compile_operand_form(
    descriptor: Descriptor,
    descriptor_ordinals: dict[str, int],
    selected_descriptors: Sequence[Descriptor],
    operand_form: OperandForm,
    match_start: int,
    operand_map_start: int,
) -> tuple[
    CompiledOperandForm,
    tuple[CompiledOperandFormMatch, ...],
    tuple[int, ...],
]:
    operand_indices, immediate_indices = _index_descriptor_fields(descriptor)
    source_packet_indices = validation.descriptor_packet_operand_indices(descriptor)
    source_packet_index_by_operand_index = {operand_index: packet_index for packet_index, operand_index in enumerate(source_packet_indices)}
    compiled_matches: list[CompiledOperandFormMatch] = []
    match_index_by_source_operand: dict[str, int] = {}
    for match in operand_form.matches:
        source_operand_index = operand_indices.get(match.source_operand)
        if source_operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form references unknown source operand '{match.source_operand}'")
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
    _replacement_operand_indices, replacement_immediate_indices = _index_descriptor_fields(replacement)
    source_result_count = validation.validate_descriptor_operands(descriptor)
    replacement_result_count = validation.validate_descriptor_operands(replacement)
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
        operand_map.append(source_packet_index)

    return (
        CompiledOperandForm(
            replacement_descriptor_ordinal=replacement_ordinal,
            source_immediate_index=source_immediate_index,
            replacement_immediate_index=replacement_immediate_index,
            immediate_match_index=immediate_match_index,
            immediate_action=operand_form.immediate_action,
            match_start=match_start,
            match_count=len(compiled_matches),
            operand_map_start=operand_map_start,
            operand_map_count=len(operand_map),
        ),
        tuple(compiled_matches),
        tuple(operand_map),
    )


def _compile_asm_form(
    string_pool: CStringPool,
    descriptor: Descriptor,
    descriptor_ordinal: int,
    asm_form: AsmForm,
    form_ordinal: int,
    *,
    label_scope: str | None = None,
) -> CompiledAsmForm:
    mnemonic = validation.asm_form_mnemonic(descriptor, asm_form)
    validation.validate_unique_asm_fields(descriptor, asm_form, mnemonic)
    operand_indices, immediate_indices = _index_descriptor_fields(descriptor)

    def scoped_label(prefix: str, *parts: str) -> str:
        label_parts = [prefix]
        if label_scope is not None:
            label_parts.append(label_scope)
        label_parts.extend((descriptor.key, str(form_ordinal), *parts))
        return "_".join(label_parts)

    mnemonic_label = string_pool.intern(scoped_label("asm_mnemonic"), mnemonic)
    native_assembly_mnemonic_label = None
    native_assembly_mnemonic = asm_form.native_assembly_mnemonic
    if native_assembly_mnemonic is not None:
        if native_assembly_mnemonic == "":
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' has an empty native assembly mnemonic")
        if native_assembly_mnemonic == mnemonic:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' repeats its low asm mnemonic as a native assembly override")
        if len(native_assembly_mnemonic.encode()) > 255:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native assembly mnemonic '{native_assembly_mnemonic}' exceeds 255 bytes")
        native_assembly_mnemonic_label = string_pool.intern(
            scoped_label("asm_native_mnemonic"),
            native_assembly_mnemonic,
        )

    result_indices = []
    for field_name in asm_form.results:
        operand_index = operand_indices.get(field_name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' result references unknown operand field '{field_name}'")
        operand = descriptor.operands[operand_index]
        if operand.role is not OperandRole.RESULT:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' result field '{field_name}' names a non-result operand")
        result_indices.append(operand_index)

    packet_operand_roles = {
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    }
    operand_order = []
    for field_name in asm_form.operands:
        operand_index = operand_indices.get(field_name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' operand references unknown operand field '{field_name}'")
        operand = descriptor.operands[operand_index]
        if operand.role not in packet_operand_roles:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' operand field '{field_name}' does not name an explicit packet operand")
        operand_order.append(operand_index)

    immediate_order = []
    seen_immediate_names: set[str] = set()
    for immediate in asm_form.immediates:
        immediate_index = immediate_indices.get(immediate.field_name)
        if immediate_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' immediate references unknown immediate field '{immediate.field_name}'")
        name_label = None
        if immediate.name is not None:
            if immediate.name == "":
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' immediate '{immediate.field_name}' has an empty name")
            if len(immediate.name.encode()) > 255:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' immediate name '{immediate.name}' exceeds 255 bytes")
            if immediate.name in seen_immediate_names:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' uses immediate name '{immediate.name}' more than once")
            seen_immediate_names.add(immediate.name)
            name_label = string_pool.intern(
                scoped_label("asm_immediate", immediate.field_name),
                immediate.name,
            )
        immediate_order.append(
            CompiledAsmImmediate(
                immediate_index=immediate_index,
                name_label=name_label,
                name=immediate.name,
            )
        )

    native_assembly_values = []
    for value_ordinal, value in enumerate(asm_form.native_assembly_values):
        native_assembly_values.append(
            _compile_native_asm_value(
                string_pool,
                descriptor,
                mnemonic,
                operand_indices,
                immediate_indices,
                packet_operand_roles,
                scoped_label,
                value,
                value_ordinal,
            )
        )

    return CompiledAsmForm(
        descriptor_ordinal=descriptor_ordinal,
        mnemonic_label=mnemonic_label,
        mnemonic=mnemonic,
        native_assembly_mnemonic_label=native_assembly_mnemonic_label,
        native_assembly_mnemonic=native_assembly_mnemonic,
        result_indices=tuple(result_indices),
        operand_indices=tuple(operand_order),
        immediates=tuple(immediate_order),
        native_assembly_values=tuple(native_assembly_values),
    )


def _compile_native_asm_value(
    string_pool: CStringPool,
    descriptor: Descriptor,
    mnemonic: str,
    operand_indices: dict[str, int],
    immediate_indices: dict[str, int],
    packet_operand_roles: set[OperandRole],
    scoped_label: Callable[..., str],
    value: NativeAsmValue,
    value_ordinal: int,
) -> CompiledNativeAsmValue:
    kind = value.kind
    field_name = value.field_name
    literal = value.literal
    bit_width = value.bit_width

    def require_field_name() -> str:
        if field_name is None or field_name == "":
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} must name a descriptor field")
        return field_name

    def reject_literal_and_bit_width() -> None:
        if literal is not None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} unexpectedly specifies a literal")
        if bit_width != 0:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} unexpectedly specifies a bit width")

    if kind is NativeAsmValueKind.LITERAL:
        if field_name is not None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} unexpectedly names field '{field_name}'")
        if literal is None or literal == "":
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} has no spelling")
        if len(literal.encode()) > 255:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} exceeds 255 bytes")
        if bit_width != 0:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} unexpectedly specifies a bit width")
        return CompiledNativeAsmValue(
            kind=kind,
            index=0,
            bit_width=0,
            literal_label=string_pool.intern(
                scoped_label("asm_native_value", str(value_ordinal)),
                literal,
            ),
            literal=literal,
        )

    if kind is NativeAsmValueKind.RESULT:
        reject_literal_and_bit_width()
        name = require_field_name()
        operand_index = operand_indices.get(name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native result references unknown operand field '{name}'")
        operand = descriptor.operands[operand_index]
        if operand.role is not OperandRole.RESULT:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native result field '{name}' names a non-result operand")
        return CompiledNativeAsmValue(
            kind=kind,
            index=operand_index,
            bit_width=0,
            literal_label=None,
            literal=None,
        )

    if kind is NativeAsmValueKind.OPERAND:
        reject_literal_and_bit_width()
        name = require_field_name()
        operand_index = operand_indices.get(name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native operand references unknown operand field '{name}'")
        operand = descriptor.operands[operand_index]
        if operand.role not in packet_operand_roles:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native operand field '{name}' does not name an explicit packet operand")
        return CompiledNativeAsmValue(
            kind=kind,
            index=operand_index,
            bit_width=0,
            literal_label=None,
            literal=None,
        )

    if kind in (
        NativeAsmValueKind.IMMEDIATE_I64,
        NativeAsmValueKind.IMMEDIATE_UNSIGNED_HEX,
    ):
        if literal is not None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native immediate value {value_ordinal} unexpectedly specifies a literal")
        name = require_field_name()
        immediate_index = immediate_indices.get(name)
        if immediate_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native immediate references unknown immediate field '{name}'")
        if kind is NativeAsmValueKind.IMMEDIATE_I64:
            if bit_width != 0:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native i64 immediate '{name}' unexpectedly specifies a bit width")
        elif bit_width <= 0 or bit_width > 32:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native unsigned-hex immediate '{name}' bit width must be in [1, 32]")
        return CompiledNativeAsmValue(
            kind=kind,
            index=immediate_index,
            bit_width=bit_width,
            literal_label=None,
            literal=None,
        )

    raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} has unsupported kind {kind!r}")


def _compile_asm_forms(
    string_pool: CStringPool,
    descriptors: Sequence[Descriptor],
    *,
    allow_ambiguous_mnemonics: bool = False,
    label_scope: str | None = None,
) -> list[CompiledAsmForm]:
    compiled_forms: list[CompiledAsmForm] = []
    seen_mnemonics: dict[str, str] = {}
    for descriptor_ordinal, descriptor in enumerate(descriptors):
        for form_ordinal, asm_form in enumerate(descriptor.asm_forms):
            compiled_form = _compile_asm_form(
                string_pool,
                descriptor,
                descriptor_ordinal,
                asm_form,
                form_ordinal,
                label_scope=label_scope,
            )
            previous_descriptor_key = seen_mnemonics.get(compiled_form.mnemonic)
            if previous_descriptor_key is not None and not allow_ambiguous_mnemonics:
                raise ValueError(f"asm mnemonic '{compiled_form.mnemonic}' is ambiguous between descriptors '{previous_descriptor_key}' and '{descriptor.key}'")
            seen_mnemonics[compiled_form.mnemonic] = descriptor.key
            compiled_forms.append(compiled_form)
    return sorted(compiled_forms, key=lambda form: form.mnemonic)


def compile_asm_forms_for_descriptors(
    string_pool: CStringPool,
    descriptors: Sequence[Descriptor],
    *,
    allow_ambiguous_mnemonics: bool = False,
    label_scope: str | None = None,
) -> list[CompiledAsmForm]:
    return _compile_asm_forms(
        string_pool,
        descriptors,
        allow_ambiguous_mnemonics=allow_ambiguous_mnemonics,
        label_scope=label_scope,
    )


def append_asm_form_table_spans(
    asm_forms: Sequence[CompiledAsmForm],
    asm_operand_indices: list[int],
    asm_immediates: list[CompiledAsmImmediate],
    native_asm_values: list[CompiledNativeAsmValue],
) -> None:
    for asm_form in asm_forms:
        asm_form.result_index_start = len(asm_operand_indices)
        asm_operand_indices.extend(asm_form.result_indices)
        asm_form.operand_index_start = len(asm_operand_indices)
        asm_operand_indices.extend(asm_form.operand_indices)
        asm_form.immediate_start = len(asm_immediates)
        asm_immediates.extend(asm_form.immediates)
        asm_form.native_assembly_value_start = len(native_asm_values)
        native_asm_values.extend(asm_form.native_assembly_values)


def compile_descriptor_set(
    spec: DescriptorSet,
    allowlist: DescriptorAllowlist | None = None,
    *,
    allow_ambiguous_asm_mnemonics: bool = False,
) -> CompiledDescriptorSet:
    if spec.generator_version == 0:
        raise ValueError(f"descriptor set '{spec.key}' has zero generator version")
    reg_class_inputs = _dedupe_by_name(spec.reg_classes, lambda item: item.name)
    register_part_inputs = _dedupe_by_name(spec.register_parts, lambda item: item.name)
    resource_inputs = _dedupe_by_name(spec.resources, lambda item: item.name)
    schedule_inputs = _dedupe_by_name(spec.schedule_classes, lambda item: item.name)
    enum_domain_inputs = _dedupe_by_name(spec.enum_domains, lambda item: item.name)
    _dedupe_by_name(spec.descriptors, lambda item: item.key)

    selected_descriptors = _select_descriptors(spec, allowlist)
    if not selected_descriptors:
        raise ValueError(f"descriptor set '{spec.key}' selected no descriptors")
    validation.validate_descriptor_asm_surface(spec, selected_descriptors)
    descriptor_ordinals = {descriptor.key: i for i, descriptor in enumerate(selected_descriptors)}
    for reg_class in spec.reg_classes:
        if reg_class.full_register_part_mask == 0:
            raise ValueError(f"register class '{reg_class.name}' has an empty full register-part mask")
        if reg_class.full_register_part_mask < 0 or reg_class.full_register_part_mask > (2**32) - 1:
            raise ValueError(f"register class '{reg_class.name}' full register-part mask does not fit u32")

    # Register classes are target vocabulary, not just descriptor closure:
    # low function signatures and allocation diagnostics may reference classes
    # that a tiny allowlisted descriptor slice does not happen to use.
    used_reg_class_names: set[str] = set(reg_class_inputs)
    used_register_part_names: set[str] = set()
    used_resource_names: set[str] = set()
    used_schedule_names: set[str] = set()
    used_enum_domain_names: set[str] = set()

    for descriptor in selected_descriptors:
        result_count = validation.validate_descriptor_operands(descriptor)
        validation.validate_descriptor_encoding_fields(descriptor)
        validation.validate_descriptor_storage_leases(descriptor, result_count)
        if descriptor.encoding_id < 0 or descriptor.encoding_id > LOW_DESCRIPTOR_ENCODING_ID_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' encoding id does not fit u16")
        if descriptor.encoding_id == LOW_DESCRIPTOR_ENCODING_ID_NONE and DescriptorFlag.PSEUDO not in descriptor.flags:
            raise ValueError(f"descriptor '{descriptor.key}' uses absent encoding id without the pseudo flag")
        if DescriptorFlag.PSEUDO in descriptor.flags and descriptor.encoding_id != LOW_DESCRIPTOR_ENCODING_ID_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' uses the pseudo flag with a target encoding id")
        if DescriptorFlag.PSEUDO in descriptor.flags and descriptor.encoding_format_id != 0:
            raise ValueError(f"descriptor '{descriptor.key}' uses the pseudo flag with a target encoding format id")
        if descriptor.schedule_class is None:
            raise ValueError(f"descriptor '{descriptor.key}' has no schedule class")
        if descriptor.schedule_class not in schedule_inputs:
            raise ValueError(f"descriptor '{descriptor.key}' references unknown schedule class '{descriptor.schedule_class}'")
        used_schedule_names.add(descriptor.schedule_class)
        for immediate in descriptor.immediates:
            validation.validate_u16(
                immediate.encoding_field_id,
                f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding field id",
            )
            validation.validate_u16(
                immediate.encoding_id,
                f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding id",
            )
            validation.validate_immediate_encoding(descriptor, immediate)
            if immediate.kind is ImmediateKind.ENUM:
                if immediate.enum_domain is None:
                    raise ValueError(f"descriptor '{descriptor.key}' enum immediate '{immediate.field_name}' has no enum domain")
                if immediate.enum_domain not in enum_domain_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' enum immediate '{immediate.field_name}' references unknown enum domain '{immediate.enum_domain}'")
                used_enum_domain_names.add(immediate.enum_domain)
            elif immediate.enum_domain is not None:
                raise ValueError(f"descriptor '{descriptor.key}' non-enum immediate '{immediate.field_name}' references enum domain '{immediate.enum_domain}'")
            validation.validate_immediate_default(descriptor, immediate, enum_domain_inputs)
        for operand in descriptor.operands:
            validation.validate_u16(
                operand.encoding_field_id,
                f"descriptor '{descriptor.key}' operand '{operand.field_name}' encoding field id",
            )
            concrete_reg_alt_count = 0
            for reg_alt in operand.reg_alts:
                if reg_alt.reg_class is None:
                    if RegClassAltFlag.IMMEDIATE not in reg_alt.flags:
                        raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has a classless alternative without the immediate flag")
                    continue
                if reg_alt.reg_class not in reg_class_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' references unknown register class '{reg_alt.reg_class}'")
                used_reg_class_names.add(reg_alt.reg_class)
                concrete_reg_alt_count += 1
            if operand.register_part is not None:
                if operand.register_part not in register_part_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' references unknown register part '{operand.register_part}'")
                if concrete_reg_alt_count != 1:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' with a register part must name exactly one concrete register class alternative")
                register_part = register_part_inputs[operand.register_part]
                validation.validate_register_part(register_part)
                if register_part.reg_class not in reg_class_inputs:
                    raise ValueError(f"register part '{register_part.name}' references unknown register class '{register_part.reg_class}'")
                if not any(reg_alt.reg_class == register_part.reg_class for reg_alt in operand.reg_alts):
                    raise ValueError(
                        f"descriptor '{descriptor.key}' operand '{operand.field_name}' uses register part '{register_part.name}' for register class '{register_part.reg_class}' but the operand does not accept that class"
                    )
                used_register_part_names.add(operand.register_part)
                used_reg_class_names.add(register_part.reg_class)
        seen_fixed_encoding_fields: set[int] = set()
        for field_value in descriptor.encoding_field_values:
            validation.validate_u16(
                field_value.encoding_field_id,
                f"descriptor '{descriptor.key}' fixed encoding field id",
            )
            if field_value.encoding_field_id == 0:
                raise ValueError(f"descriptor '{descriptor.key}' has fixed encoding field id zero")
            if field_value.encoding_field_id in seen_fixed_encoding_fields:
                raise ValueError(f"descriptor '{descriptor.key}' repeats fixed encoding field id {field_value.encoding_field_id}")
            seen_fixed_encoding_fields.add(field_value.encoding_field_id)
            validation.validate_u64(
                field_value.value,
                f"descriptor '{descriptor.key}' fixed encoding field value",
            )

    for schedule_name in list(used_schedule_names):
        schedule_class = schedule_inputs[schedule_name]
        for issue_use in schedule_class.issue_uses:
            if issue_use.resource not in resource_inputs:
                raise ValueError(f"schedule class '{schedule_name}' references unknown resource '{issue_use.resource}'")
            used_resource_names.add(issue_use.resource)
        for hazard in schedule_class.hazards:
            if validation.hazard_reference_count(hazard) != 1:
                raise ValueError(f"schedule class '{schedule_name}' hazard must reference exactly one resource, counter, or target id")
            if hazard.resource is not None:
                if hazard.resource not in resource_inputs:
                    raise ValueError(f"schedule class '{schedule_name}' hazard references unknown resource '{hazard.resource}'")
                used_resource_names.add(hazard.resource)
            if hazard.counter_id is not None:
                validation.validate_u16(
                    hazard.counter_id,
                    f"schedule class '{schedule_name}' hazard counter id",
                )
            if hazard.target_id is not None:
                validation.validate_u16(
                    hazard.target_id,
                    f"schedule class '{schedule_name}' hazard target id",
                )
            validation.validate_u16(
                hazard.producer_stage,
                f"schedule class '{schedule_name}' hazard producer stage",
            )
            validation.validate_u16(
                hazard.consumer_stage,
                f"schedule class '{schedule_name}' hazard consumer stage",
            )
            validation.validate_u16(
                hazard.distance,
                f"schedule class '{schedule_name}' hazard distance",
            )
        for pressure_delta in schedule_class.pressure_deltas:
            if pressure_delta.reg_class not in reg_class_inputs:
                raise ValueError(f"schedule class '{schedule_name}' references unknown pressure register class '{pressure_delta.reg_class}'")
            used_reg_class_names.add(pressure_delta.reg_class)

    changed = True
    while changed:
        changed = False
        for reg_class_name in list(used_reg_class_names):
            spill_class = reg_class_inputs[reg_class_name].spill_class
            if spill_class is not None:
                if spill_class not in reg_class_inputs:
                    raise ValueError(f"register class '{reg_class_name}' references unknown spill class '{spill_class}'")
                if spill_class not in used_reg_class_names:
                    used_reg_class_names.add(spill_class)
                    changed = True

    for part_name in list(used_register_part_names):
        register_part = register_part_inputs[part_name]
        reg_class = reg_class_inputs[register_part.reg_class]
        if register_part.mask & ~reg_class.full_register_part_mask:
            raise ValueError(f"register part '{part_name}' mask 0x{register_part.mask:x} exceeds full mask 0x{reg_class.full_register_part_mask:x} for register class '{reg_class.name}'")

    reg_classes = [reg_class for reg_class in spec.reg_classes if reg_class.name in used_reg_class_names]
    register_parts = [part for part in spec.register_parts if part.name in used_register_part_names]
    resources = [resource for resource in spec.resources if resource.name in used_resource_names]
    schedule_classes = [schedule_class for schedule_class in spec.schedule_classes if schedule_class.name in used_schedule_names]
    enum_domains = [domain for domain in spec.enum_domains if domain.name in used_enum_domain_names]

    validation.validate_u16_table_count(len(reg_classes), f"descriptor set '{spec.key}' register class")
    validation.validate_u16_table_count(len(register_parts), f"descriptor set '{spec.key}' register part")
    reg_class_ids = {reg_class.name: i for i, reg_class in enumerate(reg_classes)}
    register_part_ids = {part.name: i for i, part in enumerate(register_parts)}
    resource_ids = {resource.name: i for i, resource in enumerate(resources)}
    schedule_class_ids = {schedule_class.name: i for i, schedule_class in enumerate(schedule_classes)}
    enum_domain_ids = {domain.name: i for i, domain in enumerate(enum_domains)}

    string_pool = CStringPool(spec.c_enum_prefix)
    string_pool.intern("empty", "")
    string_pool.intern("set_key", spec.key)
    if spec.target_key is not None:
        string_pool.intern("target_key", spec.target_key)
    if spec.feature_key is not None:
        string_pool.intern("feature_key", spec.feature_key)
    for reg_class in reg_classes:
        string_pool.intern(f"reg_{reg_class.name}", reg_class.name)
    for part in register_parts:
        string_pool.intern(f"register_part_{part.name}", part.name)
    for resource in resources:
        string_pool.intern(f"resource_{resource.name}", resource.name)
    for schedule_class in schedule_classes:
        string_pool.intern(f"schedule_{schedule_class.name}", schedule_class.name)
    for enum_domain in enum_domains:
        string_pool.intern(f"enum_domain_{enum_domain.name}", enum_domain.name)
        for enum_value in validation.validate_enum_domain(enum_domain):
            string_pool.intern(f"enum_value_{enum_domain.name}_{enum_value.token}", enum_value.token)
    for descriptor in selected_descriptors:
        string_pool.intern(f"descriptor_{descriptor.key}", descriptor.key)
        if descriptor.mnemonic is not None:
            string_pool.intern(f"mnemonic_{descriptor.key}", descriptor.mnemonic)
        if descriptor.semantic_tag is not None:
            string_pool.intern(f"semantic_{descriptor.key}", descriptor.semantic_tag)
        for operand in descriptor.operands:
            string_pool.intern(f"field_{operand.field_name}", operand.field_name)
        for immediate in descriptor.immediates:
            string_pool.intern(f"immediate_{immediate.field_name}", immediate.field_name)
        for lease_index, lease in enumerate(descriptor.storage_leases):
            string_pool.intern(
                f"storage_lease_{descriptor.key}_{lease_index}_class",
                lease.release_class_name,
            )
            string_pool.intern(
                f"storage_lease_{descriptor.key}_{lease_index}_action",
                lease.release_action_name,
            )
            string_pool.intern(
                f"storage_lease_{descriptor.key}_{lease_index}_reason",
                lease.release_reason_name,
            )

    asm_forms = compile_asm_forms_for_descriptors(
        string_pool,
        selected_descriptors,
        allow_ambiguous_mnemonics=allow_ambiguous_asm_mnemonics,
    )
    canonical_asm_form_ordinals: list[int | None] = [None] * len(selected_descriptors)
    asm_form_counts_by_descriptor = [0] * len(selected_descriptors)
    for asm_form_ordinal, asm_form in enumerate(asm_forms):
        descriptor_ordinal = asm_form.descriptor_ordinal
        asm_form_counts_by_descriptor[descriptor_ordinal] += 1
        canonical_asm_form_ordinals[descriptor_ordinal] = asm_form_ordinal
    for descriptor_ordinal, form_count in enumerate(asm_form_counts_by_descriptor):
        if form_count != 1:
            canonical_asm_form_ordinals[descriptor_ordinal] = None

    asm_operand_indices: list[int] = []
    asm_immediates: list[CompiledAsmImmediate] = []
    native_asm_values: list[CompiledNativeAsmValue] = []
    append_asm_form_table_spans(
        asm_forms,
        asm_operand_indices,
        asm_immediates,
        native_asm_values,
    )

    reg_class_alts: list[tuple[int | None, tuple[RegClassAltFlag, ...]]] = []
    reg_alt_group_starts: dict[tuple[tuple[int | None, tuple[RegClassAltFlag, ...]], ...], int] = {}
    immediate_encoding_slice_group_starts: dict[tuple[ImmediateEncodingSlice, ...], int] = {}
    effect_group_starts: dict[tuple[Effect, ...], int] = {}
    storage_lease_group_starts: dict[tuple[StorageLease, ...], int] = {}
    operands: list[Operand] = []
    operand_alt_starts: list[int] = []
    immediates: list[Immediate] = []
    immediate_encoding_slices: list[ImmediateEncodingSlice] = []
    immediate_encoding_slice_starts: list[int] = []
    enum_values: list[EnumValue] = []
    immediate_enum_domain_ids: list[int | None] = []
    effects: list[Effect] = []
    constraints: list[Constraint] = []
    storage_leases: list[StorageLease] = []
    storage_lease_labels: list[tuple[str, int]] = []
    issue_uses: list[IssueUse] = []
    hazards: list[Hazard] = []
    pressure_deltas: list[PressureDelta] = []
    feature_mask_words: list[int] = []
    encoding_field_values: list[EncodingFieldValue] = []
    operand_forms: list[CompiledOperandForm] = []
    operand_form_matches: list[CompiledOperandFormMatch] = []
    operand_form_operand_indices: list[int] = []
    descriptor_rows: list[dict[str, int]] = []
    schedule_rows: list[dict[str, int]] = []
    enum_domain_rows: list[dict[str, int]] = []

    for schedule_class in schedule_classes:
        issue_use_start = len(issue_uses)
        issue_uses.extend(schedule_class.issue_uses)
        hazard_start = len(hazards)
        hazards.extend(schedule_class.hazards)
        pressure_delta_start = len(pressure_deltas)
        pressure_deltas.extend(schedule_class.pressure_deltas)
        schedule_rows.append(
            {
                "issue_use_start": issue_use_start,
                "issue_use_count": len(schedule_class.issue_uses),
                "hazard_start": hazard_start,
                "hazard_count": len(schedule_class.hazards),
                "pressure_delta_start": pressure_delta_start,
                "pressure_delta_count": len(schedule_class.pressure_deltas),
            }
        )

    for enum_domain in enum_domains:
        value_start = len(enum_values)
        sorted_values = validation.validate_enum_domain(enum_domain)
        enum_values.extend(sorted_values)
        enum_domain_rows.append(
            {
                "value_start": value_start,
                "value_count": len(sorted_values),
            }
        )

    for descriptor in selected_descriptors:
        operand_start = len(operands)
        for operand in descriptor.operands:
            alt_group: tuple[tuple[int | None, tuple[RegClassAltFlag, ...]], ...] = tuple(
                (
                    None if reg_alt.reg_class is None else reg_class_ids[reg_alt.reg_class],
                    reg_alt.flags,
                )
                for reg_alt in operand.reg_alts
            )
            alt_start = reg_alt_group_starts.get(alt_group)
            if alt_start is None:
                alt_start = len(reg_class_alts)
                reg_alt_group_starts[alt_group] = alt_start
                reg_class_alts.extend(alt_group)
            operands.append(operand)
            operand_alt_starts.append(alt_start)
        immediate_start = len(immediates)
        for immediate in descriptor.immediates:
            slice_start = 0
            if immediate.encoding_slices:
                group_start = immediate_encoding_slice_group_starts.get(immediate.encoding_slices)
                if group_start is None:
                    group_start = len(immediate_encoding_slices)
                    immediate_encoding_slice_group_starts[immediate.encoding_slices] = group_start
                    immediate_encoding_slices.extend(immediate.encoding_slices)
                slice_start = group_start
            immediate_encoding_slice_starts.append(slice_start)
            immediates.append(immediate)
            immediate_enum_domain_ids.append(None if immediate.enum_domain is None else enum_domain_ids[immediate.enum_domain])
        if descriptor.effects:
            effect_start = effect_group_starts.get(descriptor.effects)
            if effect_start is None:
                effect_start = len(effects)
                effect_group_starts[descriptor.effects] = effect_start
                effects.extend(descriptor.effects)
        else:
            effect_start = 0
        constraint_start = len(constraints)
        constraints.extend(descriptor.constraints)
        if descriptor.storage_leases:
            storage_lease_start = storage_lease_group_starts.get(descriptor.storage_leases)
            if storage_lease_start is None:
                storage_lease_start = len(storage_leases)
                storage_lease_group_starts[descriptor.storage_leases] = storage_lease_start
                storage_leases.extend(descriptor.storage_leases)
                storage_lease_labels.extend((descriptor.key, i) for i in range(len(descriptor.storage_leases)))
        else:
            storage_lease_start = 0
        feature_mask_word_start = len(feature_mask_words)
        feature_mask_words.extend(descriptor.feature_mask_words)
        encoding_field_value_start = len(encoding_field_values)
        encoding_field_values.extend(descriptor.encoding_field_values)
        operand_form_start = len(operand_forms)
        for operand_form in descriptor.operand_forms:
            if operand_form.replacement_descriptor not in descriptor_ordinals:
                raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{operand_form.replacement_descriptor}' was not selected")
            compiled_form, compiled_matches, operand_map = _compile_operand_form(
                descriptor,
                descriptor_ordinals,
                selected_descriptors,
                operand_form,
                len(operand_form_matches),
                len(operand_form_operand_indices),
            )
            operand_forms.append(compiled_form)
            operand_form_matches.extend(compiled_matches)
            operand_form_operand_indices.extend(operand_map)
        descriptor_rows.append(
            {
                "operand_start": operand_start,
                "operand_count": len(descriptor.operands),
                "result_count": validation.validate_descriptor_operands(descriptor),
                "immediate_start": immediate_start,
                "immediate_count": len(descriptor.immediates),
                "effect_start": effect_start,
                "effect_count": len(descriptor.effects),
                "constraint_start": constraint_start,
                "constraint_count": len(descriptor.constraints),
                "storage_lease_start": storage_lease_start,
                "storage_lease_count": len(descriptor.storage_leases),
                "feature_mask_word_start": feature_mask_word_start,
                "feature_mask_word_count": len(descriptor.feature_mask_words),
                "encoding_field_value_start": encoding_field_value_start,
                "encoding_field_value_count": len(descriptor.encoding_field_values),
                "operand_form_start": operand_form_start,
                "operand_form_count": len(descriptor.operand_forms),
            }
        )

    descriptor_refs = sorted((descriptor.key, i) for i, descriptor in enumerate(selected_descriptors))
    seen_stable_ids: dict[int, str] = {}
    for descriptor in selected_descriptors:
        stable_id = descriptor_stable_id(descriptor.key)
        previous_key = seen_stable_ids.get(stable_id)
        if previous_key is not None:
            raise ValueError(f"descriptor '{descriptor.key}' stable ID collides with '{previous_key}'")
        seen_stable_ids[stable_id] = descriptor.key

    return CompiledDescriptorSet(
        spec=spec,
        descriptors=selected_descriptors,
        reg_classes=reg_classes,
        register_parts=register_parts,
        resources=resources,
        schedule_classes=schedule_classes,
        enum_domains=enum_domains,
        reg_class_ids=reg_class_ids,
        register_part_ids=register_part_ids,
        resource_ids=resource_ids,
        schedule_class_ids=schedule_class_ids,
        enum_domain_ids=enum_domain_ids,
        string_pool=string_pool,
        reg_class_alts=reg_class_alts,
        operands=operands,
        operand_alt_starts=operand_alt_starts,
        immediates=immediates,
        immediate_encoding_slices=immediate_encoding_slices,
        immediate_encoding_slice_starts=immediate_encoding_slice_starts,
        enum_values=enum_values,
        immediate_enum_domain_ids=immediate_enum_domain_ids,
        effects=effects,
        constraints=constraints,
        storage_leases=storage_leases,
        storage_lease_labels=storage_lease_labels,
        issue_uses=issue_uses,
        hazards=hazards,
        pressure_deltas=pressure_deltas,
        feature_mask_words=feature_mask_words,
        encoding_field_values=encoding_field_values,
        operand_forms=operand_forms,
        operand_form_matches=operand_form_matches,
        operand_form_operand_indices=operand_form_operand_indices,
        descriptor_rows=descriptor_rows,
        descriptor_refs=descriptor_refs,
        canonical_asm_form_ordinals=canonical_asm_form_ordinals,
        asm_forms=asm_forms,
        asm_operand_indices=asm_operand_indices,
        asm_immediates=asm_immediates,
        native_asm_values=native_asm_values,
        schedule_rows=schedule_rows,
        enum_domain_rows=enum_domain_rows,
    )
