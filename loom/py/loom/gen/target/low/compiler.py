# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler from low descriptor inputs to the dense intermediate model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from loom.gen.support.string_pool import CStringPool
from loom.gen.target.low import validation
from loom.gen.target.low.compiled import (
    CompiledAsmForm,
    CompiledAsmImmediate,
    CompiledAsmOperandSegment,
    CompiledAsmTableStorage,
    CompiledDescriptorSet,
    CompiledNativeAsmValue,
    CompiledOperandForm,
    CompiledOperandFormMatch,
    DescriptorAllowlist,
    append_interned_sequence,
)
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    LOW_DESCRIPTOR_SET_ORDINAL_NONE,
    AsmForm,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectKind,
    EncodingFieldValue,
    EnumValue,
    Hazard,
    Immediate,
    ImmediateEncodingSlice,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    NativeAsmValue,
    NativeAsmValueKind,
    Operand,
    OperandFlag,
    OperandForm,
    OperandFormImmediateAction,
    OperandRole,
    PressureDelta,
    RegClassAltFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    StorageLease,
    descriptor_stable_id,
)


@dataclass(frozen=True, slots=True)
class _CompiledOperandRow:
    """Operand row with its derived parallel-table fields."""

    # Authored operand semantics emitted into the runtime row.
    operand: Operand
    # Source value position derived from the descriptor operand layout.
    source_value_index: int | None
    # First row in the shared register-class-alternative table.
    reg_class_alt_start: int
    # Whether the result constraint permits rematerialization.
    rematerializable: bool


@dataclass(frozen=True, slots=True)
class _CompiledImmediateRow:
    """Immediate row with its derived parallel-table fields."""

    # Authored immediate semantics emitted into the runtime row.
    immediate: Immediate
    # First row in the shared immediate-encoding-slice table.
    encoding_slice_start: int
    # Resolved enum-domain identifier, or None for non-enum immediates.
    enum_domain_id: int | None


_SEMANTIC_INSTRUCTION_CLASSES = (
    ("matrix.smfmac", (InstructionClass.SMFMAC,)),
    ("matrix.mfma", (InstructionClass.MFMA,)),
    ("matrix.swmmac", (InstructionClass.SWMMAC,)),
    ("matrix.wmma", (InstructionClass.WMMA,)),
    ("matrix", (InstructionClass.MATRIX,)),
    ("dot", (InstructionClass.DOT,)),
    ("memory.cache", (InstructionClass.CACHE,)),
    ("memory.global", (InstructionClass.GLOBAL_MEMORY,)),
    ("memory.workgroup", (InstructionClass.LOCAL_MEMORY,)),
    ("memory.stack", (InstructionClass.PRIVATE_MEMORY,)),
    ("memory.private", (InstructionClass.PRIVATE_MEMORY,)),
    ("memory.generic", (InstructionClass.GENERIC_MEMORY,)),
    ("memory.load", (InstructionClass.GENERIC_MEMORY,)),
    ("memory.store", (InstructionClass.GENERIC_MEMORY,)),
    ("memory.hal", (InstructionClass.GENERIC_MEMORY,)),
    ("control.branch", (InstructionClass.BRANCH,)),
    ("control.cond_branch", (InstructionClass.BRANCH,)),
    ("control.return", (InstructionClass.BRANCH,)),
    ("control.call", (InstructionClass.BRANCH,)),
    ("control.barrier", (InstructionClass.BARRIER,)),
    ("control", (InstructionClass.CONTROL,)),
    ("convert", (InstructionClass.CONVERSION,)),
    ("register.copy", (InstructionClass.REGISTER_MOVE,)),
    ("integer.move", (InstructionClass.REGISTER_MOVE,)),
)

_RESOURCE_INSTRUCTION_CLASSES = {
    ResourceKind.SCALAR_ALU: InstructionClass.SCALAR_ALU,
    ResourceKind.VECTOR_ALU: InstructionClass.VECTOR_ALU,
    ResourceKind.MATRIX: InstructionClass.MATRIX,
    ResourceKind.CONTROL: InstructionClass.CONTROL,
}

_MEMORY_INSTRUCTION_CLASSES = frozenset(
    {
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.GLOBAL_LOAD,
        InstructionClass.GLOBAL_STORE,
        InstructionClass.BUFFER_LOAD,
        InstructionClass.BUFFER_STORE,
        InstructionClass.FLAT_MEMORY,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.SCALAR_MEMORY,
        InstructionClass.PRIVATE_MEMORY,
        InstructionClass.GENERIC_MEMORY,
    }
)

_DERIVED_OPERAND_FLAGS = frozenset((OperandFlag.TIED, OperandFlag.EARLY_CLOBBER))

_INSTRUCTION_CLASS_IMPLICATIONS = {
    InstructionClass.SMFMAC: (InstructionClass.MFMA,),
    InstructionClass.MFMA: (InstructionClass.MATRIX,),
    InstructionClass.SWMMAC: (InstructionClass.WMMA,),
    InstructionClass.WMMA: (InstructionClass.MATRIX,),
    InstructionClass.BRANCH: (InstructionClass.CONTROL,),
    InstructionClass.BARRIER: (InstructionClass.CONTROL,),
    InstructionClass.GLOBAL_LOAD: (InstructionClass.GLOBAL_MEMORY,),
    InstructionClass.GLOBAL_STORE: (InstructionClass.GLOBAL_MEMORY,),
    InstructionClass.BUFFER_LOAD: (InstructionClass.GLOBAL_MEMORY,),
    InstructionClass.BUFFER_STORE: (InstructionClass.GLOBAL_MEMORY,),
    InstructionClass.FLAT_MEMORY: (InstructionClass.GLOBAL_MEMORY,),
}


def _semantic_tag_has_prefix(semantic_tag: str, prefix: str) -> bool:
    return semantic_tag == prefix or semantic_tag.startswith(f"{prefix}.")


def derive_instruction_classes(
    descriptor: Descriptor,
    schedule_class: ScheduleClass,
    resources: dict[str, Resource],
) -> tuple[InstructionClass, ...]:
    classes = set(descriptor.instruction_classes)
    if len(classes) != len(descriptor.instruction_classes):
        raise ValueError(f"descriptor '{descriptor.key}' repeats an instruction class")
    if len(set(schedule_class.instruction_classes)) != len(schedule_class.instruction_classes):
        raise ValueError(f"schedule class '{schedule_class.name}' repeats an instruction class")
    classes.update(schedule_class.instruction_classes)

    for issue_use in schedule_class.issue_uses:
        resource = resources.get(issue_use.resource)
        if resource is None:
            raise ValueError(f"descriptor '{descriptor.key}' schedule class '{schedule_class.name}' references unknown resource '{issue_use.resource}'")
        instruction_class = _RESOURCE_INSTRUCTION_CLASSES.get(resource.kind)
        if instruction_class is not None:
            classes.add(instruction_class)
    if ScheduleClassFlag.CONTROL in schedule_class.flags:
        classes.add(InstructionClass.CONTROL)

    semantic_tag = descriptor.semantic_tag
    if semantic_tag is not None:
        for prefix, semantic_classes in _SEMANTIC_INSTRUCTION_CLASSES:
            if _semantic_tag_has_prefix(semantic_tag, prefix):
                classes.update(semantic_classes)
        if "atomic" in semantic_tag.split("."):
            classes.add(InstructionClass.ATOMIC)

    effect_kinds = {effect.kind for effect in descriptor.effects}
    if EffectKind.BARRIER in effect_kinds:
        classes.add(InstructionClass.BARRIER)
    if EffectKind.CALL in effect_kinds:
        classes.add(InstructionClass.BRANCH)
    if EffectKind.CONTROL in effect_kinds:
        classes.add(InstructionClass.CONTROL)

    has_memory_effect = bool({EffectKind.READ, EffectKind.WRITE}.intersection(effect_kinds))
    has_memory_resource = any(resources[issue_use.resource].kind in (ResourceKind.LOAD, ResourceKind.STORE) for issue_use in schedule_class.issue_uses)
    if (has_memory_effect or has_memory_resource) and not _MEMORY_INSTRUCTION_CLASSES.intersection(classes):
        classes.add(InstructionClass.GENERIC_MEMORY)

    pending_classes = list(classes)
    while pending_classes:
        instruction_class = pending_classes.pop()
        for implied_class in _INSTRUCTION_CLASS_IMPLICATIONS.get(instruction_class, ()):
            if implied_class not in classes:
                classes.add(implied_class)
                pending_classes.append(implied_class)

    if InstructionClass.OTHER in classes and len(classes) != 1:
        class_names = ", ".join(sorted(item.name for item in classes))
        raise ValueError(f"descriptor '{descriptor.key}' combines the exclusive OTHER instruction class with {class_names}")
    if not classes:
        raise ValueError(f"descriptor '{descriptor.key}' has no generated instruction class; classify it explicitly or use OTHER")
    if InstructionClass.PRIVATE_MEMORY in classes and InstructionClass.GLOBAL_MEMORY in classes:
        raise ValueError(f"descriptor '{descriptor.key}' combines private and global memory instruction classes")
    if InstructionClass.ATOMIC in classes and not has_memory_effect:
        raise ValueError(f"descriptor '{descriptor.key}' has the atomic instruction class without a read or write effect")
    if InstructionClass.BARRIER in classes and EffectKind.BARRIER not in effect_kinds:
        raise ValueError(f"descriptor '{descriptor.key}' has the barrier instruction class without a barrier effect")
    read_classes = {
        InstructionClass.GLOBAL_LOAD,
        InstructionClass.BUFFER_LOAD,
    }
    if read_classes.intersection(classes) and EffectKind.READ not in effect_kinds:
        raise ValueError(f"descriptor '{descriptor.key}' has a load instruction class without a read effect")
    write_classes = {
        InstructionClass.GLOBAL_STORE,
        InstructionClass.BUFFER_STORE,
    }
    if write_classes.intersection(classes) and EffectKind.WRITE not in effect_kinds:
        raise ValueError(f"descriptor '{descriptor.key}' has a store instruction class without a write effect")

    return tuple(instruction_class for instruction_class in InstructionClass if instruction_class in classes)


def derive_descriptor_projections(
    descriptor: Descriptor,
    operand_layout: validation.DescriptorOperandLayout,
) -> Descriptor:
    """Projects validated descriptor constraints onto compact runtime flags."""

    has_barrier_effect = any(effect.kind is EffectKind.BARRIER for effect in descriptor.effects)
    has_barrier_flag = DescriptorFlag.BARRIER in descriptor.flags
    if has_barrier_flag and not has_barrier_effect:
        raise ValueError(f"descriptor '{descriptor.key}' has the barrier flag without a barrier effect")
    has_early_clobber_constraint = any(constraint.kind is ConstraintKind.EARLY_CLOBBER for constraint in descriptor.constraints)
    has_early_clobber_flag = DescriptorFlag.EARLY_CLOBBER in descriptor.flags
    if has_early_clobber_flag and not has_early_clobber_constraint:
        raise ValueError(f"descriptor '{descriptor.key}' has the early-clobber flag without an early-clobber constraint")

    derived_flags = list(descriptor.flags)
    if has_barrier_effect and not has_barrier_flag:
        derived_flags.append(DescriptorFlag.BARRIER)
    if has_early_clobber_constraint and not has_early_clobber_flag:
        derived_flags.append(DescriptorFlag.EARLY_CLOBBER)
    has_variadic_operand = operand_layout.has_variadic_operands
    has_variadic_flag = DescriptorFlag.VARIADIC_OPERANDS in descriptor.flags
    if has_variadic_flag and not has_variadic_operand:
        raise ValueError(f"descriptor '{descriptor.key}' has the variadic-operands flag without a variadic operand")
    if has_variadic_operand and not has_variadic_flag:
        derived_flags.append(DescriptorFlag.VARIADIC_OPERANDS)

    operand_flags: list[list[OperandFlag]] = []
    for operand in descriptor.operands:
        authored_projection_flags = _DERIVED_OPERAND_FLAGS.intersection(operand.flags)
        if authored_projection_flags:
            names = ", ".join(sorted(flag.name.lower() for flag in authored_projection_flags))
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' authors derived projection flag(s): {names}")
        operand_flags.append(list(operand.flags))

    for constraint in descriptor.constraints:
        if constraint.kind is ConstraintKind.TIED:
            result_index = constraint.lhs_operand_index
            operand_index = constraint.rhs_operand_index
            assert operand_index is not None
            for tied_index in (result_index, operand_index):
                if OperandFlag.TIED not in operand_flags[tied_index]:
                    operand_flags[tied_index].append(OperandFlag.TIED)
            if OperandFlag.STORAGE_CONTINUATION in operand_flags[operand_index] and OperandFlag.STORAGE_CONTINUATION not in operand_flags[result_index]:
                operand_flags[result_index].append(OperandFlag.STORAGE_CONTINUATION)
        elif constraint.kind is ConstraintKind.EARLY_CLOBBER:
            flags = operand_flags[constraint.lhs_operand_index]
            if OperandFlag.EARLY_CLOBBER not in flags:
                flags.append(OperandFlag.EARLY_CLOBBER)

    projected_flags = tuple(derived_flags)
    projection_changed = projected_flags != descriptor.flags
    projected_operands: list[Operand] = []
    for operand, flags in zip(
        descriptor.operands,
        operand_flags,
        strict=True,
    ):
        projected_operand_flags = tuple(flags)
        if projected_operand_flags == operand.flags:
            projected_operands.append(operand)
            continue
        projected_operands.append(replace(operand, flags=projected_operand_flags))
        projection_changed = True

    if not projection_changed:
        return descriptor
    return replace(
        descriptor,
        flags=projected_flags,
        operands=tuple(projected_operands),
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

    if asm_form.result_value_types:
        if len(asm_form.result_value_types) != len(result_indices):
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' has {len(asm_form.result_value_types)} result value types for {len(result_indices)} results")
        if not any(value_type is not None for value_type in asm_form.result_value_types):
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' has an empty result value type recipe")
        operand_inferred_result_indices = {constraint.lhs_operand_index for constraint in descriptor.constraints if constraint.kind in (ConstraintKind.TIED, ConstraintKind.DESTRUCTIVE)}
        for result_index, (operand_index, value_type) in enumerate(zip(result_indices, asm_form.result_value_types, strict=True)):
            if value_type is not None and operand_index in operand_inferred_result_indices:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' result {result_index} is operand-inferred and must use the exact operand type")

    packet_operand_roles = {
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    }
    if asm_form.operands and asm_form.operand_segments:
        raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' uses both flat operands and operand segments")
    asm_operand_fields = asm_form.operands
    if asm_form.operand_segments:
        asm_operand_fields = tuple(field_name for segment in asm_form.operand_segments for field_name in segment.operands)

    operand_order = []
    for field_name in asm_operand_fields:
        operand_index = operand_indices.get(field_name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' operand references unknown operand field '{field_name}'")
        operand = descriptor.operands[operand_index]
        if operand.role not in packet_operand_roles:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' operand field '{field_name}' does not name an explicit packet operand")
        operand_order.append(operand_index)

    operand_segments = []
    for segment_index, segment in enumerate(asm_form.operand_segments):
        has_variadic_operand = False
        for field_index, field_name in enumerate(segment.operands):
            operand_index = operand_indices[field_name]
            is_variadic = OperandFlag.VARIADIC in descriptor.operands[operand_index].flags
            if is_variadic:
                if segment_index != len(asm_form.operand_segments) - 1 or field_index != len(segment.operands) - 1:
                    raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' variadic operand '{field_name}' must terminate the final operand segment")
                has_variadic_operand = True
        operand_segments.append(
            CompiledAsmOperandSegment(
                delimiter=segment.delimiter,
                operand_count=len(segment.operands),
                has_variadic_operand=has_variadic_operand,
            )
        )

    variadic_operand_fields = {operand.field_name for operand in descriptor.operands if OperandFlag.VARIADIC in operand.flags}
    if variadic_operand_fields:
        if not asm_form.operand_segments:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' must place its variadic operand in a delimited operand segment")
        referenced_variadic_fields = variadic_operand_fields.intersection(asm_operand_fields)
        if referenced_variadic_fields != variadic_operand_fields:
            missing_fields = ", ".join(sorted(variadic_operand_fields - referenced_variadic_fields))
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' omits variadic operand field(s): {missing_fields}")

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
        operand_segments=tuple(operand_segments),
        result_value_types=asm_form.result_value_types,
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
    target_format_id = value.target_format_id

    def require_field_name() -> str:
        if field_name is None or field_name == "":
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} must name a descriptor field")
        return field_name

    def reject_literal_and_bit_width() -> None:
        if literal is not None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} unexpectedly specifies a literal")
        if bit_width != 0:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} unexpectedly specifies a bit width")
        if target_format_id != 0:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native value {value_ordinal} unexpectedly specifies a target format")

    if kind in (
        NativeAsmValueKind.LITERAL,
        NativeAsmValueKind.MODIFIER_LITERAL,
    ):
        if field_name is not None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} unexpectedly names field '{field_name}'")
        if literal is None or literal == "":
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} has no spelling")
        if len(literal.encode()) > 255:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} exceeds 255 bytes")
        if bit_width != 0:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} unexpectedly specifies a bit width")
        if target_format_id != 0:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native literal value {value_ordinal} unexpectedly specifies a target format")
        return CompiledNativeAsmValue(
            kind=kind,
            index=0,
            bit_width=0,
            target_format_id=0,
            literal_label=string_pool.intern(
                scoped_label("asm_native_value", str(value_ordinal)),
                literal,
            ),
            literal=literal,
        )

    if kind is NativeAsmValueKind.REGISTER_PART:
        reject_literal_and_bit_width()
        name = require_field_name()
        operand_index = operand_indices.get(name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native register-part value references unknown operand field '{name}'")
        operand = descriptor.operands[operand_index]
        if operand.role is not OperandRole.RESULT and operand.role not in packet_operand_roles:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native register-part field '{name}' does not name a result or explicit packet operand")
        if operand.register_part is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native register-part field '{name}' names a full-register operand")
        return CompiledNativeAsmValue(
            kind=kind,
            index=operand_index,
            bit_width=0,
            target_format_id=0,
            literal_label=None,
            literal=None,
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
            target_format_id=0,
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
            target_format_id=0,
            literal_label=None,
            literal=None,
        )

    if kind in (
        NativeAsmValueKind.IMMEDIATE_I64,
        NativeAsmValueKind.IMMEDIATE_UNSIGNED_HEX,
        NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
    ):
        if kind is not NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT and literal is not None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native immediate value {value_ordinal} unexpectedly specifies a literal")
        name = require_field_name()
        immediate_index = immediate_indices.get(name)
        if immediate_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native immediate references unknown immediate field '{name}'")
        if kind is NativeAsmValueKind.IMMEDIATE_I64:
            if bit_width != 0:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native i64 immediate '{name}' unexpectedly specifies a bit width")
        elif kind is NativeAsmValueKind.IMMEDIATE_UNSIGNED_HEX:
            if bit_width <= 0 or bit_width > 32:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native unsigned-hex immediate '{name}' bit width must be in [1, 32]")
        elif kind is NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT:
            if bit_width < 0 or bit_width > 0xFF:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native target-format immediate '{name}' bit width must be in [0, 255]")
            if target_format_id <= 0 or target_format_id > 0xFF:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native target-format immediate '{name}' target format must be in [1, 255]")
            if literal is not None:
                if literal == "":
                    raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native target-format immediate '{name}' has an empty literal")
                if len(literal.encode()) > 255:
                    raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native target-format immediate '{name}' literal exceeds 255 bytes")
        if kind is not NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT and target_format_id != 0:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' native immediate '{name}' unexpectedly specifies a target format")
        return CompiledNativeAsmValue(
            kind=kind,
            index=immediate_index,
            bit_width=bit_width,
            target_format_id=target_format_id,
            literal_label=(
                string_pool.intern(
                    scoped_label("asm_native_value", str(value_ordinal)),
                    literal,
                )
                if literal is not None
                else None
            ),
            literal=literal,
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


def compile_descriptor_set(
    spec: DescriptorSet,
    allowlist: DescriptorAllowlist | None = None,
    *,
    allow_ambiguous_asm_mnemonics: bool = False,
    required_schedule_class_names: Sequence[str] = (),
) -> CompiledDescriptorSet:
    if spec.generator_version == 0:
        raise ValueError(f"descriptor set '{spec.key}' has zero generator version")
    reg_class_inputs = _dedupe_by_name(spec.reg_classes, lambda item: item.name)
    physical_register_inputs = _dedupe_by_name(spec.physical_registers, lambda item: item.name)
    validation.validate_register_classes(
        spec.key,
        tuple(reg_class_inputs.values()),
        tuple(physical_register_inputs.values()),
    )
    validation.validate_schedule_model(spec)
    register_part_inputs = _dedupe_by_name(spec.register_parts, lambda item: item.name)
    resource_inputs = _dedupe_by_name(spec.resources, lambda item: item.name)
    schedule_inputs = _dedupe_by_name(spec.schedule_classes, lambda item: item.name)
    enum_domain_inputs = _dedupe_by_name(spec.enum_domains, lambda item: item.name)
    _dedupe_by_name(spec.descriptors, lambda item: item.key)

    operand_layouts_by_descriptor: dict[str, validation.DescriptorOperandLayout] = {}
    rematerializable_results_by_descriptor: dict[str, tuple[int, ...]] = {}
    source_value_indices_by_descriptor: dict[str, tuple[int | None, ...]] = {}
    projected_descriptors_by_key: dict[str, Descriptor] = {}
    for descriptor in spec.descriptors:
        operand_layout = validation.validate_descriptor_operands(descriptor)
        operand_layouts_by_descriptor[descriptor.key] = operand_layout
        result_count = operand_layout.result_count
        source_value_indices_by_descriptor[descriptor.key] = validation.descriptor_operand_source_value_indices(
            descriptor,
            result_count,
        )
        rematerializable_results_by_descriptor[descriptor.key] = validation.validate_descriptor_constraints(descriptor)
        validation.validate_descriptor_op_kind(descriptor, result_count)
        validation.validate_descriptor_storage_continuations(
            descriptor,
            register_part_inputs,
        )
        projected_descriptors_by_key[descriptor.key] = derive_descriptor_projections(
            descriptor,
            operand_layout,
        )
    validation.validate_physical_descriptor_set(spec)

    source_descriptors = _select_descriptors(spec, allowlist)
    selected_descriptors = [projected_descriptors_by_key[descriptor.key] for descriptor in source_descriptors]
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
    used_schedule_names = set(required_schedule_class_names)
    used_timing_event_names: set[str] = set()
    unknown_required_schedule_names = sorted(used_schedule_names - schedule_inputs.keys())
    if unknown_required_schedule_names:
        raise ValueError(f"descriptor set '{spec.key}' requires unknown schedule classes: {', '.join(unknown_required_schedule_names)}")
    used_enum_domain_names: set[str] = set()
    for descriptor in selected_descriptors:
        result_count = operand_layouts_by_descriptor[descriptor.key].result_count
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
            validation.validate_u64(
                immediate.value_step,
                f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' value step",
            )
            if immediate.value_step == 0:
                raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' has zero value step")
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
                if immediate.value_step != 1:
                    raise ValueError(f"descriptor '{descriptor.key}' enum immediate '{immediate.field_name}' has non-unit value step {immediate.value_step}")
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
            if operand.read_event is not None:
                used_timing_event_names.add(operand.read_event)
            if operand.write_event is not None:
                used_timing_event_names.add(operand.write_event)
        for effect in descriptor.effects:
            if effect.timing_event is not None:
                used_timing_event_names.add(effect.timing_event)
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

    instruction_classes = [
        derive_instruction_classes(
            descriptor,
            schedule_inputs[descriptor.schedule_class],
            resource_inputs,
        )
        for descriptor in selected_descriptors
    ]

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
    physical_registers = list(spec.physical_registers)
    register_parts = [part for part in spec.register_parts if part.name in used_register_part_names]
    resources = [resource for resource in spec.resources if resource.name in used_resource_names]
    schedule_classes = [schedule_class for schedule_class in spec.schedule_classes if schedule_class.name in used_schedule_names]
    timing_events = [timing_event for timing_event in spec.timing_events if timing_event.name in used_timing_event_names]
    enum_domains = [domain for domain in spec.enum_domains if domain.name in used_enum_domain_names]

    validation.validate_u16_table_count(len(reg_classes), f"descriptor set '{spec.key}' register class")
    validation.validate_u16_table_count(
        len(physical_registers),
        f"descriptor set '{spec.key}' physical register",
    )
    validation.validate_u16_table_count(len(register_parts), f"descriptor set '{spec.key}' register part")
    reg_class_ids = {reg_class.name: i for i, reg_class in enumerate(reg_classes)}
    physical_register_ids = {physical_register.name: i for i, physical_register in enumerate(physical_registers)}
    register_part_ids = {part.name: i for i, part in enumerate(register_parts)}
    resource_ids = {resource.name: i for i, resource in enumerate(resources)}
    schedule_class_ids = {schedule_class.name: i for i, schedule_class in enumerate(schedule_classes)}
    timing_event_ids = {timing_event.name: i for i, timing_event in enumerate(timing_events)}
    event_separations = sorted(
        (separation for separation in spec.event_separations if separation.producer_event in timing_event_ids and separation.consumer_event in timing_event_ids),
        key=lambda separation: (
            timing_event_ids[separation.producer_event],
            timing_event_ids[separation.consumer_event],
        ),
    )
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
    for physical_register in physical_registers:
        string_pool.intern(
            f"physical_register_{physical_register.name}",
            physical_register.name,
        )
    for part in register_parts:
        string_pool.intern(f"register_part_{part.name}", part.name)
    for resource in resources:
        string_pool.intern(f"resource_{resource.name}", resource.name)
    for schedule_class in schedule_classes:
        string_pool.intern(f"schedule_{schedule_class.name}", schedule_class.name)
    for timing_event in timing_events:
        string_pool.intern(f"timing_event_{timing_event.name}", timing_event.name)
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
    validation.validate_u16_table_count(len(asm_forms), f"descriptor set '{spec.key}' asm form")
    canonical_asm_form_ordinals: list[int | None] = [None] * len(selected_descriptors)
    asm_form_counts_by_descriptor = [0] * len(selected_descriptors)
    for asm_form_ordinal, asm_form in enumerate(asm_forms):
        descriptor_ordinal = asm_form.descriptor_ordinal
        asm_form_counts_by_descriptor[descriptor_ordinal] += 1
        canonical_asm_form_ordinals[descriptor_ordinal] = asm_form_ordinal
    for descriptor_ordinal, form_count in enumerate(asm_form_counts_by_descriptor):
        if form_count != 1:
            canonical_asm_form_ordinals[descriptor_ordinal] = None

    asm_table_storage = CompiledAsmTableStorage()
    asm_table_storage.append_forms(asm_forms)

    reg_class_alts: list[tuple[int | None, tuple[RegClassAltFlag, ...]]] = []
    reg_alt_group_starts: dict[tuple[tuple[int | None, tuple[RegClassAltFlag, ...]], ...], int] = {}
    immediate_encoding_slice_group_starts: dict[tuple[ImmediateEncodingSlice, ...], int] = {}
    effect_group_starts: dict[tuple[Effect, ...], int] = {}
    constraint_group_starts: dict[tuple[Constraint, ...], int] = {}
    storage_lease_group_starts: dict[tuple[StorageLease, ...], int] = {}
    feature_mask_group_starts: dict[tuple[int, ...], int] = {}
    encoding_field_value_group_starts: dict[tuple[EncodingFieldValue, ...], int] = {}
    compiled_operand_rows: list[_CompiledOperandRow] = []
    operand_group_starts: dict[tuple[_CompiledOperandRow, ...], int] = {}
    compiled_immediate_rows: list[_CompiledImmediateRow] = []
    immediate_group_starts: dict[tuple[_CompiledImmediateRow, ...], int] = {}
    immediate_encoding_slices: list[ImmediateEncodingSlice] = []
    enum_values: list[EnumValue] = []
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

    physical_register_atomic_units: list[int] = []
    physical_register_atomic_unit_starts: list[int] = []
    for physical_register in physical_registers:
        physical_register_atomic_unit_starts.append(len(physical_register_atomic_units))
        physical_register_atomic_units.extend(physical_register.atomic_units)

    physical_register_candidate_ids: list[int] = []
    physical_register_candidate_starts: list[int] = []
    for reg_class in reg_classes:
        physical_register_candidate_starts.append(len(physical_register_candidate_ids))
        physical_register_candidate_ids.extend(physical_register_ids[name] for name in reg_class.physical_registers)

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
        rematerializable_result_set = set(rematerializable_results_by_descriptor[descriptor.key])
        descriptor_operand_rows: list[_CompiledOperandRow] = []
        for operand_index, (operand, source_value_index) in enumerate(
            zip(
                descriptor.operands,
                source_value_indices_by_descriptor[descriptor.key],
                strict=True,
            )
        ):
            alt_group: tuple[tuple[int | None, tuple[RegClassAltFlag, ...]], ...] = tuple(
                (
                    None if reg_alt.reg_class is None else reg_class_ids[reg_alt.reg_class],
                    reg_alt.flags,
                )
                for reg_alt in operand.reg_alts
            )
            reg_class_alt_start, _ = append_interned_sequence(
                alt_group,
                reg_class_alts,
                reg_alt_group_starts,
            )
            descriptor_operand_rows.append(
                _CompiledOperandRow(
                    operand=operand,
                    source_value_index=source_value_index,
                    reg_class_alt_start=reg_class_alt_start,
                    rematerializable=(operand_index in rematerializable_result_set),
                )
            )
        operand_start, _ = append_interned_sequence(
            descriptor_operand_rows,
            compiled_operand_rows,
            operand_group_starts,
        )

        descriptor_immediate_rows: list[_CompiledImmediateRow] = []
        for immediate in descriptor.immediates:
            encoding_slice_start, _ = append_interned_sequence(
                immediate.encoding_slices,
                immediate_encoding_slices,
                immediate_encoding_slice_group_starts,
            )
            descriptor_immediate_rows.append(
                _CompiledImmediateRow(
                    immediate=immediate,
                    encoding_slice_start=encoding_slice_start,
                    enum_domain_id=(None if immediate.enum_domain is None else enum_domain_ids[immediate.enum_domain]),
                )
            )
        immediate_start, _ = append_interned_sequence(
            descriptor_immediate_rows,
            compiled_immediate_rows,
            immediate_group_starts,
        )
        effect_start, _ = append_interned_sequence(
            descriptor.effects,
            effects,
            effect_group_starts,
        )
        constraint_start, _ = append_interned_sequence(
            descriptor.constraints,
            constraints,
            constraint_group_starts,
        )
        storage_lease_start, inserted_storage_leases = append_interned_sequence(
            descriptor.storage_leases,
            storage_leases,
            storage_lease_group_starts,
        )
        if inserted_storage_leases:
            storage_lease_labels.extend((descriptor.key, i) for i in range(len(descriptor.storage_leases)))
        feature_mask_word_start, _ = append_interned_sequence(
            descriptor.feature_mask_words,
            feature_mask_words,
            feature_mask_group_starts,
        )
        encoding_field_value_start, _ = append_interned_sequence(
            descriptor.encoding_field_values,
            encoding_field_values,
            encoding_field_value_group_starts,
        )
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
                "result_count": operand_layouts_by_descriptor[descriptor.key].result_count,
                "minimum_packet_operand_count": operand_layouts_by_descriptor[descriptor.key].minimum_packet_operand_count,
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

    operands = [row.operand for row in compiled_operand_rows]
    operand_source_value_indices = [row.source_value_index for row in compiled_operand_rows]
    operand_alt_starts = [row.reg_class_alt_start for row in compiled_operand_rows]
    operand_rematerializable = [row.rematerializable for row in compiled_operand_rows]
    immediates = [row.immediate for row in compiled_immediate_rows]
    immediate_encoding_slice_starts = [row.encoding_slice_start for row in compiled_immediate_rows]
    immediate_enum_domain_ids = [row.enum_domain_id for row in compiled_immediate_rows]

    compact_start_tables = (
        ("descriptor", selected_descriptors),
        ("asm operand index", asm_table_storage.operand_indices),
        ("asm operand segment", asm_table_storage.operand_segments),
        ("asm result value type", asm_table_storage.result_value_types),
        ("asm immediate", asm_table_storage.immediates),
        ("native asm value", asm_table_storage.native_values),
        ("operand", operands),
        ("immediate", immediates),
        ("effect", effects),
        ("constraint", constraints),
        ("storage lease", storage_leases),
        ("feature mask word", feature_mask_words),
        ("encoding field value", encoding_field_values),
        ("operand form", operand_forms),
    )
    for table_name, rows in compact_start_tables:
        validation.validate_u16_table_count(len(rows), f"descriptor set '{spec.key}' {table_name}")

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
        source_descriptors=source_descriptors,
        descriptors=selected_descriptors,
        instruction_classes=instruction_classes,
        reg_classes=reg_classes,
        physical_registers=physical_registers,
        physical_register_candidate_ids=physical_register_candidate_ids,
        physical_register_candidate_starts=physical_register_candidate_starts,
        physical_register_atomic_units=physical_register_atomic_units,
        physical_register_atomic_unit_starts=physical_register_atomic_unit_starts,
        register_parts=register_parts,
        resources=resources,
        schedule_classes=schedule_classes,
        timing_events=timing_events,
        event_separations=event_separations,
        enum_domains=enum_domains,
        reg_class_ids=reg_class_ids,
        register_part_ids=register_part_ids,
        resource_ids=resource_ids,
        schedule_class_ids=schedule_class_ids,
        timing_event_ids=timing_event_ids,
        enum_domain_ids=enum_domain_ids,
        string_pool=string_pool,
        reg_class_alts=reg_class_alts,
        operands=operands,
        operand_source_value_indices=operand_source_value_indices,
        operand_alt_starts=operand_alt_starts,
        operand_rematerializable=operand_rematerializable,
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
        asm_table_storage=asm_table_storage,
        schedule_rows=schedule_rows,
        enum_domain_rows=enum_domain_rows,
    )
