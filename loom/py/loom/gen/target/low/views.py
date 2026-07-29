# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor-set view construction for compiled low descriptor sets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from loom.gen.target.low import compiler, validation
from loom.gen.target.low.compiled import (
    CompiledAsmForm,
    CompiledDescriptorSet,
    CompiledOperandForm,
    DescriptorSetView,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    Hazard,
    InstructionClass,
    IssueUse,
    PressureDelta,
    RegClass,
    Resource,
    ScheduleClass,
)


class _NamedTableItem(Protocol):
    name: str


@dataclass(frozen=True, slots=True)
class _ViewScheduleTables:
    schedule_classes: tuple[ScheduleClass, ...]
    issue_uses: tuple[IssueUse, ...]
    hazards: tuple[Hazard, ...]
    pressure_deltas: tuple[PressureDelta, ...]
    schedule_rows: tuple[dict[str, int], ...]
    uses_storage_schedule_classes: bool


def _overlay_view_table[NamedTableItemT: _NamedTableItem](
    storage_items: Sequence[NamedTableItemT],
    view_items: Sequence[NamedTableItemT],
) -> tuple[NamedTableItemT, ...]:
    view_items_by_name = {item.name: item for item in view_items}
    return tuple(view_items_by_name.get(storage_item.name, storage_item) for storage_item in storage_items)


def _validate_shared_view_table[NamedTableItemT: _NamedTableItem](
    storage_items: Sequence[NamedTableItemT],
    view_items: Sequence[NamedTableItemT],
    *,
    view_spec: DescriptorSet,
    storage_spec: DescriptorSet,
    table_name: str,
) -> None:
    storage_items_by_name = {item.name: item for item in storage_items}
    for view_item in view_items:
        storage_item = storage_items_by_name.get(view_item.name)
        if storage_item is None:
            continue
        if view_item != storage_item:
            raise ValueError(f"descriptor set view '{view_spec.key}' {table_name} '{view_item.name}' differs from storage set '{storage_spec.key}'")


def _descriptor_refs_for_ordinals(
    descriptors: Sequence[Descriptor],
    descriptor_ordinals: Sequence[int],
) -> list[tuple[str, int]]:
    return sorted((descriptors[descriptor_ordinal].key, view_ordinal) for view_ordinal, descriptor_ordinal in enumerate(descriptor_ordinals))


def _clone_operand_form_for_view(
    operand_form: CompiledOperandForm,
    replacement_descriptor_ordinal: int,
) -> CompiledOperandForm:
    return CompiledOperandForm(
        replacement_descriptor_ordinal=replacement_descriptor_ordinal,
        source_immediate_index=operand_form.source_immediate_index,
        replacement_immediate_index=operand_form.replacement_immediate_index,
        immediate_match_index=operand_form.immediate_match_index,
        immediate_action=operand_form.immediate_action,
        match_start=operand_form.match_start,
        match_count=operand_form.match_count,
        operand_map_start=operand_form.operand_map_start,
        operand_map_count=operand_form.operand_map_count,
    )


def _asm_forms_have_duplicate_mnemonics(
    asm_forms: Sequence[CompiledAsmForm],
) -> bool:
    seen_mnemonics: set[str] = set()
    for asm_form in asm_forms:
        if asm_form.mnemonic in seen_mnemonics:
            return True
        seen_mnemonics.add(asm_form.mnemonic)
    return False


def _validate_view_asm_forms_unique(
    view_spec: DescriptorSet,
    asm_forms: Sequence[CompiledAsmForm],
) -> None:
    seen_mnemonics: dict[str, int] = {}
    for asm_form_ordinal, asm_form in enumerate(asm_forms):
        previous_ordinal = seen_mnemonics.get(asm_form.mnemonic)
        if previous_ordinal is not None:
            previous_descriptor = view_spec.descriptors[asm_forms[previous_ordinal].descriptor_ordinal]
            descriptor = view_spec.descriptors[asm_form.descriptor_ordinal]
            raise ValueError(f"descriptor set view '{view_spec.key}' has ambiguous asm mnemonic '{asm_form.mnemonic}' between descriptors '{previous_descriptor.key}' and '{descriptor.key}'")
        seen_mnemonics[asm_form.mnemonic] = asm_form_ordinal


def _validate_view_operand_forms_closed(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptor_ordinals: Sequence[int],
) -> None:
    selected_descriptor_ordinals = set(descriptor_ordinals)
    for storage_descriptor_ordinal in descriptor_ordinals:
        storage_descriptor = compiled.descriptors[storage_descriptor_ordinal]
        storage_row = compiled.descriptor_rows[storage_descriptor_ordinal]
        operand_form_start = storage_row["operand_form_start"]
        operand_form_count = storage_row["operand_form_count"]
        for form_ordinal in range(operand_form_count):
            operand_form = compiled.operand_forms[operand_form_start + form_ordinal]
            if operand_form.replacement_descriptor_ordinal in selected_descriptor_ordinals:
                continue
            replacement = compiled.descriptors[operand_form.replacement_descriptor_ordinal]
            raise ValueError(f"descriptor set view '{view_spec.key}' selects descriptor '{storage_descriptor.key}' without operand-form replacement descriptor '{replacement.key}'")


def _validate_view_descriptors_match_storage(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptor_ordinals: Sequence[int],
) -> tuple[Descriptor, ...]:
    projected_descriptors: list[Descriptor] = []
    for view_descriptor, storage_descriptor_ordinal in zip(view_spec.descriptors, descriptor_ordinals, strict=True):
        storage_descriptor = compiled.descriptors[storage_descriptor_ordinal]
        validation.validate_descriptor_operands(view_descriptor)
        validation.validate_descriptor_constraints(view_descriptor)
        projected_view_descriptor = compiler.derive_descriptor_projections(view_descriptor)
        projected_descriptors.append(projected_view_descriptor)
        if (
            replace(
                projected_view_descriptor,
                asm_forms=storage_descriptor.asm_forms,
                asm_surface=storage_descriptor.asm_surface,
                asm_surface_reason=storage_descriptor.asm_surface_reason,
                schedule_class=storage_descriptor.schedule_class,
            )
            == storage_descriptor
        ):
            continue
        raise ValueError(
            f"descriptor set view '{view_spec.key}' descriptor '{view_descriptor.key}' differs from storage descriptor '{storage_descriptor.key}' outside of asm forms, asm surface policy, or schedule class"
        )
    return tuple(projected_descriptors)


def _compile_view_schedule_tables(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptors: Sequence[Descriptor],
    reg_classes: Sequence[RegClass],
    resources: Sequence[Resource],
) -> _ViewScheduleTables:
    storage_schedule_classes = {schedule_class.name: schedule_class for schedule_class in compiled.schedule_classes}
    view_schedule_classes = {schedule_class.name: schedule_class for schedule_class in view_spec.schedule_classes}
    if any(descriptor.schedule_class is None for descriptor in descriptors):
        raise ValueError(f"descriptor set view '{view_spec.key}' has a descriptor without a schedule class")
    used_schedule_class_names = {descriptor.schedule_class for descriptor in descriptors if descriptor.schedule_class is not None}

    schedule_classes: list[ScheduleClass] = []
    for storage_schedule_class in compiled.schedule_classes:
        if storage_schedule_class.name not in used_schedule_class_names:
            schedule_classes.append(storage_schedule_class)
            continue
        view_schedule_class = view_schedule_classes.get(storage_schedule_class.name)
        if view_schedule_class is None:
            raise ValueError(f"descriptor set view '{view_spec.key}' references unknown schedule class '{storage_schedule_class.name}'")
        schedule_classes.append(view_schedule_class)

    missing_schedule_class_names = sorted(used_schedule_class_names - storage_schedule_classes.keys())
    if missing_schedule_class_names:
        raise ValueError(f"descriptor set view '{view_spec.key}' schedule classes are absent from storage set '{compiled.spec.key}': {', '.join(missing_schedule_class_names)}")

    resource_names = {resource.name for resource in resources}
    reg_class_names = {reg_class.name for reg_class in reg_classes}
    for schedule_class in schedule_classes:
        if schedule_class.name not in used_schedule_class_names:
            continue
        referenced_resource_names = {issue_use.resource for issue_use in schedule_class.issue_uses}
        referenced_resource_names.update(hazard.resource for hazard in schedule_class.hazards if hazard.resource is not None)
        for resource_name in referenced_resource_names:
            if resource_name not in resource_names:
                raise ValueError(f"descriptor set view '{view_spec.key}' schedule class '{schedule_class.name}' references resource '{resource_name}' absent from storage set '{compiled.spec.key}'")
        for pressure_delta in schedule_class.pressure_deltas:
            reg_class_name = pressure_delta.reg_class
            if reg_class_name not in reg_class_names:
                raise ValueError(
                    f"descriptor set view '{view_spec.key}' schedule class '{schedule_class.name}' references register class '{reg_class_name}' absent from storage set '{compiled.spec.key}'"
                )

    issue_uses: list[IssueUse] = []
    hazards: list[Hazard] = []
    pressure_deltas: list[PressureDelta] = []
    schedule_rows: list[dict[str, int]] = []
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
    return _ViewScheduleTables(
        schedule_classes=tuple(schedule_classes),
        issue_uses=tuple(issue_uses),
        hazards=tuple(hazards),
        pressure_deltas=tuple(pressure_deltas),
        schedule_rows=tuple(schedule_rows),
        uses_storage_schedule_classes=tuple(schedule_classes) == tuple(compiled.schedule_classes),
    )


def _view_instruction_classes(
    descriptors: Sequence[Descriptor],
    schedule_classes: Sequence[ScheduleClass],
    resources: Sequence[Resource],
) -> tuple[tuple[InstructionClass, ...], ...]:
    schedule_classes_by_name = {schedule_class.name: schedule_class for schedule_class in schedule_classes}
    resources_by_name = {resource.name: resource for resource in resources}
    return tuple(
        compiler.derive_instruction_classes(
            descriptor,
            schedule_classes_by_name[descriptor.schedule_class],
            resources_by_name,
        )
        for descriptor in descriptors
    )


def _view_descriptors_match_storage(
    compiled: CompiledDescriptorSet,
    descriptors: Sequence[Descriptor],
    descriptor_ordinals: Sequence[int],
) -> bool:
    return all(
        descriptor == compiled.descriptors[storage_descriptor_ordinal]
        for descriptor, storage_descriptor_ordinal in zip(
            descriptors,
            descriptor_ordinals,
            strict=True,
        )
    )


def _view_asm_forms_match_storage(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptor_ordinals: Sequence[int],
) -> bool:
    return all(
        view_descriptor.asm_forms == compiled.descriptors[storage_descriptor_ordinal].asm_forms
        for view_descriptor, storage_descriptor_ordinal in zip(view_spec.descriptors, descriptor_ordinals, strict=True)
    )


def _canonical_asm_form_ordinals(
    descriptor_count: int,
    asm_forms: Sequence[CompiledAsmForm],
) -> list[int | None]:
    canonical_asm_form_ordinals: list[int | None] = [None] * descriptor_count
    asm_form_counts_by_descriptor = [0] * descriptor_count
    for asm_form_ordinal, asm_form in enumerate(asm_forms):
        descriptor_ordinal = asm_form.descriptor_ordinal
        asm_form_counts_by_descriptor[descriptor_ordinal] += 1
        canonical_asm_form_ordinals[descriptor_ordinal] = asm_form_ordinal
    for descriptor_ordinal, form_count in enumerate(asm_form_counts_by_descriptor):
        if form_count != 1:
            canonical_asm_form_ordinals[descriptor_ordinal] = None
    return canonical_asm_form_ordinals


def _compile_view_asm_forms(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
) -> list[CompiledAsmForm]:
    asm_forms = compiler.compile_asm_forms_for_descriptors(
        compiled.string_pool,
        view_spec.descriptors,
        label_scope=f"view_{view_spec.key}",
    )
    compiler.append_asm_form_table_spans(
        asm_forms,
        compiled.asm_operand_indices,
        compiled.asm_immediates,
        compiled.native_asm_values,
    )
    return asm_forms


def descriptor_set_view_for_spec(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
) -> DescriptorSetView:
    if not view_spec.descriptors:
        raise ValueError(f"descriptor set view '{view_spec.key}' selects no descriptors")

    storage_ordinals_by_key = {descriptor.key: i for i, descriptor in enumerate(compiled.descriptors)}
    descriptor_ordinals = []
    for descriptor in view_spec.descriptors:
        descriptor_ordinal = storage_ordinals_by_key.get(descriptor.key)
        if descriptor_ordinal is None:
            raise ValueError(f"descriptor set view '{view_spec.key}' selects descriptor '{descriptor.key}' that is not in storage set '{compiled.spec.key}'")
        descriptor_ordinals.append(descriptor_ordinal)
    descriptor_ordinal_tuple = tuple(descriptor_ordinals)
    validation.validate_descriptor_asm_surface(
        view_spec,
        view_spec.descriptors,
        surface_name="descriptor set view",
    )
    descriptors = _validate_view_descriptors_match_storage(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    )
    reg_classes = _overlay_view_table(
        compiled.reg_classes,
        view_spec.reg_classes,
    )
    resources = _overlay_view_table(
        compiled.resources,
        view_spec.resources,
    )
    for reg_class in reg_classes:
        if reg_class.spill_class is not None and reg_class.spill_class not in compiled.reg_class_ids:
            raise ValueError(f"descriptor set view '{view_spec.key}' register class '{reg_class.name}' references spill class '{reg_class.spill_class}' absent from storage set '{compiled.spec.key}'")
    _validate_shared_view_table(
        compiled.register_parts,
        view_spec.register_parts,
        view_spec=view_spec,
        storage_spec=compiled.spec,
        table_name="register part",
    )
    _validate_shared_view_table(
        compiled.enum_domains,
        view_spec.enum_domains,
        view_spec=view_spec,
        storage_spec=compiled.spec,
        table_name="enum domain",
    )
    schedule_tables = _compile_view_schedule_tables(
        compiled,
        view_spec,
        descriptors,
        reg_classes,
        resources,
    )
    instruction_classes = _view_instruction_classes(
        descriptors,
        schedule_tables.schedule_classes,
        resources,
    )
    _validate_view_operand_forms_closed(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    )
    if (
        descriptor_ordinal_tuple == tuple(range(len(descriptor_ordinal_tuple)))
        and _view_descriptors_match_storage(
            compiled,
            descriptors,
            descriptor_ordinal_tuple,
        )
        and _view_asm_forms_match_storage(compiled, view_spec, descriptor_ordinal_tuple)
        and not _asm_forms_have_duplicate_mnemonics(compiled.asm_forms)
    ):
        descriptor_count = len(descriptor_ordinal_tuple)
        return DescriptorSetView(
            spec=view_spec,
            descriptors=descriptors,
            instruction_classes=instruction_classes,
            reg_classes=reg_classes,
            resources=resources,
            schedule_classes=schedule_tables.schedule_classes,
            issue_uses=schedule_tables.issue_uses,
            hazards=schedule_tables.hazards,
            pressure_deltas=schedule_tables.pressure_deltas,
            schedule_rows=schedule_tables.schedule_rows,
            descriptor_ordinals=descriptor_ordinal_tuple,
            descriptor_refs=_descriptor_refs_for_ordinals(
                compiled.descriptors,
                descriptor_ordinal_tuple,
            ),
            descriptor_rows=compiled.descriptor_rows[:descriptor_count],
            canonical_asm_form_ordinals=compiled.canonical_asm_form_ordinals[:descriptor_count],
            asm_forms=compiled.asm_forms,
            operand_forms=compiled.operand_forms,
            uses_storage_descriptor_tables=True,
            uses_storage_schedule_classes=(schedule_tables.uses_storage_schedule_classes),
            uses_storage_asm_form_tables=True,
            uses_storage_operand_form_tables=True,
        )

    storage_to_view_ordinals = {descriptor_ordinal: view_ordinal for view_ordinal, descriptor_ordinal in enumerate(descriptor_ordinal_tuple)}
    asm_forms = _compile_view_asm_forms(compiled, view_spec)
    _validate_view_asm_forms_unique(view_spec, asm_forms)

    descriptor_rows = []
    operand_forms: list[CompiledOperandForm] = []
    for storage_descriptor_ordinal in descriptor_ordinal_tuple:
        storage_row = compiled.descriptor_rows[storage_descriptor_ordinal]
        descriptor_row = dict(storage_row)
        descriptor_row["operand_form_start"] = len(operand_forms)
        operand_form_start = storage_row["operand_form_start"]
        operand_form_count = storage_row["operand_form_count"]
        for form_ordinal in range(operand_form_count):
            operand_form = compiled.operand_forms[operand_form_start + form_ordinal]
            operand_forms.append(
                _clone_operand_form_for_view(
                    operand_form,
                    storage_to_view_ordinals[operand_form.replacement_descriptor_ordinal],
                )
            )
        descriptor_rows.append(descriptor_row)

    return DescriptorSetView(
        spec=view_spec,
        descriptors=descriptors,
        instruction_classes=instruction_classes,
        reg_classes=reg_classes,
        resources=resources,
        schedule_classes=schedule_tables.schedule_classes,
        issue_uses=schedule_tables.issue_uses,
        hazards=schedule_tables.hazards,
        pressure_deltas=schedule_tables.pressure_deltas,
        schedule_rows=schedule_tables.schedule_rows,
        descriptor_ordinals=descriptor_ordinal_tuple,
        descriptor_refs=_descriptor_refs_for_ordinals(
            compiled.descriptors,
            descriptor_ordinal_tuple,
        ),
        descriptor_rows=descriptor_rows,
        canonical_asm_form_ordinals=_canonical_asm_form_ordinals(
            len(descriptor_ordinal_tuple),
            asm_forms,
        ),
        asm_forms=asm_forms,
        operand_forms=operand_forms,
        uses_storage_descriptor_tables=False,
        uses_storage_schedule_classes=(schedule_tables.uses_storage_schedule_classes),
        uses_storage_asm_form_tables=False,
        uses_storage_operand_form_tables=False,
    )
