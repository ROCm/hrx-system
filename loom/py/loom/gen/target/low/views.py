# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor-set view construction for compiled low descriptor sets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from loom.gen.target.low import compiler, validation
from loom.gen.target.low.compiled import (
    CompiledAsmForm,
    CompiledDescriptorSet,
    CompiledNamedScheduleModel,
    CompiledOperandForm,
    CompiledScheduleModel,
    DescriptorSetView,
    append_interned_sequence,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    InstructionClass,
    RegClass,
    ScheduleClassFlag,
    ScheduleModel,
    descriptor_stable_id,
)


def _named_schedule_model_view_spec(
    view_spec: DescriptorSet,
    model: ScheduleModel,
) -> DescriptorSet:
    """Projects a complete named model over one structural descriptor view."""

    model_resources = {resource.name: resource for resource in model.resources}
    missing_resources = sorted(resource.name for resource in view_spec.resources if resource.name not in model_resources)
    if missing_resources:
        raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model.key}' omits structural resources: {', '.join(missing_resources)}")
    for structural_resource in view_spec.resources:
        model_resource = model_resources[structural_resource.name]
        if model_resource.kind != structural_resource.kind or model_resource.flags != structural_resource.flags:
            raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model.key}' changes semantic resource properties for '{model_resource.name}'")

    model_schedule_classes = {schedule_class.name: schedule_class for schedule_class in model.schedule_classes}
    missing_schedule_classes = sorted(schedule_class.name for schedule_class in view_spec.schedule_classes if schedule_class.name not in model_schedule_classes)
    if missing_schedule_classes:
        raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model.key}' omits structural schedule classes: {', '.join(missing_schedule_classes)}")
    for structural_class in view_spec.schedule_classes:
        model_class = model_schedule_classes[structural_class.name]
        if model_class.flags != structural_class.flags or model_class.instruction_classes != structural_class.instruction_classes:
            raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model.key}' changes semantic class properties for '{model_class.name}'")

    model_timing_event_names = {event.name for event in model.timing_events}
    missing_timing_events = sorted(event.name for event in view_spec.timing_events if event.name not in model_timing_event_names)
    if missing_timing_events:
        raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model.key}' omits structural timing events: {', '.join(missing_timing_events)}")

    return replace(
        view_spec,
        resources=model.resources,
        schedule_classes=model.schedule_classes,
        timing_events=model.timing_events,
        event_separations=model.event_separations,
        schedule_models=(),
    )


def _descriptor_refs_for_ordinals(
    descriptors: Sequence[Descriptor],
    descriptor_ordinals: Sequence[int],
    uses_storage_ordinals: bool,
) -> list[int]:
    ordered = sorted(enumerate(descriptor_ordinals), key=lambda item: descriptor_stable_id(descriptors[item[1]].key))
    return [storage_ordinal if uses_storage_ordinals else view_ordinal for view_ordinal, storage_ordinal in ordered]


def _descriptor_membership_words(
    descriptor_ordinal_count: int,
    descriptor_ordinals: Sequence[int],
) -> tuple[int, ...]:
    if tuple(sorted(descriptor_ordinals)) == tuple(range(descriptor_ordinal_count)):
        return ()
    words = [0] * ((descriptor_ordinal_count + 63) // 64)
    for descriptor_ordinal in descriptor_ordinals:
        words[descriptor_ordinal // 64] |= 1 << (descriptor_ordinal % 64)
    return tuple(words)


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


def _compile_view_schedule_model(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptors: Sequence[Descriptor],
    model_spec: ScheduleModel | None = None,
) -> CompiledScheduleModel:
    """Compiles a view's model in the family's canonical ID namespaces."""

    schedule_view_spec = _named_schedule_model_view_spec(view_spec, model_spec) if model_spec is not None else view_spec
    if model_spec is not None:
        model_resource_names = tuple(resource.name for resource in schedule_view_spec.resources)
        canonical_resource_names = tuple(resource.name for resource in compiled.resources)
        if model_resource_names != canonical_resource_names:
            raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model_spec.key}' must define the canonical family resource namespace")
        model_schedule_names = tuple(schedule_class.name for schedule_class in schedule_view_spec.schedule_classes)
        canonical_schedule_names = tuple(schedule_class.name for schedule_class in compiled.schedule_classes)
        if model_schedule_names != canonical_schedule_names:
            raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model_spec.key}' must define the canonical family schedule-class namespace")
        model_timing_event_names = tuple(event.name for event in schedule_view_spec.timing_events)
        canonical_timing_event_names = tuple(event.name for event in compiled.timing_events)
        if model_timing_event_names != canonical_timing_event_names:
            raise ValueError(f"descriptor set view '{view_spec.key}' schedule model '{model_spec.key}' must define the canonical family timing-event namespace")
        schedule_view_spec = replace(
            schedule_view_spec,
            reg_classes=compiled.reg_classes,
        )
    disjoint_schedule_classes, minimum_issue_cycles_by_class = validation.validate_schedule_model(schedule_view_spec)
    schedule_inputs = {schedule_class.name: schedule_class for schedule_class in schedule_view_spec.schedule_classes}
    descriptor_schedule_names = {descriptor.schedule_class for descriptor in descriptors}
    unknown_schedule_names = sorted(descriptor_schedule_names - schedule_inputs.keys())
    if unknown_schedule_names:
        raise ValueError(f"descriptor set view '{view_spec.key}' references unknown schedule classes: {', '.join(unknown_schedule_names)}")
    missing_storage_schedule_names = sorted(descriptor_schedule_names - compiled.schedule_class_ids.keys())
    if missing_storage_schedule_names:
        raise ValueError(f"descriptor set view '{view_spec.key}' references schedule classes absent from storage set '{compiled.spec.key}': {', '.join(missing_storage_schedule_names)}")

    active_schedule_names = set(schedule_inputs) if model_spec is not None else descriptor_schedule_names
    used_resource_names: set[str] = set()
    for schedule_name in active_schedule_names:
        schedule_class = schedule_inputs[schedule_name]
        used_resource_names.update(issue_use.resource for issue_use in schedule_class.issue_uses)
        used_resource_names.update(hazard.resource for hazard in schedule_class.hazards if hazard.resource is not None)
        for pressure_delta in schedule_class.pressure_deltas:
            if pressure_delta.reg_class not in compiled.reg_class_ids:
                raise ValueError(
                    f"descriptor set view '{view_spec.key}' schedule class "
                    f"'{schedule_name}' references pressure register class "
                    f"'{pressure_delta.reg_class}' absent from storage set "
                    f"'{compiled.spec.key}'"
                )

    missing_storage_resource_names = sorted(used_resource_names - compiled.resource_ids.keys())
    if missing_storage_resource_names:
        raise ValueError(f"descriptor set view '{view_spec.key}' references resources absent from storage set '{compiled.spec.key}': {', '.join(missing_storage_resource_names)}")
    view_resources = {resource.name: resource for resource in schedule_view_spec.resources}
    resources = tuple(view_resources[resource.name] if model_spec is not None or resource.name in used_resource_names else resource for resource in compiled.resources)

    used_timing_event_names = (
        {event.name for event in schedule_view_spec.timing_events}
        if model_spec is not None
        else {event_name for descriptor in descriptors for operand in descriptor.operands for event_name in (operand.read_event, operand.write_event) if event_name is not None}
    )
    if model_spec is None:
        used_timing_event_names.update(
            event_name for descriptor in descriptors for effect in descriptor.effects for event_name in (effect.producer_event, effect.consumer_event) if event_name is not None
        )
    missing_timing_event_names = sorted(used_timing_event_names - compiled.timing_event_ids.keys())
    if missing_timing_event_names:
        raise ValueError(f"descriptor set view '{view_spec.key}' references timing events absent from storage set '{compiled.spec.key}': {', '.join(missing_timing_event_names)}")
    event_separations = tuple(
        sorted(
            (separation for separation in schedule_view_spec.event_separations if separation.producer_event in used_timing_event_names and separation.consumer_event in used_timing_event_names),
            key=lambda separation: (
                compiled.timing_event_ids[separation.producer_event],
                compiled.timing_event_ids[separation.consumer_event],
            ),
        )
    )

    # Descriptors retain family-canonical schedule-class IDs. Complete each
    # view's table in that same namespace so family emission can pool equal
    # rows and assign direct physical row IDs only where values differ. Named
    # processor models describe the complete namespace and therefore remain
    # shareable across structural descriptor views.
    schedule_classes = tuple(
        (
            replace(
                schedule_inputs[schedule_class.name],
                flags=(
                    *schedule_inputs[schedule_class.name].flags,
                    ScheduleClassFlag.DISJOINT_ISSUE_USES,
                ),
            )
            if schedule_class.name in disjoint_schedule_classes
            else schedule_inputs[schedule_class.name]
        )
        if schedule_class.name in active_schedule_names
        else schedule_class
        for schedule_class in compiled.schedule_classes
    )

    issue_uses = []
    issue_use_group_starts = {}
    hazards = []
    hazard_group_starts = {}
    pressure_deltas = []
    pressure_delta_group_starts = {}
    schedule_rows = []
    for schedule_class_id, schedule_class in enumerate(schedule_classes):
        issue_use_start, _ = append_interned_sequence(
            schedule_class.issue_uses,
            issue_uses,
            issue_use_group_starts,
        )
        hazard_start, _ = append_interned_sequence(
            schedule_class.hazards,
            hazards,
            hazard_group_starts,
        )
        pressure_delta_start, _ = append_interned_sequence(
            schedule_class.pressure_deltas,
            pressure_deltas,
            pressure_delta_group_starts,
        )
        schedule_rows.append(
            {
                "issue_use_start": issue_use_start,
                "issue_use_count": len(schedule_class.issue_uses),
                "hazard_start": hazard_start,
                "hazard_count": len(schedule_class.hazards),
                "minimum_issue_cycles": (
                    minimum_issue_cycles_by_class[schedule_class.name] if schedule_class.name in active_schedule_names else compiled.schedule_rows[schedule_class_id]["minimum_issue_cycles"]
                ),
                "pressure_delta_start": pressure_delta_start,
                "pressure_delta_count": len(schedule_class.pressure_deltas),
            }
        )

    for table_name, rows in (
        ("schedule class", schedule_classes),
        ("issue use", issue_uses),
        ("hazard", hazards),
        ("pressure delta", pressure_deltas),
    ):
        validation.validate_u16_table_count(len(rows), f"descriptor set view '{view_spec.key}' {table_name}")

    resource_calendar_lookback_cycles = max(
        (max(0, -separation.minimum_issue_separation_cycles) for separation in event_separations),
        default=0,
    )
    resource_calendars, resource_calendar_slot_count = compiler.compile_resource_calendars(
        resources,
        schedule_classes,
        resource_calendar_lookback_cycles,
    )
    return CompiledScheduleModel(
        resources=resources,
        resource_calendars=tuple(resource_calendars),
        resource_calendar_slot_count=resource_calendar_slot_count,
        resource_calendar_lookback_cycles=resource_calendar_lookback_cycles,
        schedule_classes=schedule_classes,
        # Shared operand and effect rows carry storage timing-event IDs. Every
        # view retains that namespace while owning its separation values.
        timing_events=tuple(compiled.timing_events),
        event_separations=event_separations,
        resource_ids=compiled.resource_ids,
        schedule_class_ids=compiled.schedule_class_ids,
        timing_event_ids=compiled.timing_event_ids,
        issue_uses=tuple(issue_uses),
        hazards=tuple(hazards),
        pressure_deltas=tuple(pressure_deltas),
        schedule_rows=tuple(schedule_rows),
    )


def _compile_named_schedule_models(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    compiled_models_by_key: dict[str, tuple[ScheduleModel, CompiledScheduleModel]],
) -> tuple[CompiledNamedScheduleModel, ...]:
    """Validates one view and reuses family-wide named model compilation."""

    compiled_models = []
    for model_spec in view_spec.schedule_models:
        # Named models use the complete family namespace, but each public view
        # still establishes that their semantic resources and classes agree.
        _named_schedule_model_view_spec(view_spec, model_spec)
        cached_model = compiled_models_by_key.get(model_spec.key)
        if cached_model is None:
            compiled_model = _compile_view_schedule_model(
                compiled,
                compiled.spec,
                compiled.descriptors,
                model_spec,
            )
            compiled_models_by_key[model_spec.key] = (
                model_spec,
                compiled_model,
            )
        else:
            cached_spec, compiled_model = cached_model
            if cached_spec is not model_spec and cached_spec != model_spec:
                raise ValueError(f"descriptor set view '{view_spec.key}' defines schedule model '{model_spec.key}' differently from another view in the same family")
        compiled_models.append(
            CompiledNamedScheduleModel(
                key=model_spec.key,
                schedule_model=compiled_model,
            )
        )
    return tuple(compiled_models)


def _schedule_alternative_rows_for_view(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptor_ordinals: Sequence[int],
    uses_storage_ordinals: bool,
) -> list[tuple[int, int]]:
    storage_to_view_ordinals = {storage_ordinal: view_ordinal for view_ordinal, storage_ordinal in enumerate(descriptor_ordinals)}
    rows: list[tuple[int, int]] = []
    for storage_source, storage_alternative in compiled.schedule_alternative_rows:
        view_source = storage_to_view_ordinals.get(storage_source)
        if view_source is None:
            continue
        view_alternative = storage_to_view_ordinals.get(storage_alternative)
        if view_alternative is None:
            source = compiled.descriptors[storage_source]
            alternative = compiled.descriptors[storage_alternative]
            raise ValueError(f"descriptor set view '{view_spec.key}' selects descriptor '{source.key}' without schedule alternative descriptor '{alternative.key}'")
        rows.append(
            (
                storage_source if uses_storage_ordinals else view_source,
                storage_alternative if uses_storage_ordinals else view_alternative,
            )
        )
    return rows


def _validate_view_descriptors_match_storage(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptor_ordinals: Sequence[int],
) -> tuple[Descriptor, ...]:
    projected_descriptors: list[Descriptor] = []
    for view_descriptor, storage_descriptor_ordinal in zip(view_spec.descriptors, descriptor_ordinals, strict=True):
        source_descriptor = compiled.source_descriptors[storage_descriptor_ordinal]
        storage_descriptor = compiled.descriptors[storage_descriptor_ordinal]
        if view_descriptor == source_descriptor:
            projected_descriptors.append(storage_descriptor)
            continue
        if (
            replace(
                view_descriptor,
                asm_forms=source_descriptor.asm_forms,
                asm_surface=source_descriptor.asm_surface,
                asm_surface_reason=source_descriptor.asm_surface_reason,
                schedule_class=source_descriptor.schedule_class,
            )
            != source_descriptor
        ):
            raise ValueError(
                f"descriptor set view '{view_spec.key}' descriptor '{view_descriptor.key}' differs from storage descriptor '{storage_descriptor.key}' outside of asm forms, asm surface policy, or schedule class"
            )
        projected_descriptors.append(
            replace(
                storage_descriptor,
                asm_forms=view_descriptor.asm_forms,
                asm_surface=view_descriptor.asm_surface,
                asm_surface_reason=view_descriptor.asm_surface_reason,
                schedule_class=view_descriptor.schedule_class,
            )
        )
    return tuple(projected_descriptors)


def _view_instruction_classes(
    view_spec: DescriptorSet,
    descriptors: Sequence[Descriptor],
    schedule_model: CompiledScheduleModel,
) -> tuple[tuple[InstructionClass, ...], ...]:
    schedule_classes = {schedule_class.name: schedule_class for schedule_class in schedule_model.schedule_classes}
    resources = {resource.name: resource for resource in schedule_model.resources}
    return tuple(
        compiler.derive_instruction_classes(
            descriptor,
            schedule_classes[descriptor.schedule_class],
            resources,
        )
        for descriptor in descriptors
    )


def _view_descriptor_views_match_storage(
    compiled: CompiledDescriptorSet,
    descriptors: Sequence[Descriptor],
    instruction_classes: Sequence[tuple[InstructionClass, ...]],
    canonical_asm_form_ordinals: Sequence[int | None],
    descriptor_ordinals: Sequence[int],
) -> bool:
    return all(
        descriptor.schedule_class == compiled.descriptors[storage_ordinal].schedule_class
        and instruction_class == compiled.instruction_classes[storage_ordinal]
        and canonical_asm_form_ordinal == compiled.canonical_asm_form_ordinals[storage_ordinal]
        for descriptor, instruction_class, canonical_asm_form_ordinal, storage_ordinal in zip(
            descriptors,
            instruction_classes,
            canonical_asm_form_ordinals,
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
        compiled.enum_domains,
        label_scope=f"view_{view_spec.key}",
    )
    validation.validate_u16_table_count(len(asm_forms), f"descriptor set view '{view_spec.key}' asm form")
    compiled.asm_table_storage.append_forms(asm_forms)
    return asm_forms


def _view_register_classes(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    schedule_model: CompiledScheduleModel,
) -> tuple[RegClass | None, ...]:
    classes_by_name = {reg_class.name: reg_class for reg_class in view_spec.reg_classes}
    if len(classes_by_name) != len(view_spec.reg_classes):
        raise ValueError(f"descriptor set view '{view_spec.key}' has duplicate register classes")
    validation.validate_register_classes(
        view_spec.key,
        view_spec.reg_classes,
        compiled.physical_registers,
        tuple(view for view in compiled.spec.physical_register_views if view.reg_class in classes_by_name),
        alias_set_count=max((reg_class.alias_set_id for reg_class in compiled.reg_classes), default=0),
    )
    for reg_class in view_spec.reg_classes:
        storage_id = compiled.reg_class_ids.get(reg_class.name)
        if storage_id is None:
            raise ValueError(f"descriptor set view '{view_spec.key}' register class '{reg_class.name}' is missing from storage")
        storage_class = compiled.reg_classes[storage_id]
        if (
            replace(
                reg_class,
                allocatable_count=storage_class.allocatable_count,
                fixed_location_base=storage_class.fixed_location_base,
                fixed_location_count=storage_class.fixed_location_count,
            )
            != storage_class
        ):
            raise ValueError(f"descriptor set view '{view_spec.key}' register class '{reg_class.name}' differs from storage outside allocation capacity or fixed locations")
        if reg_class.spill_class is not None and reg_class.spill_class not in classes_by_name:
            raise ValueError(f"descriptor set view '{view_spec.key}' register class '{reg_class.name}' references absent spill class '{reg_class.spill_class}'")

    # Shared operand and schedule tables may contain rows for other views.
    # Only the selected descriptors can reference the classes exposed here.
    schedules_by_name = {schedule.name: schedule for schedule in schedule_model.schedule_classes}
    parts_by_name = {part.name: part for part in compiled.register_parts}
    for descriptor in view_spec.descriptors:
        referenced_classes = {alternative.reg_class for operand in descriptor.operands for alternative in operand.reg_alts if alternative.reg_class is not None}
        referenced_classes.update(parts_by_name[alternative.register_part].reg_class for operand in descriptor.operands for alternative in operand.reg_alts if alternative.register_part is not None)
        referenced_classes.update(delta.reg_class for delta in schedules_by_name[descriptor.schedule_class].pressure_deltas)
        missing_classes = referenced_classes - classes_by_name.keys()
        if missing_classes:
            raise ValueError(f"descriptor set view '{view_spec.key}' descriptor '{descriptor.key}' references absent register classes: {', '.join(sorted(missing_classes))}")
    validation.validate_physical_descriptor_set(view_spec)
    return tuple(classes_by_name.get(reg_class.name) for reg_class in compiled.reg_classes)


def _descriptor_set_view_for_spec(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
    compiled_models_by_key: dict[str, tuple[ScheduleModel, CompiledScheduleModel]],
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
    validation.validate_schedule_alternatives(view_spec)
    descriptors = _validate_view_descriptors_match_storage(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    )
    schedule_model = _compile_view_schedule_model(
        compiled,
        view_spec,
        descriptors,
    )
    schedule_models = _compile_named_schedule_models(
        compiled,
        view_spec,
        compiled_models_by_key,
    )
    instruction_classes = _view_instruction_classes(
        view_spec,
        descriptors,
        schedule_model,
    )
    compiler.validate_schedule_alternative_instruction_classes(
        descriptors,
        instruction_classes,
    )
    _validate_view_operand_forms_closed(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    )
    is_storage_prefix = descriptor_ordinal_tuple == tuple(range(len(descriptor_ordinal_tuple)))
    storage_asm_forms_match = _view_asm_forms_match_storage(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    ) and not _asm_forms_have_duplicate_mnemonics(compiled.asm_forms)
    storage_canonical_asm_form_ordinals = [compiled.canonical_asm_form_ordinals[storage_ordinal] for storage_ordinal in descriptor_ordinal_tuple]
    storage_descriptor_views_match = storage_asm_forms_match and _view_descriptor_views_match_storage(
        compiled,
        descriptors,
        instruction_classes,
        storage_canonical_asm_form_ordinals,
        descriptor_ordinal_tuple,
    )
    uses_storage_ordinals = is_storage_prefix or storage_descriptor_views_match
    uses_storage_descriptor_tables = uses_storage_ordinals
    uses_storage_asm_form_tables = uses_storage_ordinals and storage_asm_forms_match
    uses_storage_operand_form_tables = uses_storage_ordinals
    if uses_storage_asm_form_tables:
        asm_forms = compiled.asm_forms
        canonical_asm_form_ordinals = storage_canonical_asm_form_ordinals
    else:
        asm_forms = _compile_view_asm_forms(compiled, view_spec)
        _validate_view_asm_forms_unique(view_spec, asm_forms)
        canonical_asm_form_ordinals = _canonical_asm_form_ordinals(
            len(descriptor_ordinal_tuple),
            asm_forms,
        )

    uses_storage_descriptor_view_tables = uses_storage_ordinals and _view_descriptor_views_match_storage(
        compiled,
        descriptors,
        instruction_classes,
        canonical_asm_form_ordinals,
        descriptor_ordinal_tuple,
    )

    descriptor_ordinal_count = max(descriptor_ordinal_tuple) + 1 if uses_storage_ordinals and descriptor_ordinal_tuple else len(descriptor_ordinal_tuple)
    descriptor_membership_words = (
        _descriptor_membership_words(
            descriptor_ordinal_count,
            descriptor_ordinal_tuple,
        )
        if uses_storage_ordinals
        else ()
    )
    schedule_alternative_rows = _schedule_alternative_rows_for_view(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
        uses_storage_ordinals,
    )

    descriptor_rows = [dict(compiled.descriptor_rows[storage_descriptor_ordinal]) for storage_descriptor_ordinal in descriptor_ordinal_tuple]
    if uses_storage_operand_form_tables:
        operand_forms = compiled.operand_forms
    else:
        storage_to_view_ordinals = {descriptor_ordinal: view_ordinal for view_ordinal, descriptor_ordinal in enumerate(descriptor_ordinal_tuple)}
        operand_forms = []
        for descriptor_row, storage_descriptor_ordinal in zip(
            descriptor_rows,
            descriptor_ordinal_tuple,
            strict=True,
        ):
            storage_row = compiled.descriptor_rows[storage_descriptor_ordinal]
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

    return DescriptorSetView(
        spec=view_spec,
        reg_classes=_view_register_classes(
            compiled,
            view_spec,
            schedule_model,
        ),
        descriptors=descriptors,
        instruction_classes=instruction_classes,
        schedule_model=schedule_model,
        schedule_class_row_ids=(),
        schedule_models=schedule_models,
        storage_descriptor_ordinals=descriptor_ordinal_tuple,
        descriptor_ordinal_count=descriptor_ordinal_count,
        descriptor_membership_words=descriptor_membership_words,
        descriptor_refs=_descriptor_refs_for_ordinals(
            compiled.descriptors,
            descriptor_ordinal_tuple,
            uses_storage_ordinals,
        ),
        schedule_alternative_rows=schedule_alternative_rows,
        descriptor_rows=descriptor_rows,
        canonical_asm_form_ordinals=canonical_asm_form_ordinals,
        asm_forms=asm_forms,
        operand_forms=operand_forms,
        uses_storage_descriptor_tables=uses_storage_descriptor_tables,
        uses_storage_descriptor_view_tables=uses_storage_descriptor_view_tables,
        uses_storage_descriptor_ref_tables=uses_storage_descriptor_tables,
        uses_storage_asm_form_tables=uses_storage_asm_form_tables,
        uses_storage_operand_form_tables=uses_storage_operand_form_tables,
        uses_storage_schedule_alternative_tables=(schedule_alternative_rows == compiled.schedule_alternative_rows[: len(schedule_alternative_rows)]),
    )


def descriptor_set_view_for_spec(
    compiled: CompiledDescriptorSet,
    view_spec: DescriptorSet,
) -> DescriptorSetView:
    """Compiles one structural descriptor-set view."""

    return _descriptor_set_view_for_spec(compiled, view_spec, {})


def descriptor_set_views_for_specs(
    compiled: CompiledDescriptorSet,
    view_specs: Sequence[DescriptorSet],
) -> tuple[DescriptorSetView, ...]:
    """Compiles views while sharing each complete named machine model."""

    compiled_models_by_key: dict[str, tuple[ScheduleModel, CompiledScheduleModel]] = {}
    return tuple(
        _descriptor_set_view_for_spec(
            compiled,
            view_spec,
            compiled_models_by_key,
        )
        for view_spec in view_specs
    )
