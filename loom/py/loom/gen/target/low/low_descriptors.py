# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: target-low descriptor inputs -> dense C tables.

The generator consumes a rich, explicit Python schema and emits compact
runtime tables under loom/src/loom. The C build only sees dense .rodata
arrays; Python owns source readability, validation, and allowlist closure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from loom.gen.support.files import write_text_file
from loom.gen.support.string_pool import CStringBank
from loom.gen.target.low import c_emit, compiler, validation, views
from loom.gen.target.low.compiled import (
    CompiledDescriptorSet,
    CompiledNamedScheduleModel,
    CompiledScheduleClassTable,
    CompiledScheduleModel,
    DescriptorAllowlist,
    DescriptorSetView,
    GeneratedDescriptorSet,
    GeneratedDescriptorSetFamily,
)
from loom.target.low_descriptors import DescriptorSet, ScheduleClass


def _descriptor_set_string_values(spec: DescriptorSet) -> set[str]:
    """Returns the conservative runtime string vocabulary of one set spec."""

    values = {"", spec.key}
    values.update(value for value in (spec.target_key, spec.feature_key) if value is not None)
    values.update(reg_class.name for reg_class in spec.reg_classes)
    values.update(register.name for register in spec.physical_registers)
    values.update(resource.name for resource in spec.register_packing_resources)
    values.update(part.name for part in spec.register_parts)
    values.update(resource.name for resource in spec.resources)
    values.update(schedule_class.name for schedule_class in spec.schedule_classes)
    values.update(event.name for event in spec.timing_events)
    for model in spec.schedule_models:
        values.update(resource.name for resource in model.resources)
        values.update(schedule_class.name for schedule_class in model.schedule_classes)
        values.update(event.name for event in model.timing_events)
    for domain in spec.enum_domains:
        values.add(domain.name)
        values.update(enum_value.token for enum_value in domain.values)
    for descriptor in spec.descriptors:
        values.add(descriptor.key)
        values.update(value for value in (descriptor.mnemonic, descriptor.semantic_tag) if value is not None)
        values.update(operand.field_name for operand in descriptor.operands)
        values.update(immediate.field_name for immediate in descriptor.immediates)
        for lease in descriptor.storage_leases:
            values.update(
                (
                    lease.release_class_name,
                    lease.release_action_name,
                    lease.release_reason_name,
                )
            )
        for asm_form in descriptor.asm_forms:
            values.add(validation.asm_form_mnemonic(descriptor, asm_form))
            if asm_form.native_assembly_mnemonic is not None:
                values.add(asm_form.native_assembly_mnemonic)
            values.update(immediate.name for immediate in asm_form.immediates if immediate.name is not None)
            values.update(native_value.literal for native_value in asm_form.native_assembly_values if native_value.literal is not None)
    return values


def descriptor_set_family_string_values(
    storage_spec: DescriptorSet,
    view_specs: Sequence[DescriptorSet],
) -> frozenset[str]:
    """Returns a generation-time superset of one family's retained strings.

    This query intentionally reads source specifications without compiling
    descriptor tables. External string-bank layout remains strict: generation
    fails if compilation ever retains a value absent from this vocabulary.
    """

    values = _descriptor_set_string_values(storage_spec)
    for view_spec in view_specs:
        values.update(_descriptor_set_string_values(view_spec))
    return frozenset(values)


def _pool_family_schedule_classes(
    compiled: CompiledDescriptorSet,
    descriptor_set_views: Sequence[DescriptorSetView],
) -> tuple[CompiledDescriptorSet, tuple[DescriptorSetView, ...]]:
    """Pools canonical rows and view variants without duplicating full tables."""

    canonical_count = len(compiled.schedule_classes)
    pooled_classes = []
    pooled_rows = []
    pooled_row_ids = {}
    pooled_models: dict[int, tuple[int, ...]] = {}

    def physical_row_key(
        schedule_class: ScheduleClass,
        row: dict[str, int],
    ) -> tuple[object, ...]:
        # Canonical names and source-only instruction classifications remain in
        # the descriptor-set namespace. Physical rows contain only these
        # scheduling fields and model-local subordinate-table spans.
        return (
            schedule_class.latency_cycles,
            schedule_class.schedule_distance_cycles,
            schedule_class.minimum_issue_separation_cycles,
            schedule_class.flags,
            schedule_class.latency_kind,
            schedule_class.model_quality,
            tuple(sorted(row.items())),
        )

    def pool_model(
        model: CompiledScheduleModel,
        description: str,
    ) -> tuple[int, ...]:
        cached_row_ids = pooled_models.get(id(model))
        if cached_row_ids is not None:
            return cached_row_ids
        if len(model.schedule_classes) != canonical_count:
            raise ValueError(f"{description} has {len(model.schedule_classes)} schedule classes in a {canonical_count}-class family namespace")
        schedule_class_row_ids = []
        for schedule_class, row in zip(
            model.schedule_classes,
            model.schedule_rows,
            strict=True,
        ):
            key = physical_row_key(schedule_class, row)
            schedule_class_row_id = pooled_row_ids.get(key)
            if schedule_class_row_id is None:
                schedule_class_row_id = len(pooled_classes)
                pooled_row_ids[key] = schedule_class_row_id
                pooled_classes.append(schedule_class)
                pooled_rows.append(dict(row))
            schedule_class_row_ids.append(schedule_class_row_id)
        row_ids = tuple(schedule_class_row_ids)
        row_ids = () if row_ids == tuple(range(canonical_count)) else row_ids
        pooled_models[id(model)] = row_ids
        return row_ids

    pooled_views = []
    for view in descriptor_set_views:
        row_ids = pool_model(
            view.schedule_model,
            f"descriptor set view '{view.spec.key}'",
        )
        named_models = tuple(
            CompiledNamedScheduleModel(
                key=entry.key,
                schedule_model=entry.schedule_model,
                schedule_class_row_ids=pool_model(
                    entry.schedule_model,
                    f"descriptor set view '{view.spec.key}' schedule model '{entry.key}'",
                ),
            )
            for entry in view.schedule_models
        )
        pooled_views.append(
            replace(
                view,
                schedule_class_row_ids=row_ids,
                schedule_models=named_models,
            )
        )

    validation.validate_u16_table_count(
        len(pooled_classes),
        f"descriptor set family '{compiled.spec.key}' schedule class",
    )
    return (
        replace(
            compiled,
            schedule_class_table=CompiledScheduleClassTable(
                schedule_classes=tuple(pooled_classes),
                schedule_rows=tuple(pooled_rows),
                canonical_count=canonical_count,
            ),
        ),
        tuple(pooled_views),
    )


def _schedule_model_vocabulary(
    specs: Sequence[DescriptorSet],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    schedule_class_names: set[str] = set()
    resource_names: set[str] = set()
    timing_event_names: set[str] = set()
    for spec in specs:
        for model in spec.schedule_models:
            schedule_class_names.update(schedule_class.name for schedule_class in model.schedule_classes)
            resource_names.update(resource.name for resource in model.resources)
            timing_event_names.update(event.name for event in model.timing_events)
    return (
        tuple(sorted(schedule_class_names)),
        tuple(sorted(resource_names)),
        tuple(sorted(timing_event_names)),
    )


def generate_descriptor_set(
    spec: DescriptorSet,
    allowlist: DescriptorAllowlist | None = None,
    *,
    primary_strings: CStringBank | None = None,
    shared_strings: CStringBank | None = None,
    required_schedule_class_names: Sequence[str] = (),
) -> GeneratedDescriptorSet:
    (
        model_schedule_class_names,
        required_resource_names,
        required_timing_event_names,
    ) = _schedule_model_vocabulary((spec,))
    required_schedule_class_names = tuple(sorted(set(required_schedule_class_names) | set(model_schedule_class_names)))
    compiled = compiler.compile_descriptor_set(
        spec,
        allowlist,
        required_schedule_class_names=required_schedule_class_names,
        required_resource_names=required_resource_names,
        required_timing_event_names=required_timing_event_names,
    )
    if spec.schedule_models:
        view_spec = replace(spec, descriptors=tuple(compiled.descriptors))
        compiled, (view,) = _pool_family_schedule_classes(
            compiled,
            (views.descriptor_set_view_for_spec(compiled, view_spec),),
        )
        return GeneratedDescriptorSet(
            header=c_emit.emit_header(compiled),
            source=c_emit.emit_source_for_views(
                compiled,
                views=(view,),
                primary_strings=primary_strings,
                shared_strings=shared_strings,
            ),
        )
    return GeneratedDescriptorSet(
        header=c_emit.emit_header(compiled),
        source=c_emit.emit_source(
            compiled,
            shared_strings,
            primary_strings=primary_strings,
        ),
    )


def compile_descriptor_set_family(
    storage_spec: DescriptorSet,
    view_specs: Sequence[DescriptorSet],
) -> tuple[CompiledDescriptorSet, tuple[DescriptorSetView, ...]]:
    """Compiles shared storage and its descriptor-set views.

    Each view selects descriptors from |storage_spec| by stable key. Supporting
    tables are shared as a storage superset. Register classes retain storage
    indices but expose each view's authored capacities and absent slots.
    The scheduling model stored inline in each descriptor-set view is compiled
    per view, while each additional processor model is compiled once in the
    complete canonical family namespace. Every emitted model has an explicit
    catalog key. Identical class rows share one family table, while views with
    distinct rows map canonical descriptor IDs directly into that pool.
    Descriptor, operand-form, and asm-form tables are reused only when the
    selected view surface matches the storage rows. A view may provide its own
    asm forms for the same descriptor keys; those forms are compiled and
    validated against the shared storage vocabulary during generation.
    """
    (
        model_schedule_class_names,
        model_resource_names,
        required_timing_event_names,
    ) = _schedule_model_vocabulary(view_specs)
    required_schedule_class_name_set = set(model_schedule_class_names)
    for view_spec in view_specs:
        required_schedule_class_name_set.update(descriptor.schedule_class for descriptor in view_spec.descriptors if descriptor.schedule_class is not None)
    required_resource_name_set = set(model_resource_names)
    for view_spec in view_specs:
        selected_schedule_class_names = {descriptor.schedule_class for descriptor in view_spec.descriptors}
        for schedule_class in view_spec.schedule_classes:
            if schedule_class.name not in selected_schedule_class_names:
                continue
            required_resource_name_set.update(issue_use.resource for issue_use in schedule_class.issue_uses)
            required_resource_name_set.update(hazard.resource for hazard in schedule_class.hazards if hazard.resource is not None)
    required_schedule_class_names = tuple(sorted(required_schedule_class_name_set))
    required_resource_names = tuple(sorted(required_resource_name_set))
    compiled = compiler.compile_descriptor_set(
        storage_spec,
        allowlist=None,
        allow_ambiguous_asm_mnemonics=True,
        required_schedule_class_names=required_schedule_class_names,
        required_resource_names=required_resource_names,
        required_timing_event_names=required_timing_event_names,
    )
    compiled, descriptor_set_views = _pool_family_schedule_classes(
        compiled,
        views.descriptor_set_views_for_specs(compiled, view_specs),
    )
    return compiled, descriptor_set_views


def generate_descriptor_set_family(
    storage_spec: DescriptorSet,
    view_specs: Sequence[DescriptorSet],
    *,
    primary_strings: CStringBank | None = None,
    shared_strings: CStringBank | None = None,
) -> GeneratedDescriptorSetFamily:
    """Generates shared C storage and public headers for descriptor-set views."""
    compiled, descriptor_set_views = compile_descriptor_set_family(
        storage_spec,
        view_specs,
    )
    return GeneratedDescriptorSetFamily(
        source=c_emit.emit_source_for_views(
            compiled,
            views=descriptor_set_views,
            primary_strings=primary_strings,
            shared_strings=shared_strings,
        ),
        view_headers=tuple(
            c_emit.emit_header_for_spec(
                compiled,
                view.spec,
                view.runtime_descriptor_ordinals,
            )
            for view in descriptor_set_views
        ),
    )


def write_descriptor_set_to_paths(
    spec: DescriptorSet,
    *,
    header_path: Path,
    source_path: Path,
    allowlist: DescriptorAllowlist | None = None,
    primary_strings: CStringBank | None = None,
    shared_strings: CStringBank | None = None,
) -> None:
    generated = generate_descriptor_set(
        spec,
        allowlist,
        primary_strings=primary_strings,
        shared_strings=shared_strings,
    )
    write_text_file(header_path, generated.header)
    write_text_file(source_path, generated.source)
