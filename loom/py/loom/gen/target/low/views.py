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
    CompiledOperandForm,
    DescriptorSetView,
)
from loom.target.low_descriptors import Descriptor, DescriptorSet


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
) -> None:
    for view_descriptor, storage_descriptor_ordinal in zip(view_spec.descriptors, descriptor_ordinals, strict=True):
        storage_descriptor = compiled.descriptors[storage_descriptor_ordinal]
        if (
            replace(
                view_descriptor,
                asm_forms=storage_descriptor.asm_forms,
                asm_surface=storage_descriptor.asm_surface,
                asm_surface_reason=storage_descriptor.asm_surface_reason,
            )
            == storage_descriptor
        ):
            continue
        raise ValueError(
            f"descriptor set view '{view_spec.key}' descriptor '{view_descriptor.key}' differs from storage descriptor '{storage_descriptor.key}' outside of asm forms or asm surface policy"
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
    _validate_view_descriptors_match_storage(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    )
    _validate_view_operand_forms_closed(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    )
    if (
        descriptor_ordinal_tuple == tuple(range(len(descriptor_ordinal_tuple)))
        and _view_asm_forms_match_storage(compiled, view_spec, descriptor_ordinal_tuple)
        and not _asm_forms_have_duplicate_mnemonics(compiled.asm_forms)
    ):
        descriptor_count = len(descriptor_ordinal_tuple)
        return DescriptorSetView(
            spec=view_spec,
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
        uses_storage_asm_form_tables=False,
        uses_storage_operand_form_tables=False,
    )
