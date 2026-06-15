# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C emission for compiled low descriptor sets."""

from __future__ import annotations

from collections.abc import Sequence

from loom.gen.support import c_arrays
from loom.gen.support.c import c_string_literal
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.low import c_spelling, validation
from loom.gen.target.low.compiled import (
    CompiledAsmForm,
    CompiledDescriptorSet,
    CompiledNativeAsmValue,
    CompiledOperandForm,
    DescriptorSetView,
)
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_SET_ORDINAL_NONE,
    Descriptor,
    DescriptorSet,
    Hazard,
    HazardReferenceKind,
    descriptor_stable_id,
)


def _register_part_id_expr(compiled: CompiledDescriptorSet, part_name: str | None) -> str:
    if part_name is None:
        return "LOOM_LOW_REGISTER_PART_NONE"
    return str(compiled.register_part_ids[part_name])


def _hazard_reference_kind(hazard: Hazard) -> HazardReferenceKind:
    if hazard.resource is not None:
        return HazardReferenceKind.RESOURCE
    if hazard.counter_id is not None:
        return HazardReferenceKind.COUNTER
    if hazard.target_id is not None:
        return HazardReferenceKind.TARGET
    raise ValueError("hazard has no reference")


def _hazard_reference_id(hazard: Hazard, resource_ids: dict[str, int]) -> int:
    if hazard.resource is not None:
        return resource_ids[hazard.resource]
    if hazard.counter_id is not None:
        return hazard.counter_id
    if hazard.target_id is not None:
        return hazard.target_id
    raise ValueError("hazard has no reference")


def emit_header_for_spec(
    compiled: CompiledDescriptorSet,
    header_spec: DescriptorSet,
) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.low.low_descriptors"),
        "",
        f"#ifndef {header_spec.header_guard}",
        f"#define {header_spec.header_guard}",
        "",
        '#include "loom/codegen/low/descriptors.h"',
        "",
    ]
    lines.extend(c_spelling.descriptor_ref_define(header_spec, descriptor.key, i) for i, descriptor in enumerate(header_spec.descriptors))
    lines.append(f"#define {header_spec.c_enum_prefix}_DESCRIPTOR_SET_ID UINT64_C(0x{descriptor_stable_id(header_spec.key):016x})")
    lines.append(
        f"#define {header_spec.c_enum_prefix}_DESCRIPTOR_SET_ORDINAL {c_spelling.u16_literal(header_spec.descriptor_set_ordinal if header_spec.descriptor_set_ordinal is not None else LOW_DESCRIPTOR_SET_ORDINAL_NONE)}"
    )
    if header_spec.target_key is not None:
        lines.append(f"#define {header_spec.c_enum_prefix}_TARGET_ID UINT64_C(0x{descriptor_stable_id(header_spec.target_key):016x})")
    missing_reg_classes = [reg_class.name for reg_class in header_spec.reg_classes if reg_class.name not in compiled.reg_class_ids]
    if missing_reg_classes:
        raise ValueError(f"descriptor set header '{header_spec.key}' references reg classes missing from storage set '{compiled.spec.key}': {', '.join(missing_reg_classes)}")
    header_reg_class_ids = [(reg_class, compiled.reg_class_ids[reg_class.name]) for reg_class in header_spec.reg_classes]
    if header_reg_class_ids:
        lines.append("")
        lines.extend(
            c_spelling.emit_id_enum(
                f"{header_spec.c_enum_prefix.lower()}_reg_class_id",
                [
                    (
                        c_spelling.reg_class_id_constant_name(header_spec, reg_class.name),
                        reg_class_id,
                    )
                    for reg_class, reg_class_id in header_reg_class_ids
                ],
            )
        )
    header_register_part_ids = [(part, compiled.register_part_ids[part.name]) for part in header_spec.register_parts if part.name in compiled.register_part_ids]
    if header_register_part_ids:
        lines.append("")
        lines.extend(
            c_spelling.emit_id_enum(
                f"{header_spec.c_enum_prefix.lower()}_register_part_id",
                [
                    (
                        c_spelling.register_part_id_constant_name(header_spec, part.name),
                        part_id,
                    )
                    for part, part_id in header_register_part_ids
                ],
            )
        )
    lines.append("")
    lines.extend(
        [
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"const loom_low_descriptor_set_t* {header_spec.function_name}(void);",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            f"#endif  // {header_spec.header_guard}",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_header(compiled: CompiledDescriptorSet) -> str:
    return emit_header_for_spec(compiled, compiled.spec)


def _emit_string_table(compiled: CompiledDescriptorSet, lines: list[str]) -> None:
    spec = compiled.spec
    pool = compiled.string_pool
    lines.extend(
        [
            "// clang-format off",
            f"static const uint8_t k{spec.c_table_prefix}StringData[] =",
        ]
    )
    for entry in pool.entries:
        length = len(entry.value.encode())
        escaped = c_string_literal(entry.value)
        lines.append(f'    LOOM_BSTRING_LITERAL({length}, "{escaped}")')
    lines[-1] += ";"
    lines.append("// clang-format on")
    lines.append("")
    lines.append("enum {")
    previous_label: str | None = None
    entries_by_label = pool.entries_by_label
    for entry in pool.entries:
        enum_name = f"{pool.c_enum_prefix}_STRING_{entry.label}"
        if previous_label is None:
            lines.append(f"  {enum_name} = 0,")
        else:
            previous_entry = entries_by_label[previous_label]
            previous_enum_name = f"{pool.c_enum_prefix}_STRING_{previous_label}"
            lines.append(f'  {enum_name} = {previous_enum_name} + sizeof("{c_string_literal(previous_entry.value)}"),')
        previous_label = entry.label
    if previous_label is None:
        lines.append(f"  {pool.c_enum_prefix}_STRING_END = 0,")
    else:
        previous_entry = entries_by_label[previous_label]
        previous_enum_name = f"{pool.c_enum_prefix}_STRING_{previous_label}"
        lines.append(f'  {pool.c_enum_prefix}_STRING_END = {previous_enum_name} + sizeof("{c_string_literal(previous_entry.value)}"),')
    lines.append("};")
    lines.append("")
    lines.append(f'static_assert({pool.c_enum_prefix}_STRING_END == sizeof(k{spec.c_table_prefix}StringData) - 1, "descriptor string offsets must cover the table payload");')
    lines.append("")


def _emit_array(
    lines: list[str],
    c_type: str,
    table_prefix: str,
    name: str,
    row_lines: Sequence[list[str]],
) -> None:
    c_arrays.append_struct_array(lines, c_type, f"k{table_prefix}{name}", row_lines)


def _metadata_string_label(storage_spec: DescriptorSet, view_spec: DescriptorSet, field_name: str) -> str:
    if view_spec.key == storage_spec.key and view_spec.target_key == storage_spec.target_key and view_spec.feature_key == storage_spec.feature_key:
        return field_name
    return f"{field_name}_{c_spelling.c_identifier(view_spec.key)}"


def _intern_descriptor_set_view_metadata(compiled: CompiledDescriptorSet, view_spec: DescriptorSet) -> None:
    pool = compiled.string_pool
    storage_spec = compiled.spec
    pool.intern(_metadata_string_label(storage_spec, view_spec, "set_key"), view_spec.key)
    if view_spec.target_key is not None:
        pool.intern(
            _metadata_string_label(storage_spec, view_spec, "target_key"),
            view_spec.target_key,
        )
    if view_spec.feature_key is not None:
        pool.intern(
            _metadata_string_label(storage_spec, view_spec, "feature_key"),
            view_spec.feature_key,
        )


def _descriptor_row_lines(
    compiled: CompiledDescriptorSet,
    descriptors: Sequence[Descriptor],
    descriptor_rows: Sequence[dict[str, int]],
    canonical_asm_form_ordinals: Sequence[int | None],
) -> list[list[str]]:
    pool = compiled.string_pool
    return [
        [
            f".key_string_offset = {pool.ref(f'descriptor_{descriptor.key}')},",
            f".stable_id = {c_spelling.hex_u64_literal(descriptor_stable_id(descriptor.key))},",
            f".mnemonic_string_offset = {c_spelling.optional_string_expr(pool, f'mnemonic_{descriptor.key}' if descriptor.mnemonic is not None else None)},",
            f".semantic_tag_string_offset = {c_spelling.optional_string_expr(pool, f'semantic_{descriptor.key}' if descriptor.semantic_tag is not None else None)},",
            f".feature_mask_word_start = {descriptor_rows[i]['feature_mask_word_start']},",
            f".feature_mask_word_count = {descriptor_rows[i]['feature_mask_word_count']},",
            f".encoding_field_value_start = {descriptor_rows[i]['encoding_field_value_start']},",
            f".encoding_field_value_count = {descriptor_rows[i]['encoding_field_value_count']},",
            f".encoding_format_id = {descriptor.encoding_format_id},",
            f".encoding_id = {c_spelling.encoding_id_expr(descriptor.encoding_id)},",
            f".operand_start = {descriptor_rows[i]['operand_start']},",
            f".operand_count = {descriptor_rows[i]['operand_count']},",
            f".result_count = {descriptor_rows[i]['result_count']},",
            f".immediate_start = {descriptor_rows[i]['immediate_start']},",
            f".immediate_count = {descriptor_rows[i]['immediate_count']},",
            f".effect_start = {descriptor_rows[i]['effect_start']},",
            f".effect_count = {descriptor_rows[i]['effect_count']},",
            f".constraint_start = {descriptor_rows[i]['constraint_start']},",
            f".constraint_count = {descriptor_rows[i]['constraint_count']},",
            f".storage_lease_start = {descriptor_rows[i]['storage_lease_start']},",
            f".storage_lease_count = {descriptor_rows[i]['storage_lease_count']},",
            f".operand_form_start = {descriptor_rows[i]['operand_form_start']},",
            f".operand_form_count = {descriptor_rows[i]['operand_form_count']},",
            f".schedule_class_id = {compiled.schedule_class_ids[descriptor.schedule_class]},",
            f".flags = {c_spelling.flag_expr(descriptor.flags)},",
            f".canonical_asm_form_ordinal = {c_spelling.canonical_asm_form_ordinal_expr(canonical_asm_form_ordinals[i])},",
        ]
        for i, descriptor in enumerate(descriptors)
    ]


def _storage_lease_row_lines(compiled: CompiledDescriptorSet) -> list[list[str]]:
    pool = compiled.string_pool
    return [
        [
            f".kind = {lease.kind.c_name},",
            f".attachment = {lease.attachment.c_name},",
            f".attachment_index = {lease.attachment_index},",
            f".unit_offset = {lease.unit_offset},",
            f".unit_count = {lease.unit_count},",
            f".release_scope = {lease.release_scope.c_name},",
            f".release_class_id = {lease.release_class_id},",
            f".release_class_name_string_offset = {pool.ref(f'storage_lease_{descriptor_key}_{lease_index}_class')},",
            f".release_action_id = {lease.release_action_id},",
            f".release_action_name_string_offset = {pool.ref(f'storage_lease_{descriptor_key}_{lease_index}_action')},",
            f".release_reason_id = {lease.release_reason_id},",
            f".release_reason_name_string_offset = {pool.ref(f'storage_lease_{descriptor_key}_{lease_index}_reason')},",
            f".flags = {c_spelling.flag_expr(lease.flags)},",
        ]
        for (descriptor_key, lease_index), lease in zip(compiled.storage_lease_labels, compiled.storage_leases, strict=True)
    ]


def _operand_form_row_lines(
    operand_forms: Sequence[CompiledOperandForm],
) -> list[list[str]]:
    return [
        [
            f".replacement_descriptor_ordinal = {operand_form.replacement_descriptor_ordinal},",
            f".operand_map_start = {operand_form.operand_map_start},",
            f".match_start = {operand_form.match_start},",
            f".source_immediate_index = {operand_form.source_immediate_index},",
            f".replacement_immediate_index = {operand_form.replacement_immediate_index},",
            f".immediate_match_index = {operand_form.immediate_match_index},",
            f".operand_map_count = {operand_form.operand_map_count},",
            f".immediate_action = {operand_form.immediate_action.c_name},",
            f".match_count = {operand_form.match_count},",
        ]
        for operand_form in operand_forms
    ]


def _asm_form_row_lines(
    compiled: CompiledDescriptorSet,
    asm_forms: Sequence[CompiledAsmForm],
) -> list[list[str]]:
    pool = compiled.string_pool
    return [
        [
            f".mnemonic_string_offset = {pool.ref(asm_form.mnemonic_label)},",
            f".native_assembly_mnemonic_string_offset = {c_spelling.optional_string_expr(pool, asm_form.native_assembly_mnemonic_label)},",
            f".descriptor_ordinal = {asm_form.descriptor_ordinal},",
            f".result_operand_index_start = {asm_form.result_index_start},",
            f".result_operand_index_count = {len(asm_form.result_indices)},",
            f".operand_index_start = {asm_form.operand_index_start},",
            f".operand_index_count = {len(asm_form.operand_indices)},",
            f".immediate_start = {asm_form.immediate_start},",
            f".immediate_count = {len(asm_form.immediates)},",
            f".native_assembly_value_start = {asm_form.native_assembly_value_start},",
            f".native_assembly_value_count = {len(asm_form.native_assembly_values)},",
        ]
        for asm_form in asm_forms
    ]


def _native_asm_value_row_lines(
    compiled: CompiledDescriptorSet,
    values: Sequence[CompiledNativeAsmValue],
) -> list[list[str]]:
    pool = compiled.string_pool
    return [
        [
            f".kind = {value.kind.c_name},",
            f".index = {value.index},",
            f".bit_width = {value.bit_width},",
            f".literal_string_offset = {c_spelling.optional_string_expr(pool, value.literal_label)},",
        ]
        for value in values
    ]


def emit_source_for_views(
    compiled: CompiledDescriptorSet,
    *,
    views: Sequence[DescriptorSetView],
) -> str:
    spec = compiled.spec
    pool = compiled.string_pool
    for view in views:
        _intern_descriptor_set_view_metadata(compiled, view.spec)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.low.low_descriptors"),
        "",
        f'#include "{spec.public_header}"',
        "",
        "#include <stdint.h>",
        "",
    ]
    for view in views:
        if view.spec is spec:
            continue
        lines.append(f"const loom_low_descriptor_set_t* {view.spec.function_name}(void);")
    if len(views) > 1:
        lines.append("")
    _emit_string_table(compiled, lines)

    _emit_array(
        lines,
        "loom_low_reg_class_t",
        spec.c_table_prefix,
        "RegClasses",
        [
            [
                f".name_string_offset = {pool.ref(f'reg_{reg_class.name}')},",
                f".target_bank_id = {reg_class.target_bank_id},",
                f".flags = {c_spelling.flag_expr(reg_class.flags)},",
                f".alloc_unit_bits = {reg_class.alloc_unit_bits},",
                f".allocatable_count = {reg_class.allocatable_count},",
                f".alias_set_id = {reg_class.alias_set_id},",
                ".spill_class_id = " + ("LOOM_LOW_REG_CLASS_NONE" if reg_class.spill_class is None else str(compiled.reg_class_ids[reg_class.spill_class])) + ",",
                f".full_register_part_mask = {c_spelling.hex_u32_literal(reg_class.full_register_part_mask)},",
                f".spill_slot_space = {reg_class.spill_slot_space.c_name},",
            ]
            for reg_class in compiled.reg_classes
        ],
    )
    _emit_array(
        lines,
        "loom_low_register_part_t",
        spec.c_table_prefix,
        "RegisterParts",
        [
            [
                f".name_string_offset = {pool.ref(f'register_part_{part.name}')},",
                f".reg_class_id = {compiled.reg_class_ids[part.reg_class]},",
                ".reserved = 0,",
                f".mask = {c_spelling.hex_u32_literal(part.mask)},",
            ]
            for part in compiled.register_parts
        ],
    )
    _emit_array(
        lines,
        "loom_low_reg_class_alt_t",
        spec.c_table_prefix,
        "RegClassAlts",
        [
            [
                ".reg_class_id = " + ("LOOM_LOW_REG_CLASS_NONE" if reg_class_id is None else str(reg_class_id)) + ",",
                f".flags = {c_spelling.flag_expr(flags)},",
            ]
            for reg_class_id, flags in compiled.reg_class_alts
        ],
    )
    _emit_array(
        lines,
        "loom_low_operand_t",
        spec.c_table_prefix,
        "Operands",
        [
            [
                f".field_name_string_offset = {pool.ref(f'field_{operand.field_name}')},",
                f".role = {operand.role.c_name},",
                f".flags = {c_spelling.flag_expr(operand.flags)},",
                f".reg_class_alt_start = {compiled.operand_alt_starts[i]},",
                f".reg_class_alt_count = {len(operand.reg_alts)},",
                f".unit_count = {operand.unit_count},",
                f".address_map_kind = {operand.address_map_kind.c_name},",
                f".addressable_unit_count = {operand.addressable_unit_count},",
                f".encoding_field_id = {operand.encoding_field_id},",
                f".data_format_id = {operand.data_format_id},",
                f".register_part_id = {_register_part_id_expr(compiled, operand.register_part)},",
                f".read_stage = {operand.read_stage},",
                f".ready_stage = {operand.ready_stage},",
            ]
            for i, operand in enumerate(compiled.operands)
        ],
    )
    _emit_array(
        lines,
        "loom_low_immediate_t",
        spec.c_table_prefix,
        "Immediates",
        [
            [
                f".field_name_string_offset = {pool.ref(f'immediate_{immediate.field_name}')},",
                f".encoding_slice_start = {compiled.immediate_encoding_slice_starts[i]},",
                f".kind = {immediate.kind.c_name},",
                f".flags = {c_spelling.flag_expr(immediate.flags)},",
                f".bit_width = {immediate.bit_width},",
                f".encoding_field_id = {immediate.encoding_field_id},",
                f".encoding_slice_count = {len(immediate.encoding_slices)},",
                ".enum_domain_id = " + ("LOOM_LOW_ENUM_DOMAIN_NONE" if compiled.immediate_enum_domain_ids[i] is None else str(compiled.immediate_enum_domain_ids[i])) + ",",
                f".encoding_id = {immediate.encoding_id},",
                f".signed_min = {c_spelling.i64_literal(immediate.signed_min)},",
                f".unsigned_max = {c_spelling.u64_literal(immediate.unsigned_max)},",
                f".default_value = {c_spelling.i64_literal(immediate.default_value)},",
            ]
            for i, immediate in enumerate(compiled.immediates)
        ],
    )
    _emit_array(
        lines,
        "loom_low_immediate_encoding_slice_t",
        spec.c_table_prefix,
        "ImmediateEncodingSlices",
        [
            [
                f".encoding_field_id = {encoding_slice.encoding_field_id},",
                f".source_bit_offset = {encoding_slice.source_bit_offset},",
                f".bit_count = {encoding_slice.bit_count},",
            ]
            for encoding_slice in compiled.immediate_encoding_slices
        ],
    )
    _emit_array(
        lines,
        "loom_low_enum_domain_t",
        spec.c_table_prefix,
        "EnumDomains",
        [
            [
                f".name_string_offset = {pool.ref(f'enum_domain_{domain.name}')},",
                f".value_start = {compiled.enum_domain_rows[i]['value_start']},",
                f".value_count = {compiled.enum_domain_rows[i]['value_count']},",
            ]
            for i, domain in enumerate(compiled.enum_domains)
        ],
    )
    _emit_array(
        lines,
        "loom_low_enum_value_t",
        spec.c_table_prefix,
        "EnumValues",
        [
            [
                f".token_string_offset = {pool.ref(f'enum_value_{domain.name}_{value.token}')},",
                f".value = {c_spelling.i64_literal(value.value)},",
            ]
            for domain in compiled.enum_domains
            for value in validation.validate_enum_domain(domain)
        ],
    )
    _emit_array(
        lines,
        "loom_low_effect_t",
        spec.c_table_prefix,
        "Effects",
        [
            [
                f".kind = {effect.kind.c_name},",
                f".memory_space = {effect.memory_space.c_name},",
                f".scope_id = {effect.scope_id},",
                f".flags = {c_spelling.flag_expr(effect.flags)},",
                f".counter_id = {effect.counter_id},",
                f".width_bits = {effect.width_bits},",
            ]
            for effect in compiled.effects
        ],
    )
    _emit_array(
        lines,
        "loom_low_constraint_t",
        spec.c_table_prefix,
        "Constraints",
        [
            [
                f".kind = {constraint.kind.c_name},",
                f".lhs_operand_index = {constraint.lhs_operand_index},",
                ".rhs_operand_index = " + ("LOOM_LOW_ID_NONE" if constraint.rhs_operand_index is None else str(constraint.rhs_operand_index)) + ",",
                f".flags = {c_spelling.flag_expr(constraint.flags)},",
            ]
            for constraint in compiled.constraints
        ],
    )
    _emit_array(
        lines,
        "loom_low_descriptor_storage_lease_t",
        spec.c_table_prefix,
        "StorageLeases",
        _storage_lease_row_lines(compiled),
    )
    _emit_array(
        lines,
        "loom_low_resource_t",
        spec.c_table_prefix,
        "Resources",
        [
            [
                f".name_string_offset = {pool.ref(f'resource_{resource.name}')},",
                f".capacity_per_cycle = {resource.capacity_per_cycle},",
                f".flags = {c_spelling.flag_expr(resource.flags)},",
                f".kind = {resource.kind.c_name},",
                f".contention_group_id = {resource.contention_group_id},",
            ]
            for resource in compiled.resources
        ],
    )
    _emit_array(
        lines,
        "loom_low_issue_use_t",
        spec.c_table_prefix,
        "IssueUses",
        [
            [
                f".resource_id = {compiled.resource_ids[issue_use.resource]},",
                f".cycles = {issue_use.cycles},",
                f".units = {issue_use.units},",
                f".stage = {issue_use.stage},",
            ]
            for issue_use in compiled.issue_uses
        ],
    )
    _emit_array(
        lines,
        "loom_low_hazard_t",
        spec.c_table_prefix,
        "Hazards",
        [
            [
                f".kind = {hazard.kind.c_name},",
                f".reference_kind = {_hazard_reference_kind(hazard).c_name},",
                f".reference_id = {_hazard_reference_id(hazard, compiled.resource_ids)},",
                f".producer_stage = {hazard.producer_stage},",
                f".consumer_stage = {hazard.consumer_stage},",
                f".distance = {hazard.distance},",
                f".flags = {c_spelling.flag_expr(hazard.flags)},",
            ]
            for hazard in compiled.hazards
        ],
    )
    _emit_array(
        lines,
        "loom_low_pressure_delta_t",
        spec.c_table_prefix,
        "PressureDeltas",
        [
            [
                f".reg_class_id = {compiled.reg_class_ids[pressure.reg_class]},",
                f".delta = {pressure.delta},",
            ]
            for pressure in compiled.pressure_deltas
        ],
    )
    _emit_array(
        lines,
        "loom_low_schedule_class_t",
        spec.c_table_prefix,
        "ScheduleClasses",
        [
            [
                f".name_string_offset = {pool.ref(f'schedule_{schedule_class.name}')},",
                f".latency_cycles = {schedule_class.latency_cycles},",
                f".latency_kind = {schedule_class.latency_kind.c_name},",
                f".issue_use_start = {compiled.schedule_rows[i]['issue_use_start']},",
                f".issue_use_count = {compiled.schedule_rows[i]['issue_use_count']},",
                f".hazard_start = {compiled.schedule_rows[i]['hazard_start']},",
                f".hazard_count = {compiled.schedule_rows[i]['hazard_count']},",
                f".flags = {c_spelling.flag_expr(schedule_class.flags)},",
                f".model_quality = {schedule_class.model_quality.c_name},",
                f".pressure_delta_start = {compiled.schedule_rows[i]['pressure_delta_start']},",
                f".pressure_delta_count = {compiled.schedule_rows[i]['pressure_delta_count']},",
            ]
            for i, schedule_class in enumerate(compiled.schedule_classes)
        ],
    )
    if compiled.feature_mask_words:
        c_arrays.append_value_array(
            lines,
            "uint64_t",
            f"k{spec.c_table_prefix}FeatureMaskWords",
            [c_spelling.hex_u64_literal(word) for word in compiled.feature_mask_words],
        )
    _emit_array(
        lines,
        "loom_low_encoding_field_value_t",
        spec.c_table_prefix,
        "EncodingFieldValues",
        [
            [
                f".encoding_field_id = {field_value.encoding_field_id},",
                ".reserved = 0,",
                f".value = {c_spelling.u64_literal(field_value.value)},",
            ]
            for field_value in compiled.encoding_field_values
        ],
    )
    _emit_array(
        lines,
        "loom_low_operand_form_t",
        spec.c_table_prefix,
        "OperandForms",
        _operand_form_row_lines(compiled.operand_forms),
    )
    _emit_array(
        lines,
        "loom_low_operand_form_match_t",
        spec.c_table_prefix,
        "OperandFormMatches",
        [
            [
                f".source_operand_index = {match.source_operand_index},",
                f".source_packet_operand_index = {match.source_packet_operand_index},",
                f".match_kind = {match.match_kind.c_name},",
                f".match_i64 = {c_spelling.i64_literal(match.match_i64)},",
            ]
            for match in compiled.operand_form_matches
        ],
    )
    if compiled.operand_form_operand_indices:
        c_arrays.append_value_array(
            lines,
            "uint16_t",
            f"k{spec.c_table_prefix}OperandFormOperandIndices",
            [str(operand_index) for operand_index in compiled.operand_form_operand_indices],
        )
    _emit_array(
        lines,
        "loom_low_descriptor_t",
        spec.c_table_prefix,
        "Descriptors",
        _descriptor_row_lines(
            compiled,
            compiled.descriptors,
            compiled.descriptor_rows,
            compiled.canonical_asm_form_ordinals,
        ),
    )
    for view in views:
        if view.uses_storage_descriptor_tables:
            continue
        view_descriptors = [compiled.descriptors[descriptor_ordinal] for descriptor_ordinal in view.descriptor_ordinals]
        _emit_array(
            lines,
            "loom_low_descriptor_t",
            view.spec.c_table_prefix,
            "Descriptors",
            _descriptor_row_lines(
                compiled,
                view_descriptors,
                view.descriptor_rows,
                view.canonical_asm_form_ordinals,
            ),
        )
        _emit_array(
            lines,
            "loom_low_operand_form_t",
            view.spec.c_table_prefix,
            "OperandForms",
            _operand_form_row_lines(view.operand_forms),
        )
    for view in views:
        _emit_array(
            lines,
            "loom_low_descriptor_ref_t",
            view.spec.c_table_prefix,
            "DescriptorRefs",
            [
                [
                    f".key_string_offset = {pool.ref(f'descriptor_{descriptor_key}')},",
                    f".descriptor_ordinal = {descriptor_ordinal},",
                ]
                for descriptor_key, descriptor_ordinal in view.descriptor_refs
            ],
        )
    if compiled.asm_operand_indices:
        c_arrays.append_value_array(
            lines,
            "uint16_t",
            f"k{spec.c_table_prefix}AsmOperandIndices",
            [str(operand_index) for operand_index in compiled.asm_operand_indices],
        )
    _emit_array(
        lines,
        "loom_low_asm_immediate_t",
        spec.c_table_prefix,
        "AsmImmediates",
        [
            [
                f".immediate_index = {immediate.immediate_index},",
                f".name_string_offset = {c_spelling.optional_string_expr(pool, immediate.name_label)},",
            ]
            for immediate in compiled.asm_immediates
        ],
    )
    _emit_array(
        lines,
        "loom_low_asm_form_t",
        spec.c_table_prefix,
        "AsmForms",
        _asm_form_row_lines(compiled, compiled.asm_forms),
    )
    _emit_array(
        lines,
        "loom_low_native_asm_value_t",
        spec.c_table_prefix,
        "NativeAsmValues",
        _native_asm_value_row_lines(compiled, compiled.native_asm_values),
    )
    for view in views:
        if view.uses_storage_asm_form_tables:
            continue
        _emit_array(
            lines,
            "loom_low_asm_form_t",
            view.spec.c_table_prefix,
            "AsmForms",
            _asm_form_row_lines(compiled, view.asm_forms),
        )

    table_count_fields = {
        "operands": "operand_count",
        "immediates": "immediate_count",
        "immediate_encoding_slices": "immediate_encoding_slice_count",
        "enum_domains": "enum_domain_count",
        "enum_values": "enum_value_count",
        "effects": "effect_count",
        "constraints": "constraint_count",
        "storage_leases": "storage_lease_count",
        "reg_classes": "reg_class_count",
        "register_parts": "register_part_count",
        "reg_class_alts": "reg_class_alt_count",
        "schedule_classes": "schedule_class_count",
        "issue_uses": "issue_use_count",
        "resources": "resource_count",
        "hazards": "hazard_count",
        "pressure_deltas": "pressure_delta_count",
        "asm_forms": "asm_form_count",
        "asm_operand_indices": "asm_operand_index_count",
        "asm_immediates": "asm_immediate_count",
        "native_asm_values": "native_asm_value_count",
        "encoding_field_values": "encoding_field_value_count",
        "operand_forms": "operand_form_count",
        "operand_form_matches": "operand_form_match_count",
        "operand_form_operand_indices": "operand_form_operand_index_count",
    }

    def append_optional_table(field_name: str, table_name: str, view_lines: list[str]) -> None:
        rows = getattr(compiled, field_name)
        if rows:
            view_lines.append(f"    .{field_name} = k{spec.c_table_prefix}{table_name},")
            view_lines.append(f"    .{table_count_fields[field_name]} = IREE_ARRAYSIZE(k{spec.c_table_prefix}{table_name}),")

    for view in views:
        view_spec = view.spec
        descriptor_table_prefix = spec.c_table_prefix if view.uses_storage_descriptor_tables else view_spec.c_table_prefix
        asm_form_table_prefix = spec.c_table_prefix if view.uses_storage_asm_form_tables else view_spec.c_table_prefix
        operand_form_table_prefix = spec.c_table_prefix if view.uses_storage_operand_form_tables else view_spec.c_table_prefix
        view_lines = [
            f"static const loom_low_descriptor_set_t k{view_spec.c_table_prefix}Set = {{",
            "    .abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION,",
            f"    .generator_version = {view_spec.generator_version},",
            f"    .stable_id = UINT64_C(0x{descriptor_stable_id(view_spec.key):016x}),",
            f"    .target_stable_id = {c_spelling.hex_u64_literal(descriptor_stable_id(view_spec.target_key)) if view_spec.target_key is not None else 'LOOM_LOW_STABLE_ID_NONE'},",
            f"    .descriptor_set_ordinal = {c_spelling.u16_literal(view_spec.descriptor_set_ordinal if view_spec.descriptor_set_ordinal is not None else LOW_DESCRIPTOR_SET_ORDINAL_NONE)},",
            f"    .key_string_offset = {pool.ref(_metadata_string_label(spec, view_spec, 'set_key'))},",
            f"    .target_key_string_offset = {c_spelling.optional_string_expr(pool, _metadata_string_label(spec, view_spec, 'target_key') if view_spec.target_key is not None else None)},",
            f"    .feature_key_string_offset = {c_spelling.optional_string_expr(pool, _metadata_string_label(spec, view_spec, 'feature_key') if view_spec.feature_key is not None else None)},",
            "    .string_table =",
            "        {",
            f"            .data = k{spec.c_table_prefix}StringData,",
            f"            .data_length = sizeof(k{spec.c_table_prefix}StringData) - 1,",
            "        },",
            f"    .descriptors = k{descriptor_table_prefix}Descriptors,",
            f"    .descriptor_count = {view.descriptor_count},",
            f"    .descriptor_refs = k{view_spec.c_table_prefix}DescriptorRefs,",
            f"    .descriptor_ref_count = IREE_ARRAYSIZE(k{view_spec.c_table_prefix}DescriptorRefs),",
        ]

        append_optional_table("operands", "Operands", view_lines)
        append_optional_table("immediates", "Immediates", view_lines)
        append_optional_table("immediate_encoding_slices", "ImmediateEncodingSlices", view_lines)
        append_optional_table("enum_domains", "EnumDomains", view_lines)
        append_optional_table("enum_values", "EnumValues", view_lines)
        append_optional_table("effects", "Effects", view_lines)
        append_optional_table("constraints", "Constraints", view_lines)
        append_optional_table("storage_leases", "StorageLeases", view_lines)
        append_optional_table("reg_classes", "RegClasses", view_lines)
        append_optional_table("register_parts", "RegisterParts", view_lines)
        append_optional_table("reg_class_alts", "RegClassAlts", view_lines)
        append_optional_table("schedule_classes", "ScheduleClasses", view_lines)
        append_optional_table("issue_uses", "IssueUses", view_lines)
        append_optional_table("resources", "Resources", view_lines)
        append_optional_table("hazards", "Hazards", view_lines)
        append_optional_table("pressure_deltas", "PressureDeltas", view_lines)
        if view.asm_forms:
            view_lines.append(f"    .asm_forms = k{asm_form_table_prefix}AsmForms,")
            view_lines.append(f"    .asm_form_count = IREE_ARRAYSIZE(k{asm_form_table_prefix}AsmForms),")
            append_optional_table("asm_operand_indices", "AsmOperandIndices", view_lines)
            append_optional_table("asm_immediates", "AsmImmediates", view_lines)
            append_optional_table("native_asm_values", "NativeAsmValues", view_lines)
        append_optional_table("encoding_field_values", "EncodingFieldValues", view_lines)
        if view.operand_forms:
            view_lines.append(f"    .operand_forms = k{operand_form_table_prefix}OperandForms,")
            view_lines.append(f"    .operand_form_count = IREE_ARRAYSIZE(k{operand_form_table_prefix}OperandForms),")
        append_optional_table(
            "operand_form_matches",
            "OperandFormMatches",
            view_lines,
        )
        append_optional_table(
            "operand_form_operand_indices",
            "OperandFormOperandIndices",
            view_lines,
        )
        if compiled.feature_mask_words:
            view_lines.append(f"    .feature_mask_words = k{spec.c_table_prefix}FeatureMaskWords,")
            view_lines.append(f"    .feature_mask_word_count = IREE_ARRAYSIZE(k{spec.c_table_prefix}FeatureMaskWords),")
        view_lines.append("};")
        view_lines.append("")
        view_lines.append(f"const loom_low_descriptor_set_t* {view_spec.function_name}(void) {{")
        view_lines.append(f"  return &k{view_spec.c_table_prefix}Set;")
        view_lines.append("}")
        lines.extend(view_lines)
        lines.append("")
    return "\n".join(lines) + "\n"


def emit_source(compiled: CompiledDescriptorSet) -> str:
    return emit_source_for_views(
        compiled,
        views=[
            DescriptorSetView(
                spec=compiled.spec,
                descriptor_ordinals=tuple(range(len(compiled.descriptors))),
                descriptor_refs=compiled.descriptor_refs,
                descriptor_rows=compiled.descriptor_rows,
                canonical_asm_form_ordinals=compiled.canonical_asm_form_ordinals,
                asm_forms=compiled.asm_forms,
                operand_forms=compiled.operand_forms,
                uses_storage_descriptor_tables=True,
                uses_storage_asm_form_tables=True,
                uses_storage_operand_form_tables=True,
            )
        ],
    )
