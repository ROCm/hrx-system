// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf_object.h"

#include <string.h>

#include "iree/base/alignment.h"

#define LOOM_NATIVE_ELF64LE_SYMBOL_SIZE 24u
#define LOOM_NATIVE_ELF64LE_RELA_SIZE 24u

typedef struct loom_native_elf_relocation_group_t {
  // First record assigned to this assembled section.
  iree_host_size_t record_base;
  // Next record to write, or the group end after serialization.
  iree_host_size_t next_record;
} loom_native_elf_relocation_group_t;

typedef struct loom_native_elf64le_relocatable_build_t {
  // Final ELF sections excluding the implicit null section and `.shstrtab`.
  loom_native_elf_section_t* sections;
  // Number of entries in |sections|.
  iree_host_size_t section_count;
} loom_native_elf64le_relocatable_build_t;

static uint8_t loom_native_elf_symbol_binding(uint32_t binding) {
  switch (binding) {
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL:
      return 0;
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL:
      return 1;
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_WEAK:
      return 2;
    default:
      IREE_ASSERT_UNREACHABLE("invalid native object symbol binding");
      return 0;
  }
}

static uint8_t loom_native_elf_symbol_type(uint32_t kind) {
  switch (kind) {
    case LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION:
      return 2;
    case LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA:
      return 1;
    default:
      IREE_ASSERT_UNREACHABLE("invalid native object symbol kind");
      return 0;
  }
}

static void loom_native_elf64le_store_symbol(
    uint8_t* target, uint32_t name_offset,
    const loom_native_object_symbol_t* symbol,
    const loom_native_object_symbol_layout_t* layout) {
  const bool is_undefined =
      symbol->definition == LOOM_NATIVE_OBJECT_SYMBOL_DEFINITION_UNDEFINED;
  const uint16_t section_index =
      is_undefined ? 0u : (uint16_t)(layout->section_index + 1u);
  const uint64_t value = is_undefined ? 0u : layout->section_offset;
  iree_unaligned_store_le_u32(target + 0, name_offset);
  iree_unaligned_store_le_u8(
      target + 4,
      (uint8_t)((loom_native_elf_symbol_binding(symbol->binding) << 4) |
                loom_native_elf_symbol_type(symbol->kind)));
  iree_unaligned_store_le_u8(target + 5, (uint8_t)symbol->visibility);
  iree_unaligned_store_le_u16(target + 6, section_index);
  iree_unaligned_store_le_u64(target + 8, value);
  iree_unaligned_store_le_u64(target + 16, symbol->size);
}

static iree_status_t loom_native_elf64le_build_symbol_tables(
    const loom_native_object_contribution_t* object,
    const loom_native_section_contribution_assembly_t* assembly,
    uint32_t** out_source_to_elf_indices, iree_const_byte_span_t* out_symbols,
    iree_const_byte_span_t* out_strings, uint32_t* out_first_nonlocal_symbol,
    iree_arena_allocator_t* arena) {
  *out_source_to_elf_indices = NULL;
  *out_symbols = iree_const_byte_span_empty();
  *out_strings = iree_const_byte_span_empty();
  *out_first_nonlocal_symbol = 1;

  if (object->symbol_count > UINT32_MAX - 1u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native object symbol count exceeds ELF64 "
                            "symbol-table capacity");
  }

  iree_host_size_t string_table_size = 1;
  for (iree_host_size_t i = 0; i < object->symbol_count; ++i) {
    if (!iree_host_size_checked_add(string_table_size,
                                    object->symbols[i].name.size,
                                    &string_table_size) ||
        !iree_host_size_checked_add(string_table_size, 1, &string_table_size) ||
        string_table_size > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native object symbol names exceed ELF64 "
                              "string-table capacity");
    }
  }

  loom_native_object_symbol_layout_t* symbol_layouts = NULL;
  uint32_t* source_to_elf_indices = NULL;
  if (object->symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, object->symbol_count,
                                                   sizeof(*symbol_layouts),
                                                   (void**)&symbol_layouts));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, object->symbol_count, sizeof(*source_to_elf_indices),
        (void**)&source_to_elf_indices));
  }
  IREE_RETURN_IF_ERROR(loom_native_object_resolve_symbol_layouts(
      object->symbols, object->symbol_count, assembly->contribution_layouts,
      assembly->contribution_layout_count, symbol_layouts));

  iree_host_size_t symbol_record_count = 0;
  if (!iree_host_size_checked_add(object->symbol_count, 1,
                                  &symbol_record_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native object symbol table size overflows");
  }
  uint8_t* symbol_bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_record_count, LOOM_NATIVE_ELF64LE_SYMBOL_SIZE,
      (void**)&symbol_bytes));
  memset(symbol_bytes, 0, LOOM_NATIVE_ELF64LE_SYMBOL_SIZE);
  uint8_t* string_bytes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, string_table_size, (void**)&string_bytes));
  string_bytes[0] = 0;

  uint32_t local_symbol_count = 0;
  for (iree_host_size_t i = 0; i < object->symbol_count; ++i) {
    local_symbol_count +=
        object->symbols[i].binding == LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL;
  }
  *out_first_nonlocal_symbol = local_symbol_count + 1u;
  uint32_t next_local_symbol_index = 1;
  uint32_t next_nonlocal_symbol_index = *out_first_nonlocal_symbol;
  iree_host_size_t next_string_offset = 1;
  for (iree_host_size_t i = 0; i < object->symbol_count; ++i) {
    const loom_native_object_symbol_t* symbol = &object->symbols[i];
    const uint32_t symbol_index =
        symbol->binding == LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL
            ? next_local_symbol_index++
            : next_nonlocal_symbol_index++;
    const uint32_t name_offset = (uint32_t)next_string_offset;
    memcpy(string_bytes + next_string_offset, symbol->name.data,
           symbol->name.size);
    next_string_offset += symbol->name.size;
    string_bytes[next_string_offset++] = 0;
    loom_native_elf64le_store_symbol(
        symbol_bytes +
            (iree_host_size_t)symbol_index * LOOM_NATIVE_ELF64LE_SYMBOL_SIZE,
        name_offset, symbol, &symbol_layouts[i]);
    source_to_elf_indices[i] = symbol_index;
  }

  *out_source_to_elf_indices = source_to_elf_indices;
  *out_symbols = iree_make_const_byte_span(
      symbol_bytes, symbol_record_count * LOOM_NATIVE_ELF64LE_SYMBOL_SIZE);
  *out_strings = iree_make_const_byte_span(string_bytes, string_table_size);
  return iree_ok_status();
}

static iree_status_t loom_native_elf64le_build_relocations(
    const loom_native_object_contribution_t* object,
    const loom_native_elf64le_relocatable_options_t* options,
    const loom_native_section_contribution_assembly_t* assembly,
    const uint32_t* source_to_elf_indices,
    loom_native_elf_relocation_group_t** out_groups,
    iree_const_byte_span_t* out_records, iree_host_size_t* out_group_count,
    iree_arena_allocator_t* arena) {
  *out_groups = NULL;
  *out_records = iree_const_byte_span_empty();
  *out_group_count = 0;
  loom_native_elf_relocation_group_t* groups = NULL;
  if (assembly->section_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, assembly->section_count, sizeof(*groups), (void**)&groups));
    memset(groups, 0, assembly->section_count * sizeof(*groups));
  }
  if (object->fixup_count == 0) {
    *out_groups = groups;
    return iree_ok_status();
  }
  if (options->relocation_mapper.map == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF64 relocation mapper is required for native "
                            "object fixups");
  }

  loom_native_object_fixup_layout_t* fixup_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, object->fixup_count,
                                                 sizeof(*fixup_layouts),
                                                 (void**)&fixup_layouts));
  IREE_RETURN_IF_ERROR(loom_native_object_resolve_fixup_layouts(
      object->fixups, object->fixup_count, object->symbol_count,
      assembly->contribution_layouts, assembly->contribution_layout_count,
      fixup_layouts));

  for (iree_host_size_t i = 0; i < object->fixup_count; ++i) {
    ++groups[fixup_layouts[i].section_index].next_record;
  }

  iree_host_size_t next_record_base = 0;
  iree_host_size_t group_count = 0;
  for (iree_host_size_t i = 0; i < assembly->section_count; ++i) {
    const iree_host_size_t record_count = groups[i].next_record;
    groups[i].record_base = next_record_base;
    groups[i].next_record = next_record_base;
    if (!iree_host_size_checked_add(next_record_base, record_count,
                                    &next_record_base)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native object relocation count overflows");
    }
    group_count += record_count != 0;
  }

  uint8_t* relocation_bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, object->fixup_count,
                                                 LOOM_NATIVE_ELF64LE_RELA_SIZE,
                                                 (void**)&relocation_bytes));
  for (iree_host_size_t i = 0; i < object->fixup_count; ++i) {
    const loom_native_object_fixup_t* fixup = &object->fixups[i];
    const loom_native_object_fixup_layout_t* layout = &fixup_layouts[i];
    loom_native_elf_relocation_group_t* group = &groups[layout->section_index];
    uint8_t* target =
        relocation_bytes + group->next_record++ * LOOM_NATIVE_ELF64LE_RELA_SIZE;
    const uint32_t relocation_type = options->relocation_mapper.map(
        options->relocation_mapper.user_data, fixup->relocation_kind);
    const uint64_t relocation_info =
        ((uint64_t)source_to_elf_indices[fixup->target_symbol_index] << 32) |
        relocation_type;
    iree_unaligned_store_le_u64(target + 0, layout->section_offset);
    iree_unaligned_store_le_u64(target + 8, relocation_info);
    iree_unaligned_store_le_u64(target + 16, (uint64_t)fixup->addend);
  }

  *out_groups = groups;
  *out_records = iree_make_const_byte_span(
      relocation_bytes, object->fixup_count * LOOM_NATIVE_ELF64LE_RELA_SIZE);
  *out_group_count = group_count;
  return iree_ok_status();
}

static iree_status_t loom_native_elf64le_build_relocatable(
    const loom_native_object_contribution_t* object,
    const loom_native_elf64le_relocatable_options_t* options,
    loom_native_elf64le_relocatable_build_t* out_build,
    iree_arena_allocator_t* arena) {
  *out_build = (loom_native_elf64le_relocatable_build_t){0};
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_RETURN_IF_ERROR(loom_native_assemble_section_contributions(
      object->sections, object->section_count, &assembly, arena));
  if (assembly.section_count > UINT16_MAX - 4u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native object section count exceeds ELF64 "
                            "section-table capacity");
  }

  uint32_t* source_to_elf_indices = NULL;
  iree_const_byte_span_t symbol_bytes = iree_const_byte_span_empty();
  iree_const_byte_span_t string_bytes = iree_const_byte_span_empty();
  uint32_t first_nonlocal_symbol = 1;
  IREE_RETURN_IF_ERROR(loom_native_elf64le_build_symbol_tables(
      object, &assembly, &source_to_elf_indices, &symbol_bytes, &string_bytes,
      &first_nonlocal_symbol, arena));

  loom_native_elf_relocation_group_t* relocation_groups = NULL;
  iree_const_byte_span_t relocation_bytes = iree_const_byte_span_empty();
  iree_host_size_t relocation_section_count = 0;
  IREE_RETURN_IF_ERROR(loom_native_elf64le_build_relocations(
      object, options, &assembly, source_to_elf_indices, &relocation_groups,
      &relocation_bytes, &relocation_section_count, arena));

  iree_host_size_t section_count = 0;
  if (!iree_host_size_checked_add(assembly.section_count,
                                  relocation_section_count, &section_count) ||
      !iree_host_size_checked_add(section_count, 2, &section_count) ||
      section_count > UINT16_MAX - 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native object section count exceeds ELF64 "
                            "section-table capacity");
  }
  loom_native_elf_section_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, section_count, sizeof(*sections), (void**)&sections));

  for (iree_host_size_t i = 0; i < assembly.section_count; ++i) {
    sections[i] = loom_native_elf_section_from_native(&assembly.sections[i]);
  }

  iree_host_size_t relocation_name_bytes = 0;
  for (iree_host_size_t i = 0; i < assembly.section_count; ++i) {
    const loom_native_elf_relocation_group_t* group = &relocation_groups[i];
    if (group->next_record == group->record_base) {
      continue;
    }
    if (!iree_host_size_checked_add(relocation_name_bytes,
                                    IREE_SV(".rela").size,
                                    &relocation_name_bytes) ||
        !iree_host_size_checked_add(relocation_name_bytes,
                                    assembly.sections[i].name.size,
                                    &relocation_name_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF64 relocation section names are too large");
    }
  }
  char* relocation_names = NULL;
  if (relocation_name_bytes != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, relocation_name_bytes,
                                             (void**)&relocation_names));
  }

  const iree_host_size_t symbol_section_index =
      assembly.section_count + relocation_section_count;
  const iree_host_size_t string_section_index = symbol_section_index + 1u;
  iree_host_size_t next_relocation_section_index = assembly.section_count;
  char* next_relocation_name = relocation_names;
  for (iree_host_size_t i = 0; i < assembly.section_count; ++i) {
    const loom_native_elf_relocation_group_t* group = &relocation_groups[i];
    const iree_host_size_t record_count =
        group->next_record - group->record_base;
    if (record_count == 0) {
      continue;
    }
    memcpy(next_relocation_name, IREE_SV(".rela").data, IREE_SV(".rela").size);
    memcpy(next_relocation_name + IREE_SV(".rela").size,
           assembly.sections[i].name.data, assembly.sections[i].name.size);
    const iree_host_size_t relocation_name_size =
        IREE_SV(".rela").size + assembly.sections[i].name.size;
    sections[next_relocation_section_index++] = (loom_native_elf_section_t){
        .name =
            iree_make_string_view(next_relocation_name, relocation_name_size),
        .type = LOOM_NATIVE_ELF_SECTION_TYPE_RELA,
        .flags = LOOM_NATIVE_ELF_SECTION_FLAG_INFO_LINK,
        .address = 0,
        .alignment = 8,
        .entry_size = LOOM_NATIVE_ELF64LE_RELA_SIZE,
        .link = (uint32_t)(symbol_section_index + 1u),
        .info = (uint32_t)(i + 1u),
        .contents = iree_make_const_byte_span(
            relocation_bytes.data +
                group->record_base * LOOM_NATIVE_ELF64LE_RELA_SIZE,
            record_count * LOOM_NATIVE_ELF64LE_RELA_SIZE),
        .zero_fill_length = 0,
    };
    next_relocation_name += relocation_name_size;
  }

  sections[symbol_section_index] = (loom_native_elf_section_t){
      .name = IREE_SV(".symtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
      .flags = 0,
      .address = 0,
      .alignment = 8,
      .entry_size = LOOM_NATIVE_ELF64LE_SYMBOL_SIZE,
      .link = (uint32_t)(string_section_index + 1u),
      .info = first_nonlocal_symbol,
      .contents = symbol_bytes,
      .zero_fill_length = 0,
  };
  sections[string_section_index] = (loom_native_elf_section_t){
      .name = IREE_SV(".strtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
      .flags = 0,
      .address = 0,
      .alignment = 1,
      .entry_size = 0,
      .link = 0,
      .info = 0,
      .contents = string_bytes,
      .zero_fill_length = 0,
  };

  *out_build = (loom_native_elf64le_relocatable_build_t){
      .sections = sections,
      .section_count = section_count,
  };
  return iree_ok_status();
}

iree_status_t loom_native_elf64le_write_relocatable_object(
    const loom_native_object_contribution_t* object,
    const loom_native_elf64le_relocatable_options_t* options,
    iree_io_stream_t* stream, iree_arena_allocator_t* scratch_arena) {
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  loom_native_elf64le_relocatable_build_t build = {0};
  iree_status_t status = loom_native_elf64le_build_relocatable(
      object, options, &build, scratch_arena);
  if (iree_status_is_ok(status)) {
    const loom_native_elf64le_file_t file = {
        .type = LOOM_NATIVE_ELF_FILE_TYPE_REL,
        .machine = options->machine,
        .os_abi = options->os_abi,
        .abi_version = options->abi_version,
        .flags = options->flags,
        .entry = 0,
        .sections = build.sections,
        .section_count = build.section_count,
        .segments = NULL,
        .segment_count = 0,
    };
    status = loom_native_elf64le_write_file(&file, stream, scratch_arena);
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
