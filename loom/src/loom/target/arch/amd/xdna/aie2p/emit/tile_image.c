// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/tile_image.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"
#include "loom/target/emit/native/elf.h"

enum {
  LOOM_AIE2P_ELF32_SYMBOL_SIZE = 16,
};

static void loom_aie2p_tile_image_store_u16(uint8_t* target, uint16_t value) {
  target[0] = (uint8_t)value;
  target[1] = (uint8_t)(value >> 8);
}

static void loom_aie2p_tile_image_store_u32(uint8_t* target, uint32_t value) {
  target[0] = (uint8_t)value;
  target[1] = (uint8_t)(value >> 8);
  target[2] = (uint8_t)(value >> 16);
  target[3] = (uint8_t)(value >> 24);
}

static iree_status_t loom_aie2p_tile_image_measure_symbol_names(
    const loom_native_object_contribution_t* object,
    iree_host_size_t* out_string_table_size) {
  iree_host_size_t string_table_size = 1;
  for (iree_host_size_t i = 0; i < object->symbol_count; ++i) {
    iree_host_size_t symbol_name_size = 0;
    if (!iree_host_size_checked_add(object->symbols[i].name.size, 1,
                                    &symbol_name_size) ||
        !iree_host_size_checked_add(string_table_size, symbol_name_size,
                                    &string_table_size) ||
        string_table_size > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P tile symbol string table is too large");
    }
  }
  *out_string_table_size = string_table_size;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_tile_image_build_symbol_order(
    const loom_native_object_contribution_t* object,
    iree_arena_allocator_t* arena, iree_host_size_t** out_symbol_order,
    iree_host_size_t* out_local_symbol_count) {
  *out_symbol_order = NULL;
  *out_local_symbol_count = 0;
  iree_host_size_t* symbol_order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, object->symbol_count,
                                                 sizeof(*symbol_order),
                                                 (void**)&symbol_order));

  iree_host_size_t next_index = 0;
  for (iree_host_size_t i = 0; i < object->symbol_count; ++i) {
    if (object->symbols[i].binding == LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL) {
      symbol_order[next_index++] = i;
    }
  }
  *out_local_symbol_count = next_index;
  for (iree_host_size_t i = 0; i < object->symbol_count; ++i) {
    if (object->symbols[i].binding != LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL) {
      symbol_order[next_index++] = i;
    }
  }
  IREE_ASSERT(next_index == object->symbol_count);
  *out_symbol_order = symbol_order;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_tile_image_elf_binding(uint32_t binding,
                                                       uint8_t* out_binding) {
  switch (binding) {
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL:
      *out_binding = 0;
      return iree_ok_status();
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL:
      *out_binding = 1;
      return iree_ok_status();
    case LOOM_NATIVE_OBJECT_SYMBOL_BINDING_WEAK:
      *out_binding = 2;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P tile symbol binding is invalid");
  }
}

static iree_status_t loom_aie2p_tile_image_elf_symbol_kind(uint32_t kind,
                                                           uint8_t* out_kind) {
  switch (kind) {
    case LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION:
      *out_kind = 2;
      return iree_ok_status();
    case LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA:
      *out_kind = 1;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P tile symbol kind is invalid");
  }
}

static iree_status_t loom_aie2p_tile_image_build_symbol_tables(
    const loom_native_object_contribution_t* object,
    const loom_native_section_contribution_assembly_t* assembly,
    uint32_t entry_symbol_index, iree_arena_allocator_t* arena,
    iree_const_byte_span_t* out_string_table,
    iree_const_byte_span_t* out_symbol_table,
    iree_host_size_t* out_local_symbol_count, uint32_t* out_entry_address) {
  *out_string_table = iree_const_byte_span_empty();
  *out_symbol_table = iree_const_byte_span_empty();
  *out_local_symbol_count = 0;
  *out_entry_address = 0;

  loom_native_object_symbol_layout_t* symbol_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, object->symbol_count,
                                                 sizeof(*symbol_layouts),
                                                 (void**)&symbol_layouts));
  IREE_RETURN_IF_ERROR(loom_native_object_resolve_symbol_layouts(
      object->symbols, object->symbol_count, assembly->contribution_layouts,
      assembly->contribution_layout_count, symbol_layouts));
  if (entry_symbol_index >= object->symbol_count ||
      object->symbols[entry_symbol_index].kind !=
          LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION ||
      symbol_layouts[entry_symbol_index].section_offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile entry symbol is invalid");
  }
  *out_entry_address =
      (uint32_t)symbol_layouts[entry_symbol_index].section_offset;

  iree_host_size_t* symbol_order = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_tile_image_build_symbol_order(
      object, arena, &symbol_order, out_local_symbol_count));

  iree_host_size_t string_table_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_tile_image_measure_symbol_names(object, &string_table_size));
  uint8_t* string_table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, string_table_size, (void**)&string_table));
  memset(string_table, 0, string_table_size);

  iree_host_size_t symbol_record_count = 0;
  if (!iree_host_size_checked_add(object->symbol_count, 1,
                                  &symbol_record_count) ||
      symbol_record_count > IREE_HOST_SIZE_MAX / LOOM_AIE2P_ELF32_SYMBOL_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P tile symbol table is too large");
  }
  const iree_host_size_t symbol_table_size =
      symbol_record_count * LOOM_AIE2P_ELF32_SYMBOL_SIZE;
  uint8_t* symbol_table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, symbol_table_size, (void**)&symbol_table));
  memset(symbol_table, 0, symbol_table_size);

  iree_host_size_t string_offset = 1;
  for (iree_host_size_t i = 0; i < object->symbol_count; ++i) {
    const iree_host_size_t source_index = symbol_order[i];
    const loom_native_object_symbol_t* symbol = &object->symbols[source_index];
    const loom_native_object_symbol_layout_t* layout =
        &symbol_layouts[source_index];
    if (layout->section_index >= assembly->section_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P tile symbol section is invalid");
    }
    const loom_native_elf_section_t* section =
        &assembly->sections[layout->section_index];
    uint64_t symbol_end = 0;
    if (!iree_checked_add_u64(layout->section_offset, symbol->size,
                              &symbol_end) ||
        symbol_end > section->contents.data_length ||
        layout->section_offset > UINT32_MAX || symbol->size > UINT32_MAX ||
        layout->section_index >= UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P tile symbol %" PRIhsz
                              " lies outside its ELF32 section",
                              source_index);
    }
    if (string_offset > UINT32_MAX ||
        symbol->name.size >= string_table_size - string_offset) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "AIE2P tile symbol string layout overflow");
    }

    const uint32_t name_offset = (uint32_t)string_offset;
    memcpy(string_table + string_offset, symbol->name.data, symbol->name.size);
    string_offset += symbol->name.size + 1;

    uint8_t elf_binding = 0;
    uint8_t elf_kind = 0;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_tile_image_elf_binding(symbol->binding, &elf_binding));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_tile_image_elf_symbol_kind(symbol->kind, &elf_kind));
    if (symbol->visibility > UINT8_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P tile symbol visibility is too large");
    }

    uint8_t* record = symbol_table + (i + 1) * LOOM_AIE2P_ELF32_SYMBOL_SIZE;
    loom_aie2p_tile_image_store_u32(record + 0, name_offset);
    loom_aie2p_tile_image_store_u32(record + 4,
                                    (uint32_t)layout->section_offset);
    loom_aie2p_tile_image_store_u32(record + 8, (uint32_t)symbol->size);
    record[12] = (uint8_t)((elf_binding << 4) | elf_kind);
    record[13] = (uint8_t)symbol->visibility;
    loom_aie2p_tile_image_store_u16(record + 14,
                                    (uint16_t)(layout->section_index + 1));
  }
  IREE_ASSERT(string_offset == string_table_size);

  *out_string_table =
      iree_make_const_byte_span(string_table, string_table_size);
  *out_symbol_table =
      iree_make_const_byte_span(symbol_table, symbol_table_size);
  return iree_ok_status();
}

iree_status_t loom_aie2p_tile_image_write(
    const loom_aie2p_leaf_contribution_t* contribution,
    iree_io_stream_t* stream, iree_arena_allocator_t* scratch_arena) {
  const loom_native_object_contribution_t* object = &contribution->object;
  const loom_aie2p_leaf_realization_t* realization = &contribution->realization;
  if (object->section_count == 0 || object->sections == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile image requires executable code");
  }
  if (object->symbol_count == 0 || object->symbols == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile image requires an entry symbol");
  }
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_RETURN_IF_ERROR(loom_native_assemble_section_contributions(
      object->sections, object->section_count, &assembly, scratch_arena));
  if (assembly.section_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P leaf tile image requires one assembled code section");
  }
  loom_native_elf_section_t* code_section = &assembly.sections[0];
  const uint64_t required_flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                                  LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR;
  if (code_section->type != LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS ||
      code_section->flags != required_flags) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P tile code must be an allocated executable PROGBITS section");
  }
  if (code_section->contents.data_length >
      LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P tile code requires %" PRIhsz
                            " bytes but program memory has %u bytes",
                            code_section->contents.data_length,
                            LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE);
  }
  code_section->address = 0;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_object_apply_fixups(object, &assembly, scratch_arena));

  iree_const_byte_span_t string_table = iree_const_byte_span_empty();
  iree_const_byte_span_t symbol_table = iree_const_byte_span_empty();
  iree_host_size_t local_symbol_count = 0;
  uint32_t entry_address = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_tile_image_build_symbol_tables(
      object, &assembly, realization->entry_symbol_index, scratch_arena,
      &string_table, &symbol_table, &local_symbol_count, &entry_address));
  if (local_symbol_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P tile has too many local symbols");
  }

  loom_native_elf_section_t sections[3] = {
      *code_section,
      {
          .name = IREE_SV(".strtab"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
          .alignment = 1,
          .contents = string_table,
      },
      {
          .name = IREE_SV(".symtab"),
          .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
          .alignment = 4,
          .entry_size = LOOM_AIE2P_ELF32_SYMBOL_SIZE,
          .link = 2,
          .info = (uint32_t)local_symbol_count + 1,
          .contents = symbol_table,
      },
  };
  const loom_native_elf_segment_t load_segment = {
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE,
      .first_section = 0,
      .section_count = 1,
      .alignment = 16,
  };
  const loom_native_elf32le_file_t file = {
      .type = LOOM_NATIVE_ELF_FILE_TYPE_EXEC,
      .machine = realization->elf_machine,
      .os_abi = LOOM_NATIVE_ELF_OS_ABI_NONE,
      .abi_version = LOOM_NATIVE_ELF_ABI_VERSION_NONE,
      .flags = realization->elf_flags,
      .entry = entry_address,
      .sections = sections,
      .section_count = IREE_ARRAYSIZE(sections),
      .segments = &load_segment,
      .segment_count = 1,
  };
  return loom_native_elf32le_write_file(&file, stream, scratch_arena);
}
