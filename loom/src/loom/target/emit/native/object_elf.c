// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/object_elf.h"

#include "loom/target/emit/native/elf_sections.h"

static const uint8_t kSymbolBindings[] = {
    [LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL] = 0,
    [LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL] = 1,
    [LOOM_NATIVE_OBJECT_SYMBOL_BINDING_WEAK] = 2,
};

static const uint8_t kSymbolKinds[] = {
    [LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION] = 2,
    [LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA] = 1,
};

iree_status_t loom_native_object_write_elf64le(
    const loom_native_object_contribution_t* contribution,
    loom_native_elf_machine_t machine, iree_io_stream_t* stream,
    iree_arena_allocator_t* arena) {
  if (contribution->fixup_count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "ELF object relocations require a target mapping");
  }
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_RETURN_IF_ERROR(loom_native_assemble_section_contributions(
      contribution->sections, contribution->section_count, &assembly, arena));
  // Five additional entries are the null and section-name sections (inserted
  // by the serializer), symbol table, string table, and stack declaration.
  // Extended section indices are not part of this ELF writer's format.
  if (assembly.section_count > 0xff00u - 5u ||
      contribution->symbol_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF object exceeds section or symbol index range");
  }
  iree_host_size_t string_length = 1;
  bool names_fit = true;
  for (iree_host_size_t i = 0; i < contribution->symbol_count && names_fit;
       ++i) {
    names_fit = iree_host_size_checked_add(string_length,
                                           contribution->symbols[i].name.size,
                                           &string_length) &&
                iree_host_size_checked_add(string_length, 1, &string_length) &&
                string_length <= UINT32_MAX;
  }
  if (!names_fit) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF object symbol names exceed string table");
  }
  enum { kSymbolSize = 24 };
  uint8_t* symbols = NULL;
  char* strings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, contribution->symbol_count + 1, kSymbolSize, (void**)&symbols));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, string_length, (void**)&strings));
  memset(symbols, 0, kSymbolSize);
  strings[0] = 0;
  uint32_t name_offset = 1;
  uint32_t local_symbol_count = 0;
  for (iree_host_size_t i = 0; i < contribution->symbol_count; ++i) {
    const loom_native_object_symbol_t* source = &contribution->symbols[i];
    const loom_native_section_contribution_layout_t* layout =
        &assembly.contribution_layouts[source->section_contribution_index];
    uint8_t* symbol = symbols + (i + 1) * kSymbolSize;
    iree_unaligned_store_le_u32(symbol, name_offset);
    symbol[4] = (uint8_t)((kSymbolBindings[source->binding] << 4) |
                          kSymbolKinds[source->kind]);
    symbol[5] = (uint8_t)source->visibility;
    iree_unaligned_store_le_u16(symbol + 6,
                                (uint16_t)(layout->section_index + 1));
    iree_unaligned_store_le_u64(
        symbol + 8, layout->section_offset + source->section_offset);
    iree_unaligned_store_le_u64(symbol + 16, source->size);
    local_symbol_count +=
        source->binding == LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL;
    memcpy(strings + name_offset, source->name.data, source->name.size);
    name_offset += (uint32_t)source->name.size;
    strings[name_offset++] = 0;
  }
  loom_native_elf_section_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assembly.section_count + 3, sizeof(*sections), (void**)&sections));
  for (iree_host_size_t i = 0; i < assembly.section_count; ++i) {
    sections[i] = loom_native_elf_section_from_native(&assembly.sections[i]);
  }
  sections[assembly.section_count] = (loom_native_elf_section_t){
      .name = IREE_SV(".symtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
      .alignment = 8,
      .entry_size = kSymbolSize,
      .link = (uint32_t)assembly.section_count + 2,
      .info = local_symbol_count + 1,
      .contents = iree_make_const_byte_span(
          symbols, (contribution->symbol_count + 1) * kSymbolSize),
  };
  sections[assembly.section_count + 1] = (loom_native_elf_section_t){
      .name = IREE_SV(".strtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
      .alignment = 1,
      .contents = iree_make_const_byte_span(strings, string_length),
  };
  sections[assembly.section_count + 2] = (loom_native_elf_section_t){
      .name = IREE_SV(".note.GNU-stack"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .alignment = 1,
  };
  const loom_native_elf64le_file_t file = {
      .type = LOOM_NATIVE_ELF_FILE_TYPE_REL,
      .machine = machine,
      .sections = sections,
      .section_count = assembly.section_count + 3,
  };
  return loom_native_elf64le_write_file(&file, stream, arena);
}
