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

void loom_native_object_elf_encode_symbol(
    const loom_native_object_symbol_t* symbol, uint32_t name_offset,
    uint16_t section_index, uint64_t value, uint8_t* record) {
  iree_unaligned_store_le_u32(record, name_offset);
  record[4] = (uint8_t)((kSymbolBindings[symbol->binding] << 4) |
                        kSymbolKinds[symbol->kind]);
  record[5] = (uint8_t)symbol->visibility;
  iree_unaligned_store_le_u16(record + 6, section_index);
  iree_unaligned_store_le_u64(record + 8, value);
  iree_unaligned_store_le_u64(record + 16, symbol->size);
}

typedef struct loom_native_elf_relocation_section_t {
  // Number of fixups whose contributions were joined into this section.
  iree_host_size_t count;
  // Next byte in the section's arena-owned RELA record array.
  uint8_t* cursor;
} loom_native_elf_relocation_section_t;

// Contributions retain their own symbol and section indices. Only this writer
// translates them into the assembled section and ELF local-first namespaces.
static iree_status_t loom_native_object_elf_relocations(
    const loom_native_object_contribution_t* contribution,
    const loom_native_section_contribution_assembly_t* assembly,
    const loom_native_elf_relocation_t* relocations,
    const uint32_t* symbol_indices, loom_native_elf_section_t* sections,
    iree_host_size_t* section_count, iree_arena_allocator_t* arena) {
  if (contribution->fixup_count == 0) {
    return iree_ok_status();
  }
  loom_native_elf_relocation_section_t* relocation_sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, assembly->section_count,
                                                 sizeof(*relocation_sections),
                                                 (void**)&relocation_sections));
  memset(relocation_sections, 0,
         assembly->section_count * sizeof(*relocation_sections));
  for (iree_host_size_t i = 0; i < contribution->fixup_count; ++i) {
    const loom_native_object_fixup_t* fixup = &contribution->fixups[i];
    const iree_host_size_t section_index =
        assembly->contribution_layouts[fixup->section_contribution_index]
            .section_index;
    ++relocation_sections[section_index].count;
  }
  enum { kRelocationSize = 24 };
  for (iree_host_size_t i = 0; i < assembly->section_count; ++i) {
    loom_native_elf_relocation_section_t* relocation = &relocation_sections[i];
    if (relocation->count == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, relocation->count, kRelocationSize,
                                  (void**)&relocation->cursor));
    const iree_string_view_t target_name = assembly->sections[i].name;
    char* name = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, 5 + target_name.size, (void**)&name));
    memcpy(name, ".rela", 5);
    memcpy(name + 5, target_name.data, target_name.size);
    sections[(*section_count)++] = (loom_native_elf_section_t){
        .name = iree_make_string_view(name, 5 + target_name.size),
        .type = LOOM_NATIVE_ELF_SECTION_TYPE_RELA,
        .alignment = 8,
        .entry_size = kRelocationSize,
        .link = (uint32_t)assembly->section_count + 1,
        .info = (uint32_t)i + 1,
        .contents = iree_make_const_byte_span(
            relocation->cursor, relocation->count * kRelocationSize),
    };
  }
  for (iree_host_size_t i = 0; i < contribution->fixup_count; ++i) {
    const loom_native_object_fixup_t* fixup = &contribution->fixups[i];
    const loom_native_section_contribution_layout_t* layout =
        &assembly->contribution_layouts[fixup->section_contribution_index];
    uint8_t* record = relocation_sections[layout->section_index].cursor;
    iree_unaligned_store_le_u64(record,
                                layout->section_offset + fixup->section_offset);
    iree_unaligned_store_le_u64(
        record + 8,
        ((uint64_t)symbol_indices[fixup->target_symbol_index] << 32) |
            relocations[fixup->relocation_kind].type);
    iree_unaligned_store_le_u64(record + 16, (uint64_t)fixup->addend);
    relocation_sections[layout->section_index].cursor += kRelocationSize;
  }
  return iree_ok_status();
}

iree_status_t loom_native_object_write_elf64le(
    const loom_native_object_contribution_t* contribution,
    loom_native_elf_machine_t machine,
    const loom_native_elf_relocation_t* relocations, iree_io_stream_t* stream,
    iree_arena_allocator_t* arena) {
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_RETURN_IF_ERROR(loom_native_assemble_section_contributions(
      contribution->sections, contribution->section_count, arena, &assembly));
  // Five additional entries are the null and section-name sections (inserted
  // by the serializer), symbol table, string table, and stack declaration.
  // Extended section indices are not part of this ELF writer's format.
  if (assembly.section_count > 0xff00u - 5u ||
      contribution->symbol_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF object exceeds section or symbol index range");
  }
  iree_host_size_t string_length = 1;
  uint32_t local_symbol_count = 0;
  bool names_fit = true;
  for (iree_host_size_t i = 0; i < contribution->symbol_count && names_fit;
       ++i) {
    names_fit = iree_host_size_checked_add(string_length,
                                           contribution->symbols[i].name.size,
                                           &string_length) &&
                iree_host_size_checked_add(string_length, 1, &string_length) &&
                string_length <= UINT32_MAX;
    local_symbol_count += contribution->symbols[i].binding ==
                          LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL;
  }
  if (!names_fit) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF object symbol names exceed string table");
  }
  enum { kSymbolSize = 24 };
  uint8_t* symbols = NULL;
  char* strings = NULL;
  uint32_t* symbol_indices = NULL;
  if (contribution->fixup_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, contribution->symbol_count, sizeof(*symbol_indices),
        (void**)&symbol_indices));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, contribution->symbol_count + 1, kSymbolSize, (void**)&symbols));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, string_length, (void**)&strings));
  memset(symbols, 0, kSymbolSize);
  strings[0] = 0;
  uint32_t name_offset = 1;
  uint32_t next_local = 1;
  uint32_t next_global = local_symbol_count + 1;
  for (iree_host_size_t i = 0; i < contribution->symbol_count; ++i) {
    const loom_native_object_symbol_t* source = &contribution->symbols[i];
    const uint32_t symbol_index =
        source->binding == LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL
            ? next_local++
            : next_global++;
    if (symbol_indices) {
      symbol_indices[i] = symbol_index;
    }
    uint8_t* symbol = symbols + (iree_host_size_t)symbol_index * kSymbolSize;
    uint16_t section_index = 0;
    uint64_t section_offset = 0;
    if (source->section_contribution_index != IREE_HOST_SIZE_MAX) {
      const loom_native_section_contribution_layout_t* layout =
          &assembly.contribution_layouts[source->section_contribution_index];
      section_index = (uint16_t)(layout->section_index + 1);
      section_offset = layout->section_offset + source->section_offset;
    }
    loom_native_object_elf_encode_symbol(source, name_offset, section_index,
                                         section_offset, symbol);
    memcpy(strings + name_offset, source->name.data, source->name.size);
    name_offset += (uint32_t)source->name.size;
    strings[name_offset++] = 0;
  }
  loom_native_elf_section_t* sections = NULL;
  const iree_host_size_t relocation_capacity =
      contribution->fixup_count ? assembly.section_count : 0;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assembly.section_count + 3 + relocation_capacity,
      sizeof(*sections), (void**)&sections));
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
  iree_host_size_t section_count = assembly.section_count + 3;
  IREE_RETURN_IF_ERROR(loom_native_object_elf_relocations(
      contribution, &assembly, relocations, symbol_indices, sections,
      &section_count, arena));
  if (section_count > 0xff00u - 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF relocations exceed section index range");
  }
  const loom_native_elf64le_file_t file = {
      .type = LOOM_NATIVE_ELF_FILE_TYPE_REL,
      .machine = machine,
      .sections = sections,
      .section_count = section_count,
  };
  loom_native_elf_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_native_elf64le_build_layout(&file, arena, &layout));
  return loom_native_elf64le_write_file(&file, &layout, stream);
}
