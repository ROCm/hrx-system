// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"

#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

static iree_status_t loom_aie2p_tile_link_assign_addresses(
    const loom_aie2p_leaf_contribution_t* contribution,
    const loom_aie2p_tile_link_layout_t* layout,
    loom_native_section_contribution_assembly_t* assembly) {
  const loom_native_object_contribution_t* object = &contribution->object;
  const loom_aie2p_leaf_realization_t* realization = &contribution->realization;
  if (layout == NULL ||
      layout->storage_placement_count != realization->storage_domain_count ||
      (layout->storage_placement_count != 0 &&
       layout->storage_placements == NULL)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P tile layout must place every retained storage domain");
  }
  if (realization->entry_symbol_index >= object->symbol_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile entry symbol is invalid");
  }
  const loom_native_object_symbol_t* entry_symbol =
      &object->symbols[realization->entry_symbol_index];
  if (entry_symbol->section_contribution_index >=
      assembly->contribution_layout_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile entry section is invalid");
  }
  const iree_host_size_t code_section_index =
      assembly->contribution_layouts[entry_symbol->section_contribution_index]
          .section_index;
  if (code_section_index >= assembly->section_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile code section is invalid");
  }
  loom_native_elf_section_t* code_section =
      &assembly->sections[code_section_index];
  const uint64_t code_flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR;
  const uint64_t code_length =
      loom_native_elf_section_byte_length(code_section);
  if (code_section->type != LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS ||
      code_section->flags != code_flags ||
      code_length != realization->code.byte_length) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P tile code section does not match its leaf realization");
  }
  if (layout->program_byte_capacity == 0 ||
      code_length > layout->program_byte_capacity ||
      (uint64_t)layout->program_address + code_length >
          (uint64_t)UINT32_MAX + 1u) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P tile code exceeds program memory");
  }
  if (code_section->alignment != 0 &&
      layout->program_address % code_section->alignment != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P tile code address is misaligned");
  }
  code_section->address = layout->program_address;

  bool placement_seen[LOOM_STORAGE_SPACE_COUNT_] = {false};
  for (iree_host_size_t i = 0; i < layout->storage_placement_count; ++i) {
    const loom_storage_space_t storage_space =
        layout->storage_placements[i].storage_space;
    if (!loom_storage_space_is_valid(storage_space) ||
        placement_seen[storage_space]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P tile storage placements must have unique spaces");
    }
    placement_seen[storage_space] = true;
  }

  bool domain_seen[LOOM_STORAGE_SPACE_COUNT_] = {false};
  for (iree_host_size_t i = 0; i < realization->storage_domain_count; ++i) {
    const loom_aie2p_leaf_storage_domain_t* domain =
        &realization->storage_domains[i];
    if (!loom_storage_space_is_valid(domain->storage_space) ||
        domain_seen[domain->storage_space] ||
        domain->section_contribution_index >=
            assembly->contribution_layout_count ||
        domain->symbol_index >= object->symbol_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P leaf storage domain is invalid");
    }
    domain_seen[domain->storage_space] = true;

    const loom_aie2p_tile_storage_placement_t* placement = NULL;
    for (iree_host_size_t j = 0; j < layout->storage_placement_count; ++j) {
      if (layout->storage_placements[j].storage_space ==
          domain->storage_space) {
        placement = &layout->storage_placements[j];
        break;
      }
    }
    if (placement == NULL) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P tile storage domain has no placement");
    }

    const iree_host_size_t section_index =
        assembly->contribution_layouts[domain->section_contribution_index]
            .section_index;
    if (section_index >= assembly->section_count ||
        section_index == code_section_index) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P leaf storage section is invalid");
    }
    const loom_native_object_symbol_t* domain_symbol =
        &object->symbols[domain->symbol_index];
    if (domain_symbol->section_contribution_index !=
            domain->section_contribution_index ||
        domain_symbol->kind != LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P leaf storage symbol is invalid");
    }

    loom_native_elf_section_t* section = &assembly->sections[section_index];
    const loom_aie2p_leaf_storage_requirement_t* requirement =
        loom_aie2p_leaf_storage_requirement(realization, domain->storage_space);
    const uint64_t storage_flags =
        LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC | LOOM_NATIVE_ELF_SECTION_FLAG_WRITE;
    const uint64_t section_length =
        loom_native_elf_section_byte_length(section);
    const uint64_t section_end =
        (uint64_t)placement->load_address + section_length;
    if (section->type != LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS ||
        section->flags != storage_flags ||
        section_length != requirement->byte_length ||
        section->alignment < requirement->minimum_alignment) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P tile storage section does not match its realization");
    }
    if (section_end > (uint64_t)UINT32_MAX + 1u) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P tile storage exceeds ELF32 address space");
    }
    if (section->alignment != 0 &&
        placement->load_address % section->alignment != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P tile storage address is misaligned");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_aie2p_leaf_storage_domain_t* previous_domain =
          &realization->storage_domains[j];
      const iree_host_size_t previous_section_index =
          assembly
              ->contribution_layouts[previous_domain
                                         ->section_contribution_index]
              .section_index;
      if (previous_section_index == section_index) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P leaf storage domains must use distinct sections");
      }
      const loom_native_elf_section_t* previous_section =
          &assembly->sections[previous_section_index];
      const uint64_t previous_end =
          previous_section->address +
          loom_native_elf_section_byte_length(previous_section);
      if ((uint64_t)placement->load_address < previous_end &&
          previous_section->address < section_end) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P tile storage placements overlap");
      }
    }
    section->address = placement->load_address;
  }

  for (iree_host_size_t i = 0; i < assembly->section_count; ++i) {
    if (!iree_any_bit_set(assembly->sections[i].flags,
                          LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC) ||
        i == code_section_index) {
      continue;
    }
    bool has_domain = false;
    for (iree_host_size_t j = 0; j < realization->storage_domain_count; ++j) {
      const iree_host_size_t domain_section_index =
          assembly
              ->contribution_layouts[realization->storage_domains[j]
                                         .section_contribution_index]
              .section_index;
      has_domain |= domain_section_index == i;
    }
    if (!has_domain) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P tile has an allocated section without placement facts");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_tile_link(
    const loom_aie2p_leaf_contribution_t* contribution,
    const loom_aie2p_tile_link_layout_t* layout, iree_arena_allocator_t* arena,
    loom_aie2p_linked_tile_t* out_tile) {
  IREE_ASSERT_ARGUMENT(contribution);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_tile);
  *out_tile = (loom_aie2p_linked_tile_t){0};

  const loom_native_object_contribution_t* object = &contribution->object;
  const loom_aie2p_leaf_realization_t* realization = &contribution->realization;
  if (object->section_count == 0 || object->sections == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile link requires executable code");
  }
  if (object->symbol_count == 0 || object->symbols == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile link requires an entry symbol");
  }

  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_RETURN_IF_ERROR(loom_native_assemble_section_contributions(
      object->sections, object->section_count, &assembly, arena));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_tile_link_assign_addresses(contribution, layout, &assembly));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_object_apply_fixups(object, &assembly, arena));

  loom_native_object_symbol_layout_t* symbol_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, object->symbol_count,
                                                 sizeof(*symbol_layouts),
                                                 (void**)&symbol_layouts));
  IREE_RETURN_IF_ERROR(loom_native_object_resolve_symbol_layouts(
      object->symbols, object->symbol_count, assembly.contribution_layouts,
      assembly.contribution_layout_count, symbol_layouts));
  if (realization->entry_symbol_index >= object->symbol_count ||
      object->symbols[realization->entry_symbol_index].kind !=
          LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION ||
      symbol_layouts[realization->entry_symbol_index].section_index >=
          assembly.section_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P tile entry symbol is invalid");
  }
  const loom_native_object_symbol_layout_t* entry_layout =
      &symbol_layouts[realization->entry_symbol_index];
  uint64_t entry_address = 0;
  if (!iree_checked_add_u64(
          assembly.sections[entry_layout->section_index].address,
          entry_layout->section_offset, &entry_address) ||
      entry_address > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P tile entry address exceeds ELF32");
  }

  *out_tile = (loom_aie2p_linked_tile_t){
      .assembly = assembly,
      .symbol_layouts = symbol_layouts,
      .symbol_layout_count = object->symbol_count,
      .entry_section_index = entry_layout->section_index,
      .entry_address = (uint32_t)entry_address,
  };
  return iree_ok_status();
}
