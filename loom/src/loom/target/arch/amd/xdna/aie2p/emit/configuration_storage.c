// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/configuration_storage.h"

iree_status_t loom_aie2p_configuration_storage_place(
    const loom_aie2p_leaf_contribution_t* contribution,
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate,
    loom_source_storage_packing_t* packing, iree_arena_allocator_t* arena,
    loom_aie2p_tile_link_layout_t* out_layout, bool* out_fits) {
  *out_fits = false;
  *out_layout = (loom_aie2p_tile_link_layout_t){0};
  const loom_xdna_tile_facts_t* tile =
      loom_xdna_array_tile_facts(family, coordinate);
  const loom_aie2p_leaf_realization_t* realization = &contribution->realization;
  loom_aie2p_tile_storage_placement_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, realization->storage_domain_count,
                                sizeof(*storage), (void**)&storage));
  for (iree_host_size_t i = 0; i < realization->storage_domain_count; ++i) {
    const loom_storage_space_t space =
        realization->storage_domains[i].storage_space;
    const loom_aie2p_leaf_storage_requirement_t* requirement =
        loom_aie2p_leaf_storage_requirement(realization, space);
    uint64_t offset = 0;
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
        packing, requirement->byte_length, requirement->minimum_alignment, NULL,
        0, &offset));
    if (loom_source_storage_packing_requirement(packing).byte_length >
        tile->memory.local_capacity) {
      return iree_ok_status();
    }
    storage[i] = (loom_aie2p_tile_storage_placement_t){
        .storage_space = space,
        .owner_offset = (uint32_t)offset,
        .load_address = tile->memory.local_load_base + (uint32_t)offset,
    };
  }
  loom_aie2p_tile_read_only_data_placement_t* read_only = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, realization->read_only_data_count,
                                sizeof(*read_only), (void**)&read_only));
  for (iree_host_size_t i = 0; i < realization->read_only_data_count; ++i) {
    const loom_aie2p_leaf_read_only_data_domain_t* domain =
        &realization->read_only_data[i];
    const loom_native_section_contribution_t* section =
        &contribution->object.sections[realization->read_only_data[i]
                                           .section_contribution_index];
    const uint64_t length = section->contents.data_length;
    loom_source_storage_packing_range_t* excluded = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, domain->bank_conflict_count,
                                  sizeof(*excluded), (void**)&excluded));
    iree_host_size_t excluded_count = 0;
    const uint64_t bank_length =
        tile->memory.local_capacity / tile->memory.bank_count;
    for (iree_host_size_t j = 0; j < domain->bank_conflict_count; ++j) {
      const uint32_t other = domain->bank_conflicts[j];
      if (other >= i || read_only[other].byte_length == 0) {
        continue;
      }
      const uint64_t first_bank = read_only[other].owner_offset / bank_length;
      const uint64_t last_bank =
          (read_only[other].owner_offset + read_only[other].byte_length - 1) /
          bank_length;
      excluded[excluded_count++] = (loom_source_storage_packing_range_t){
          first_bank * bank_length, (last_bank - first_bank + 1) * bank_length};
    }
    uint64_t offset = 0;
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
        packing, length, iree_max(section->contribution_alignment, 1), excluded,
        excluded_count, &offset));
    if (loom_source_storage_packing_requirement(packing).byte_length >
        tile->memory.local_capacity) {
      return iree_ok_status();
    }
    read_only[i] = (loom_aie2p_tile_read_only_data_placement_t){
        .owner_offset = (uint32_t)offset,
        .load_address = tile->memory.local_load_base + (uint32_t)offset,
        .byte_length = (uint32_t)length,
    };
  }
  *out_layout = (loom_aie2p_tile_link_layout_t){
      .program_address = tile->memory.program_base,
      .program_byte_capacity = tile->memory.program_capacity,
      .storage_placements = storage,
      .storage_placement_count = realization->storage_domain_count,
      .read_only_data_placements = read_only,
      .read_only_data_placement_count = realization->read_only_data_count,
  };
  *out_fits = true;
  return iree_ok_status();
}
