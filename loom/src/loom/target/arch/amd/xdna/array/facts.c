// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/array/facts.h"

#include "loom/target/arch/amd/xdna/array/facts_tables.inl"

const loom_xdna_array_family_t* loom_xdna_npu2_array_family(void) {
  return &kLoomXdnaNpu2ArrayFamily;
}

iree_status_t loom_xdna_array_tile_facts(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_xdna_tile_facts_t** out_facts) {
  IREE_ASSERT_ARGUMENT(family);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;
  if (coordinate.column >= family->column_count ||
      coordinate.row >= family->row_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "XDNA tile coordinate (%u, %u) is outside %ux%u family %s",
        coordinate.column, coordinate.row, family->column_count,
        family->row_count, family->key);
  }
  for (uint8_t i = 0; i < family->tile_count; ++i) {
    const loom_xdna_tile_facts_t* facts = &family->tiles[i];
    if (coordinate.row >= facts->first_row &&
        coordinate.row < facts->first_row + facts->row_count) {
      *out_facts = facts;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "XDNA family %s has incomplete row facts",
                          family->key);
}

iree_status_t loom_xdna_array_stream_port_range(
    const loom_xdna_array_family_t* family, loom_xdna_tile_kind_t tile_kind,
    loom_xdna_stream_direction_t direction, loom_xdna_stream_port_t port,
    const loom_xdna_stream_port_range_t** out_range) {
  IREE_ASSERT_ARGUMENT(family);
  IREE_ASSERT_ARGUMENT(out_range);
  *out_range = NULL;
  for (uint8_t i = 0; i < family->stream_port_count; ++i) {
    const loom_xdna_stream_port_range_t* range = &family->stream_ports[i];
    if (range->tile_kind == tile_kind && range->direction == direction &&
        range->port == port) {
      *out_range = range;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "XDNA tile kind %u has no stream port %u in direction %u", tile_kind,
      port, direction);
}

iree_status_t loom_xdna_array_register_address(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate, loom_xdna_register_module_t module,
    uint32_t register_offset, uint64_t* out_address) {
  IREE_ASSERT_ARGUMENT(family);
  IREE_ASSERT_ARGUMENT(out_address);
  *out_address = 0;
  const loom_xdna_tile_facts_t* tile = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(family, coordinate, &tile));
  if (module == 0 || module > LOOM_XDNA_REGISTER_MODULE_SHIM_PL ||
      (tile->register_module_bits & LOOM_XDNA_REGISTER_MODULE_BIT(module)) ==
          0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA register module %u is unavailable at tile (%u, %u)", module,
        coordinate.column, coordinate.row);
  }
  if (register_offset >= (UINT32_C(1) << family->row_shift)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA register offset 0x%08" PRIx32
                            " exceeds the tile aperture",
                            register_offset);
  }
  *out_address = ((uint64_t)coordinate.column << family->column_shift) |
                 ((uint64_t)coordinate.row << family->row_shift) |
                 register_offset;
  return iree_ok_status();
}

static bool loom_xdna_array_memory_range_contains(uint32_t base,
                                                  uint32_t capacity,
                                                  uint32_t address,
                                                  uint32_t byte_length,
                                                  uint32_t* out_offset) {
  const uint64_t end = (uint64_t)address + byte_length;
  const uint64_t range_end = (uint64_t)base + capacity;
  if (address < base || end > range_end) {
    return false;
  }
  *out_offset = address - base;
  return true;
}

iree_status_t loom_xdna_array_resolve_local_memory(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate, uint32_t address,
    uint32_t byte_length, loom_xdna_memory_placement_t* out_placement) {
  IREE_ASSERT_ARGUMENT(family);
  IREE_ASSERT_ARGUMENT(out_placement);
  *out_placement = (loom_xdna_memory_placement_t){0};
  if (byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA memory placement must not be empty");
  }
  const loom_xdna_tile_facts_t* tile = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(family, coordinate, &tile));
  uint32_t owner_offset = 0;
  if (!loom_xdna_array_memory_range_contains(
          tile->memory.local_base, tile->memory.local_capacity, address,
          byte_length, &owner_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA local-memory placement exceeds the tile");
  }
  *out_placement = (loom_xdna_memory_placement_t){
      .owner = coordinate,
      .owner_offset = owner_offset,
      .byte_length = byte_length,
      .available_capacity = tile->memory.local_capacity - owner_offset,
  };
  return iree_ok_status();
}

iree_status_t loom_xdna_array_resolve_load_memory(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t accessor, loom_xdna_memory_space_t memory_space,
    uint32_t address, uint32_t byte_length,
    loom_xdna_memory_placement_t* out_placement) {
  IREE_ASSERT_ARGUMENT(family);
  IREE_ASSERT_ARGUMENT(out_placement);
  *out_placement = (loom_xdna_memory_placement_t){0};
  if (byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA memory placement must not be empty");
  }
  const loom_xdna_tile_facts_t* accessor_tile = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_array_tile_facts(family, accessor, &accessor_tile));
  if (memory_space == LOOM_XDNA_MEMORY_SPACE_PROGRAM) {
    uint32_t owner_offset = 0;
    if (!loom_xdna_array_memory_range_contains(
            accessor_tile->memory.program_base,
            accessor_tile->memory.program_capacity, address, byte_length,
            &owner_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA program-memory placement exceeds the tile");
    }
    *out_placement = (loom_xdna_memory_placement_t){
        .owner = accessor,
        .owner_offset = owner_offset,
        .byte_length = byte_length,
        .available_capacity =
            accessor_tile->memory.program_capacity - owner_offset,
    };
    return iree_ok_status();
  }
  if (memory_space != LOOM_XDNA_MEMORY_SPACE_DATA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown XDNA memory space %u", memory_space);
  }

  for (uint8_t i = 0; i < accessor_tile->memory.window_count; ++i) {
    const loom_xdna_address_window_t* window =
        &family->address_windows[accessor_tile->memory.window_start + i];
    uint32_t owner_offset = 0;
    if (!loom_xdna_array_memory_range_contains(window->base, window->capacity,
                                               address, byte_length,
                                               &owner_offset)) {
      continue;
    }
    const int32_t owner_column =
        (int32_t)accessor.column + window->owner_column_delta;
    const int32_t owner_row = (int32_t)accessor.row + window->owner_row_delta;
    if (owner_column < 0 || owner_column >= family->column_count ||
        owner_row < 0 || owner_row >= family->row_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XDNA %s window at tile (%u, %u) has no physical owner", window->name,
          accessor.column, accessor.row);
    }
    const loom_xdna_tile_coordinate_t owner = {
        .column = (uint16_t)owner_column,
        .row = (uint16_t)owner_row,
    };
    const loom_xdna_tile_facts_t* owner_tile = NULL;
    IREE_RETURN_IF_ERROR(
        loom_xdna_array_tile_facts(family, owner, &owner_tile));
    if (owner_tile->kind != window->owner_kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA %s window at tile (%u, %u) targets incompatible tile kind %u",
          window->name, accessor.column, accessor.row, owner_tile->kind);
    }
    *out_placement = (loom_xdna_memory_placement_t){
        .owner = owner,
        .owner_offset = owner_offset,
        .byte_length = byte_length,
        .available_capacity = window->capacity - owner_offset,
    };
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "XDNA data placement at 0x%08" PRIx32
                          " does not fit a load window",
                          address);
}
