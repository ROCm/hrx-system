// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/worker_upload_plan.h"

#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"

typedef struct loom_aie2p_worker_row_t {
  // Workers in physical column order, including noncontiguous columns.
  const loom_aie2p_native_connection_endpoint_t* targets;
  // Number of workers accepting the shared transfer.
  iree_host_size_t count;
  // Shared instruction blocks and each worker's remaining writes.
  loom_aie2p_program_broadcast_t code;
} loom_aie2p_worker_row_t;

// Append-only storage for all waves. Groups in a wave index its own slice of
// the destination array.
typedef struct loom_aie2p_worker_wave_builder_t {
  // Next unused group across all disjoint sender sets.
  loom_aie2p_native_connection_t* groups;
  // Next unused slice of delivery endpoints.
  loom_aie2p_native_connection_endpoint_t* destinations;
  // Next payload in group order.
  loom_aie2p_worker_code_stream_t* streams;
} loom_aie2p_worker_wave_builder_t;

static void loom_aie2p_worker_wave_begin(
    const loom_aie2p_worker_wave_builder_t* builder, uint32_t credit,
    loom_aie2p_worker_wave_t* wave) {
  *wave = (loom_aie2p_worker_wave_t){
      .connections = {.groups = builder->groups,
                      .destinations = builder->destinations},
      .streams = builder->streams,
      .credit = credit,
  };
}

static void loom_aie2p_worker_wave_add(
    loom_aie2p_worker_wave_builder_t* builder, uint16_t column,
    const loom_aie2p_native_connection_endpoint_t* destinations,
    iree_host_size_t count, loom_aie2p_worker_code_stream_t stream,
    loom_aie2p_worker_wave_t* wave) {
  *builder->groups++ = (loom_aie2p_native_connection_t){
      .source = {{column, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0},
      .packet = UINT8_MAX,
      .destination_offset = wave->connections.destination_count,
      .destination_count = count,
  };
  memcpy(builder->destinations, destinations, count * sizeof(*destinations));
  builder->destinations += count;
  *builder->streams++ = stream;
  ++wave->connections.group_count;
  wave->connections.destination_count += count;
}

iree_status_t loom_aie2p_worker_upload_plan(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_xdna_tile_t* tiles, iree_host_size_t tile_count,
    iree_arena_allocator_t* arena, loom_aie2p_worker_upload_plan_t* out_plan) {
  *out_plan = (loom_aie2p_worker_upload_plan_t){0};
  if (tile_count < 2) {
    return iree_ok_status();
  }
  const loom_xdna_array_family_t* family = plan->routing.family;
  const loom_xdna_tile_facts_t* compute =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_COMPUTE);
  const iree_host_size_t places = loom_xdna_array_physical_tile_count(family);
  const loom_aie2p_xdna_tile_t** grid = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, places, sizeof(*grid), (void**)&grid));
  memset(grid, 0, places * sizeof(*grid));
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    const loom_xdna_tile_coordinate_t target = tiles[i].coordinate;
    const loom_xdna_tile_coordinate_t shim = {target.column, 0};
    const iree_host_size_t slot =
        loom_xdna_array_physical_tile_index(family, target);
    const loom_aie2p_execution_tile_t* source =
        &plan->tiles[loom_xdna_array_physical_tile_index(family, shim)];
    if (source->mm2s_count && source->descriptor_counts[0] &&
        plan->tiles[slot].lock_count) {
      grid[slot] = &tiles[i];
    }
  }
  loom_aie2p_worker_row_t* rows = NULL;
  loom_aie2p_native_connection_endpoint_t* targets = NULL;
  loom_aie2p_worker_code_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, compute->row_count,
                                                 sizeof(*rows), (void**)&rows));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*targets), (void**)&targets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, family->column_count, sizeof(*sections), (void**)&sections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, tile_count,
                                                 sizeof(*out_plan->workers),
                                                 (void**)&out_plan->workers));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, family->column_count,
                                                 sizeof(*out_plan->senders),
                                                 (void**)&out_plan->senders));
  memset(out_plan->workers, 0, tile_count * sizeof(*out_plan->workers));
  memset(out_plan->senders, 0,
         family->column_count * sizeof(*out_plan->senders));
  iree_host_size_t row_count = 0, worker_count = 0;
  for (uint16_t i = 0; i < compute->row_count; ++i) {
    const uint16_t row = compute->first_row + i;
    iree_host_size_t count = 0;
    for (uint16_t column = 0; column < family->column_count; ++column) {
      const loom_xdna_tile_coordinate_t place = {column, row};
      const loom_aie2p_xdna_tile_t* tile =
          grid[loom_xdna_array_physical_tile_index(family, place)];
      if (!tile) {
        continue;
      }
      const loom_aie2p_linked_tile_t* linked = tile->linked_tile;
      const iree_host_size_t entry = linked->entry_section_index;
      sections[count] = (loom_aie2p_worker_code_t){
          compute->memory.program_load_base +
              linked->section_placements[entry].owner_offset,
          linked->assembly.sections[entry].contents};
      targets[count++] = (loom_aie2p_native_connection_endpoint_t){
          place, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
    }
    if (count < 2) {
      continue;
    }
    loom_aie2p_worker_row_t* selected = &rows[row_count];
    IREE_RETURN_IF_ERROR(loom_aie2p_program_broadcast_build(
        sections, count, arena, &selected->code));
    if (!selected->code.shared.word_count) {
      continue;
    }
    selected->targets = targets;
    selected->count = count;
    for (iree_host_size_t j = 0; j < count; ++j) {
      const loom_xdna_tile_coordinate_t target = targets[j].coordinate;
      const loom_aie2p_xdna_tile_t* tile =
          grid[loom_xdna_array_physical_tile_index(family, target)];
      out_plan->workers[tile - tiles] = true;
      out_plan->senders[target.column] = true;
      out_plan->inline_size +=
          16 + iree_host_align(sections[j].contents.data_length, 4);
    }
    ++row_count;
    worker_count += count;
    targets += count;
  }
  if (!row_count) {
    return iree_ok_status();
  }

  uint16_t* columns = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, family->column_count, sizeof(*columns), (void**)&columns));
  iree_host_size_t column_count = 0;
  for (uint16_t i = 0; i < family->column_count; ++i) {
    if (out_plan->senders[i]) {
      columns[column_count++] = i;
    }
  }
  memset(out_plan->senders, 0,
         family->column_count * sizeof(*out_plan->senders));
  loom_aie2p_worker_wave_builder_t builder;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, row_count + worker_count, sizeof(*builder.groups),
      (void**)&builder.groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, 2 * worker_count, sizeof(*builder.destinations),
      (void**)&builder.destinations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, row_count + worker_count, sizeof(*builder.streams),
      (void**)&builder.streams));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, 2 * row_count,
                                                 sizeof(*out_plan->waves),
                                                 (void**)&out_plan->waves));
  iree_host_size_t wave_count = 0;
  for (iree_host_size_t i = 0; i < row_count;) {
    loom_aie2p_worker_wave_t* wave = &out_plan->waves[wave_count++];
    loom_aie2p_worker_wave_begin(&builder, 1, wave);
    for (iree_host_size_t j = 0; j < column_count && i < row_count; ++j, ++i) {
      const loom_aie2p_worker_row_t* row = &rows[i];
      loom_aie2p_worker_wave_add(&builder, columns[j], row->targets, row->count,
                                 row->code.shared, wave);
    }
  }
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    const loom_aie2p_worker_row_t* row = &rows[i];
    loom_aie2p_worker_wave_t* wave = &out_plan->waves[wave_count];
    loom_aie2p_worker_wave_begin(&builder, 2, wave);
    for (iree_host_size_t j = 0; j < row->count; ++j) {
      if (row->code.individual[j].word_count) {
        loom_aie2p_worker_wave_add(&builder, row->targets[j].coordinate.column,
                                   &row->targets[j], 1, row->code.individual[j],
                                   wave);
      }
    }
    if (wave->connections.group_count) {
      ++wave_count;
    }
  }
  out_plan->record_capacity = 2 * column_count + worker_count;
  for (iree_host_size_t i = 0; i < wave_count; ++i) {
    loom_aie2p_worker_wave_t* wave = &out_plan->waves[i];
    loom_aie2p_native_switch_t* switches = NULL;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_ingress_circuits(
        family, &wave->connections, arena, &switches));
    if (!switches) {
      return iree_ok_status();
    }
    for (const loom_aie2p_native_switch_t* state = switches; state;
         state = state->next) {
      // Each completed route configures the stream switch at every traversed
      // tile. At a shim its DMA mux connects to that same tile's switch.
      if (!plan->routing.tile_route_counts[loom_xdna_array_physical_tile_index(
              family, state->coordinate)]) {
        return iree_ok_status();
      }
      out_plan->record_capacity += 2;
    }
    for (iree_host_size_t j = 0; j < wave->connections.group_count; ++j) {
      out_plan->senders[wave->connections.groups[j].source.coordinate.column] =
          true;
    }
    wave->switches = switches;
    out_plan->transfer_count += wave->connections.group_count;
    out_plan->record_capacity +=
        3 * wave->connections.group_count + wave->connections.destination_count;
  }
  out_plan->wave_count = wave_count;
  return iree_ok_status();
}
