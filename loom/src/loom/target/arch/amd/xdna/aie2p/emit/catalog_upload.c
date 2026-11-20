// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/catalog_upload.h"

#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"

typedef struct loom_aie2p_catalog_column_t {
  // First byte in the tile-local SRAM bank.
  uint32_t begin;
  // Exclusive end of the placed catalogs in this bank.
  uint32_t end;
  // Sum of placed bytes; gaps in the envelope retain their original contents.
  uint32_t contents_length;
  // Corresponding aligned beginning in the immutable source allocation.
  uint32_t source_offset;
} loom_aie2p_catalog_column_t;

// Only the shim and its adjacent SRAM switch participate in a cold transfer.
// Each route contributes its shim mux, shim switch, and SRAM switch.
static iree_status_t loom_aie2p_catalog_routes(
    const loom_xdna_array_family_t* family, const bool* columns,
    iree_host_size_t count, iree_arena_allocator_t* arena,
    loom_aie2p_native_switch_t** out_switches) {
  loom_aie2p_array_route_builder_t routing;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_builder_initialize(
      family, count * 3, arena, &routing));
  loom_aie2p_native_connection_t* connections = NULL;
  loom_aie2p_native_connection_endpoint_t* destinations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*connections), (void**)&connections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*destinations), (void**)&destinations));
  const uint16_t memory_row =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_MEMORY)
          ->first_row;
  iree_host_size_t index = 0;
  for (uint16_t column = 0; column < family->column_count; ++column) {
    if (!columns[column]) {
      continue;
    }
    const loom_xdna_tile_coordinate_t shim = {column, 0};
    const loom_xdna_tile_coordinate_t memory = {column, memory_row};
    loom_aie2p_array_route_tree_t tree = {0};
    const bool routed = loom_aie2p_array_route_ingress(
        &routing, (uint32_t)index, &tree, shim, 0, memory, 0);
    IREE_ASSERT(routed);
    (void)routed;
    connections[index] = (loom_aie2p_native_connection_t){
        .source = {shim, LOOM_XDNA_STREAM_PORT_DMA, 0},
        .packet = UINT8_MAX,
        .destination_offset = index,
        .destination_count = 1,
    };
    destinations[index++] = (loom_aie2p_native_connection_endpoint_t){
        memory, LOOM_XDNA_STREAM_PORT_DMA, 0};
  }
  loom_aie2p_native_route_t* routes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, routing.route_count, sizeof(*routes), (void**)&routes));
  for (iree_host_size_t i = 0; i < routing.route_count; ++i) {
    routes[i] = (loom_aie2p_native_route_t){.edge = &routing.routes[i],
                                            .packet = UINT8_MAX,
                                            .control_source = UINT32_MAX};
  }
  const loom_aie2p_native_switch_input_t input = {.routing = &routing,
                                                  .routes = routes};
  iree_string_view_t requirement;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_switches_build(
      &input, arena, out_switches, &requirement));
  IREE_ASSERT(iree_string_view_is_empty(requirement));
  const loom_aie2p_native_connections_t intended = {
      .groups = connections,
      .group_count = count,
      .destinations = destinations,
      .destination_count = count,
  };
  return loom_aie2p_native_connections_verify(&routing, *out_switches,
                                              &intended, arena);
}

static loom_aie2p_program_record_t loom_aie2p_catalog_write(uint32_t address,
                                                            uint32_t value) {
  return (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
      .value.register_write32 = {address, value}};
}

static loom_aie2p_program_record_t loom_aie2p_catalog_block(
    uint32_t address, const uint32_t* words, iree_host_size_t word_count) {
  return (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32,
      .value.register_block_write32 = {address, words, word_count}};
}

static loom_aie2p_program_record_t loom_aie2p_catalog_wait(uint32_t address,
                                                           uint32_t mask,
                                                           uint32_t value) {
  return (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32,
      .value.register_mask_wait32 = {address, mask, value}};
}

// All temporary resources use engine, descriptor, and hardware semaphore zero.
static uint32_t loom_aie2p_catalog_address(
    const loom_xdna_array_family_t* family, loom_xdna_tile_coordinate_t tile,
    loom_xdna_register_field_id_t field) {
  const uint16_t index = 0;
  return (uint32_t)loom_xdna_register_field_address(family, field, tile,
                                                    &index);
}

static iree_host_size_t loom_aie2p_catalog_select_columns(
    const loom_aie2p_execution_plan_t* plan, const loom_xdna_tile_facts_t* shim,
    const loom_xdna_tile_facts_t* memory,
    const loom_aie2p_program_catalog_t* catalogs,
    iree_host_size_t catalog_count, loom_aie2p_catalog_column_t* ranges,
    bool* columns, uint32_t* out_size) {
  const loom_xdna_array_family_t* family = plan->routing.family;
  memset(ranges, 0, family->column_count * sizeof(*ranges));
  for (iree_host_size_t i = 0; i < catalog_count; ++i) {
    const loom_aie2p_program_catalog_t* catalog = &catalogs[i];
    if (!catalog->descriptor_count) {
      continue;
    }
    loom_aie2p_catalog_column_t* range = &ranges[catalog->memory.column];
    const uint32_t bytes = (uint32_t)catalog->contents.word_count * 4;
    const uint32_t begin = range->contents_length
                               ? iree_min(range->begin, catalog->offset)
                               : catalog->offset;
    const uint32_t end = iree_max(range->end, catalog->offset + bytes);
    range->begin = begin;
    range->end = end;
    range->contents_length += bytes;
  }
  const uint64_t maximum_transfer =
      ((uint64_t)memory->dma.maximum_encoded_transfer_length +
       memory->dma.transfer_length_offset) *
      memory->dma.transfer_length_granularity;
  iree_host_size_t count = 0;
  uint32_t data_size = 0;
  for (uint16_t column = 0; column < family->column_count; ++column) {
    loom_aie2p_catalog_column_t* range = &ranges[column];
    const loom_xdna_tile_coordinate_t source = {column, shim->first_row};
    const loom_xdna_tile_coordinate_t target = {column, memory->first_row};
    const loom_aie2p_execution_tile_t* sender =
        &plan->tiles[loom_xdna_array_physical_tile_index(family, source)];
    const loom_aie2p_execution_tile_t* receiver =
        &plan->tiles[loom_xdna_array_physical_tile_index(family, target)];
    // These app-owned switches, queue controls, descriptor banks, and lock
    // are initialized after the upload. A sparse pipeline keeps its ownership
    // boundary. Inline writes also avoid arming two DMA engines for short data.
    columns[column] = range->contents_length >= 8192 &&
                      range->contents_length <= maximum_transfer &&
                      range->contents_length == range->end - range->begin &&
                      sender->mm2s_count && sender->descriptor_counts[0] &&
                      receiver->s2mm_count && receiver->descriptor_counts[0] &&
                      receiver->lock_count;
    if (!columns[column]) {
      continue;
    }
    range->source_offset = (uint32_t)iree_host_align(
        data_size, LOOM_AIE2P_PROGRAM_UPLOAD_ALIGNMENT);
    data_size = range->source_offset + range->contents_length;
    ++count;
  }
  *out_size = data_size;
  return count;
}

static iree_status_t loom_aie2p_catalog_pack_contents(
    const loom_aie2p_program_catalog_t* catalogs,
    iree_host_size_t catalog_count, const loom_aie2p_catalog_column_t* ranges,
    const bool* columns, uint16_t column_count, uint32_t data_size,
    iree_arena_allocator_t* arena, iree_const_byte_span_t* out_contents) {
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, data_size, (void**)&data));
  // Each independent source begins on a cache line. Initialize only the small
  // alignment gaps; placed spans cover every other byte.
  uint32_t previous_end = 0;
  for (uint16_t column = 0; column < column_count; ++column) {
    if (!columns[column]) {
      continue;
    }
    memset(data + previous_end, 0, ranges[column].source_offset - previous_end);
    previous_end =
        ranges[column].source_offset + ranges[column].contents_length;
  }
  for (iree_host_size_t i = 0; i < catalog_count; ++i) {
    const loom_aie2p_program_catalog_t* catalog = &catalogs[i];
    if (!catalog->descriptor_count || !columns[catalog->memory.column]) {
      continue;
    }
    const loom_aie2p_catalog_column_t* range = &ranges[catalog->memory.column];
    uint8_t* output =
        data + range->source_offset + (catalog->offset - range->begin);
    for (iree_host_size_t w = 0; w < catalog->contents.word_count; ++w) {
      iree_unaligned_store_le_u32(output + w * 4, catalog->contents.words[w]);
    }
  }
  *out_contents = iree_make_const_byte_span(data, data_size);
  return iree_ok_status();
}

iree_status_t loom_aie2p_catalog_upload_build(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_program_catalog_t* catalogs,
    iree_host_size_t catalog_count, iree_host_size_t first_record,
    iree_arena_allocator_t* arena, loom_aie2p_catalog_upload_t** out_upload) {
  *out_upload = NULL;
  const loom_xdna_array_family_t* family = plan->routing.family;
  const loom_xdna_tile_facts_t* shim =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  const loom_xdna_tile_facts_t* memory =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_MEMORY);
  loom_aie2p_catalog_column_t* ranges = NULL;
  bool* columns = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, family->column_count, sizeof(*ranges), (void**)&ranges));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, family->column_count, sizeof(*columns), (void**)&columns));
  uint32_t data_size = 0;
  const iree_host_size_t count = loom_aie2p_catalog_select_columns(
      plan, shim, memory, catalogs, catalog_count, ranges, columns, &data_size);
  if (!count) {
    return iree_ok_status();
  }
  iree_const_byte_span_t contents;
  IREE_RETURN_IF_ERROR(loom_aie2p_catalog_pack_contents(
      catalogs, catalog_count, ranges, columns, family->column_count, data_size,
      arena, &contents));
  loom_aie2p_native_switch_t* switches = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_catalog_routes(family, columns, count, arena, &switches));
  // Four switch-bank writes, seven descriptor setup records, two starts,
  // and four retirement records per active column.
  enum {
    kSetupCount = 7,
    kRetireCount = 4,
    kRecordsPerColumn = 4 + kSetupCount + 2 + kRetireCount
  };
  loom_aie2p_program_record_t* records = NULL;
  loom_aie2p_program_upload_patch_t* patches = NULL;
  uint32_t (*descriptors)[2][8] = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count * kRecordsPerColumn, sizeof(*records), (void**)&records));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*patches),
                                                 (void**)&patches));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*descriptors), (void**)&descriptors));
  memset(descriptors, 0, count * sizeof(*descriptors));
  iree_host_size_t position = 0;
  for (const loom_aie2p_native_switch_t* state = switches; state;
       state = state->next) {
    // Circuit slave mode bypasses packet filters. Replace both complete port
    // banks so a previous master cannot still consume the same stream.
    for (unsigned i = LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS;
         i <= LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES; ++i) {
      const loom_aie2p_native_register_bank_t* bank = &state->banks[i];
      records[position++] = loom_aie2p_catalog_block(bank->address, bank->words,
                                                     bank->word_count);
    }
  }
  loom_aie2p_program_record_t* setup = records + position;
  loom_aie2p_program_record_t* receivers = setup + count * kSetupCount;
  loom_aie2p_program_record_t* senders = receivers + count;
  loom_aie2p_program_record_t* retire = senders + count;
  iree_host_size_t index = 0;
  const loom_xdna_register_field_id_t mux =
      loom_xdna_shim_dma_mux_field(LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                                   shim->dma.memory_to_stream_port_base);
  const loom_xdna_register_field_id_t reset =
      loom_aie2p_sram_memory.directions[1].reset;
  const loom_xdna_register_field_id_t lock = loom_aie2p_sram_memory.lock_value;
  for (uint16_t column = 0; column < family->column_count; ++column) {
    if (!columns[column]) {
      continue;
    }
    const loom_xdna_tile_coordinate_t source_tile = {column, shim->first_row},
                                      memory_tile = {column, memory->first_row};
    const loom_aie2p_catalog_column_t* range = &ranges[column];
    const uint32_t size = range->contents_length;
    const uint32_t reset_address =
        loom_aie2p_catalog_address(family, memory_tile, reset);
    *setup++ = (loom_aie2p_program_record_t){
        .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32,
        .value.register_mask_write32 = {
            loom_aie2p_catalog_address(family, source_tile, mux),
            loom_xdna_register_field_encode(mux, -1),
            loom_xdna_register_field_encode(mux, 1)}};
    *setup++ = loom_aie2p_catalog_write(
        reset_address, loom_xdna_register_field_encode(reset, 1));
    *setup++ = loom_aie2p_catalog_write(reset_address, 0);
    *setup++ = loom_aie2p_catalog_write(
        loom_aie2p_catalog_address(
            family, source_tile,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_CONTROL_CONTROLLER_ID),
        0);
    *setup++ = loom_aie2p_catalog_write(
        loom_aie2p_catalog_address(family, memory_tile, lock), 0);
    uint32_t* receive = descriptors[index][0];
    loom_aie2p_native_encode_linear_descriptor(&memory->dma, memory->kind, size,
                                               receive);
    receive[1] = loom_xdna_register_field_encode(
        loom_aie2p_sram_memory.address,
        (memory->memory.local_load_base + range->begin) >>
            memory->dma.address_encoding_shift);
    receive[7] |= loom_xdna_register_field_encode(
                      loom_aie2p_sram_memory.locks.release_id,
                      loom_aie2p_sram_memory.local_lock_base) |
                  loom_xdna_register_field_encode(
                      loom_aie2p_sram_memory.locks.release_value, 1);
    *setup++ = loom_aie2p_catalog_block(
        loom_aie2p_catalog_address(family, memory_tile,
                                   loom_aie2p_sram_memory.descriptor_base),
        receive, 8);
    uint32_t* send = descriptors[index][1];
    loom_aie2p_native_encode_linear_descriptor(&shim->dma, shim->kind, size,
                                               send);
    patches[index] = (loom_aie2p_program_upload_patch_t){
        (uint32_t)(first_record + (setup - records)), range->source_offset,
        size};
    *setup++ = loom_aie2p_catalog_block(
        loom_aie2p_catalog_address(
            family, source_tile,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD0_BUFFER_LENGTH),
        send, 8);
    receivers[index] = loom_aie2p_catalog_write(
        loom_aie2p_catalog_address(family, memory_tile,
                                   loom_aie2p_sram_memory.directions[1].start),
        0);
    senders[index] = loom_aie2p_catalog_write(
        loom_aie2p_catalog_address(
            family, source_tile,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID),
        0);
    const uint32_t address =
        loom_aie2p_catalog_address(family, memory_tile, lock);
    *retire++ = loom_aie2p_catalog_wait(
        address, loom_xdna_register_field_encode(lock, -1),
        loom_xdna_register_field_encode(lock, 1));
    const loom_aie2p_native_queue_status_t sink =
        loom_aie2p_native_queue_status(
            family, memory_tile, &loom_aie2p_sram_memory, 0,
            LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY);
    const loom_aie2p_native_queue_status_t source =
        loom_aie2p_native_queue_status(
            family, source_tile, NULL, 0,
            LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM);
    *retire++ =
        loom_aie2p_catalog_wait((uint32_t)sink.address, sink.pending_mask, 0);
    *retire++ = loom_aie2p_catalog_wait((uint32_t)source.address,
                                        source.pending_mask, 0);
    *retire++ = loom_aie2p_catalog_write(address, 0);
    ++index;
  }
  IREE_ASSERT_EQ(retire - records, count * kRecordsPerColumn);
  loom_aie2p_catalog_upload_t* upload = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*upload), (void**)&upload));
  *upload = (loom_aie2p_catalog_upload_t){
      .program = {.records = records,
                  .record_count = count * kRecordsPerColumn,
                  .contents = contents,
                  .patches = patches,
                  .patch_count = count},
      .columns = columns,
  };
  *out_upload = upload;
  return iree_ok_status();
}
