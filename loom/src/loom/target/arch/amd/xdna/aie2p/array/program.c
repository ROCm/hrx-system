// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/program.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amd/xdna/array/registers.h"

enum {
  LOOM_AIE2P_COMPUTE_DMA_BUFFER_DESCRIPTOR_WORD_COUNT = 6,
  LOOM_AIE2P_SHIM_DMA_BUFFER_DESCRIPTOR_WORD_COUNT = 8,
  // The native AIE2P runtime initializes shim descriptors with a 128-byte AXI
  // burst and cache encoding 2. These values reproduce that linear DMA policy.
  LOOM_AIE2P_SHIM_DMA_BURST_LENGTH_ENCODING = 3,
  LOOM_AIE2P_SHIM_DMA_AXI_CACHE_ENCODING = 2,
  // Shim mux fields select DMA with value one.
  LOOM_AIE2P_SHIM_MUX_DMA_SELECTION = 1,
  // Controller packet identities are five bits when routed as TCT actors.
  LOOM_AIE2P_TASK_COMPLETION_CONTROLLER_MASK = 0x00001F00,
};

typedef struct loom_aie2p_program_record_builder_t {
  loom_aie2p_program_record_t* records;
  iree_host_size_t record_capacity;
  iree_host_size_t record_count;
  uint32_t* words;
  iree_host_size_t word_capacity;
  iree_host_size_t word_count;
} loom_aie2p_program_record_builder_t;

typedef struct loom_aie2p_register_update_t {
  uint32_t address;
  uint32_t mask;
  uint32_t value;
  bool requires_mask;
} loom_aie2p_register_update_t;

typedef struct loom_aie2p_array_program_builder_t {
  const loom_aie2p_array_plan_t* plan;
  loom_aie2p_program_record_builder_t array;
  loom_aie2p_program_record_builder_t control;
  loom_aie2p_program_relocation_t* relocations;
  iree_host_size_t relocation_capacity;
  iree_host_size_t relocation_count;
  loom_aie2p_register_update_t* route_updates;
  iree_host_size_t route_update_capacity;
  iree_host_size_t route_update_count;
} loom_aie2p_array_program_builder_t;

static iree_status_t loom_aie2p_program_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static iree_status_t loom_aie2p_program_add_capacity(
    iree_host_size_t amount, iree_host_size_t* inout_capacity) {
  iree_host_size_t capacity = 0;
  if (!iree_host_size_checked_add(*inout_capacity, amount, &capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P program cardinality overflows");
  }
  *inout_capacity = capacity;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_add_scaled_capacity(
    iree_host_size_t count, iree_host_size_t scale,
    iree_host_size_t* inout_capacity) {
  iree_host_size_t amount = 0;
  if (!iree_host_size_checked_mul(count, scale, &amount)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P program cardinality overflows");
  }
  return loom_aie2p_program_add_capacity(amount, inout_capacity);
}

static loom_xdna_register_field_id_t loom_aie2p_program_required_field(
    const char* key) {
  loom_xdna_register_field_id_t field_id = 0;
  IREE_CHECK_OK(
      loom_xdna_register_field_lookup(iree_make_cstring_view(key), &field_id));
  return field_id;
}

static iree_status_t loom_aie2p_program_resolve_field_update(
    const loom_aie2p_array_plan_t* plan, const char* key,
    loom_xdna_tile_coordinate_t coordinate, iree_host_size_t index_count,
    const uint16_t* indices, int64_t value,
    loom_aie2p_register_update_t* out_update) {
  *out_update = (loom_aie2p_register_update_t){0};
  const loom_xdna_register_field_id_t field_id =
      loom_aie2p_program_required_field(key);
  loom_xdna_register_field_info_t field_info = {0};
  IREE_CHECK_OK(loom_xdna_register_field_info(field_id, &field_info));
  uint32_t register_bits = 0;
  IREE_RETURN_IF_ERROR(
      loom_xdna_register_field_encode(field_id, value, &register_bits));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(loom_xdna_register_field_address(
      plan->family, field_id, coordinate, index_count, indices, &address));
  if (address > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P register address 0x%" PRIx64
                            " exceeds its 32-bit transaction domain",
                            address);
  }
  const uint32_t value_mask = field_info.bit_width == 32
                                  ? UINT32_MAX
                                  : (UINT32_C(1) << field_info.bit_width) - 1;
  *out_update = (loom_aie2p_register_update_t){
      .address = (uint32_t)address,
      .mask = value_mask << field_info.least_significant_bit,
      .value = register_bits,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_merge_register_update(
    const loom_aie2p_register_update_t* source,
    loom_aie2p_register_update_t* target) {
  if (target->address != source->address) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P register fields expected in one word resolve to 0x%08" PRIx32
        " and 0x%08" PRIx32,
        target->address, source->address);
  }
  const uint32_t overlap = target->mask & source->mask;
  if (((target->value ^ source->value) & overlap) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P register updates conflict at 0x%08" PRIx32,
                            target->address);
  }
  target->value = (target->value & ~source->mask) | source->value;
  target->mask |= source->mask;
  target->requires_mask |= source->requires_mask;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_accumulate_route_update(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_register_update_t* update) {
  for (iree_host_size_t i = 0; i < builder->route_update_count; ++i) {
    loom_aie2p_register_update_t* existing = &builder->route_updates[i];
    if (existing->address == update->address) {
      return loom_aie2p_program_merge_register_update(update, existing);
    }
  }
  IREE_ASSERT_LT(builder->route_update_count, builder->route_update_capacity);
  builder->route_updates[builder->route_update_count++] = *update;
  return iree_ok_status();
}

static loom_aie2p_program_record_t* loom_aie2p_program_append_record(
    loom_aie2p_program_record_builder_t* builder,
    loom_aie2p_program_record_type_t type) {
  IREE_ASSERT_LT(builder->record_count, builder->record_capacity);
  loom_aie2p_program_record_t* record =
      &builder->records[builder->record_count++];
  *record = (loom_aie2p_program_record_t){.type = type};
  return record;
}

static void loom_aie2p_program_append_register_write32(
    loom_aie2p_program_record_builder_t* builder, uint32_t address,
    uint32_t value) {
  loom_aie2p_program_record_t* record = loom_aie2p_program_append_record(
      builder, LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32);
  record->value.register_write32 = (loom_aie2p_program_register_write32_t){
      .address = address,
      .value = value,
  };
}

static void loom_aie2p_program_append_register_mask_write32(
    loom_aie2p_program_record_builder_t* builder, uint32_t address,
    uint32_t mask, uint32_t value) {
  loom_aie2p_program_record_t* record = loom_aie2p_program_append_record(
      builder, LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32);
  record->value.register_mask_write32 =
      (loom_aie2p_program_register_mask_write32_t){
          .address = address,
          .mask = mask,
          .value = value,
      };
}

static uint32_t* loom_aie2p_program_append_register_block_write32(
    loom_aie2p_program_record_builder_t* builder, uint32_t address,
    iree_host_size_t word_count) {
  IREE_ASSERT_LE(word_count, builder->word_capacity - builder->word_count);
  uint32_t* words = &builder->words[builder->word_count];
  builder->word_count += word_count;
  loom_aie2p_program_record_t* record = loom_aie2p_program_append_record(
      builder, LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32);
  record->value.register_block_write32 =
      (loom_aie2p_program_register_block_write32_t){
          .address = address,
          .words = words,
          .word_count = word_count,
      };
  return words;
}

static void loom_aie2p_program_append_tile_program_load(
    loom_aie2p_program_record_builder_t* builder, uint32_t tile_program_index) {
  loom_aie2p_program_record_t* record = loom_aie2p_program_append_record(
      builder, LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD);
  record->value.tile_program_load = (loom_aie2p_program_tile_program_load_t){
      .tile_program_index = tile_program_index,
  };
}

static void loom_aie2p_program_append_dma_task_wait(
    loom_aie2p_program_record_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_dma_direction_t direction, uint8_t dma_channel) {
  loom_aie2p_program_record_t* record = loom_aie2p_program_append_record(
      builder, LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT);
  record->value.dma_task_wait = (loom_aie2p_program_dma_task_wait_t){
      .coordinate = coordinate,
      .direction = direction,
      .dma_channel = dma_channel,
      .column_count = 1,
      .row_count = 1,
  };
}

static const char* loom_aie2p_program_stream_master_enable_key(
    loom_xdna_tile_kind_t tile_kind) {
  switch (tile_kind) {
    case LOOM_XDNA_TILE_KIND_SHIM_NOC:
      return "shim_noc.stream.master_config.enable";
    case LOOM_XDNA_TILE_KIND_MEMORY:
      return "memory_tile.stream.master_config.enable";
    case LOOM_XDNA_TILE_KIND_COMPUTE:
      return "core.stream.master_config.enable";
  }
  IREE_ASSERT_UNREACHABLE("validated AIE2P tile kind");
  return NULL;
}

static const char* loom_aie2p_program_stream_master_configuration_key(
    loom_xdna_tile_kind_t tile_kind) {
  switch (tile_kind) {
    case LOOM_XDNA_TILE_KIND_SHIM_NOC:
      return "shim_noc.stream.master_config.configuration";
    case LOOM_XDNA_TILE_KIND_MEMORY:
      return "memory_tile.stream.master_config.configuration";
    case LOOM_XDNA_TILE_KIND_COMPUTE:
      return "core.stream.master_config.configuration";
  }
  IREE_ASSERT_UNREACHABLE("validated AIE2P tile kind");
  return NULL;
}

static const char* loom_aie2p_program_stream_slave_enable_key(
    loom_xdna_tile_kind_t tile_kind) {
  switch (tile_kind) {
    case LOOM_XDNA_TILE_KIND_SHIM_NOC:
      return "shim_noc.stream.slave_config.enable";
    case LOOM_XDNA_TILE_KIND_MEMORY:
      return "memory_tile.stream.slave_config.enable";
    case LOOM_XDNA_TILE_KIND_COMPUTE:
      return "core.stream.slave_config.enable";
  }
  IREE_ASSERT_UNREACHABLE("validated AIE2P tile kind");
  return NULL;
}

static const char* loom_aie2p_program_compute_dma_queue_key(
    loom_aie2p_array_dma_direction_t direction) {
  switch (direction) {
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM:
      return "compute_memory.dma.channel.mm2s.start_queue.start_bd_id";
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY:
      return "compute_memory.dma.channel.s2mm.start_queue.start_bd_id";
  }
  IREE_ASSERT_UNREACHABLE("validated AIE2P DMA direction");
  return NULL;
}

static const char* loom_aie2p_program_shim_dma_control_controller_key(
    loom_aie2p_array_dma_direction_t direction) {
  switch (direction) {
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM:
      return "shim_noc.dma.channel.mm2s.control.controller_id";
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY:
      return "shim_noc.dma.channel.s2mm.control.controller_id";
  }
  IREE_ASSERT_UNREACHABLE("validated AIE2P DMA direction");
  return NULL;
}

static const char* loom_aie2p_program_shim_dma_queue_start_key(
    loom_aie2p_array_dma_direction_t direction) {
  switch (direction) {
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM:
      return "shim_noc.dma.channel.mm2s.task_queue.start_bd_id";
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY:
      return "shim_noc.dma.channel.s2mm.task_queue.start_bd_id";
  }
  IREE_ASSERT_UNREACHABLE("validated AIE2P DMA direction");
  return NULL;
}

static const char* loom_aie2p_program_shim_dma_queue_token_key(
    loom_aie2p_array_dma_direction_t direction) {
  switch (direction) {
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM:
      return "shim_noc.dma.channel.mm2s.task_queue.enable_token_issue";
    case LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY:
      return "shim_noc.dma.channel.s2mm.task_queue.enable_token_issue";
  }
  IREE_ASSERT_UNREACHABLE("validated AIE2P DMA direction");
  return NULL;
}

static iree_status_t loom_aie2p_program_append_masked_field(
    const loom_aie2p_array_plan_t* plan,
    loom_aie2p_program_record_builder_t* records, const char* key,
    loom_xdna_tile_coordinate_t coordinate, iree_host_size_t index_count,
    const uint16_t* indices, int64_t value) {
  loom_aie2p_register_update_t update = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      plan, key, coordinate, index_count, indices, value, &update));
  loom_aie2p_program_append_register_mask_write32(records, update.address,
                                                  update.mask, update.value);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_append_core_reset(
    loom_aie2p_array_program_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate) {
  loom_aie2p_register_update_t reset = {0};
  loom_aie2p_register_update_t disable = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan, "core.control.reset", coordinate, 0, NULL, 1, &reset));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan, "core.control.enable", coordinate, 0, NULL, 0, &disable));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_merge_register_update(&disable, &reset));
  loom_aie2p_program_append_register_mask_write32(
      &builder->array, reset.address, reset.mask, reset.value);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_append_lock_initialization(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_array_lock_plan_t* lock) {
  const uint16_t indices[] = {lock->lock_id};
  loom_aie2p_register_update_t update = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan, "compute_memory.lock.value.value", lock->coordinate,
      IREE_ARRAYSIZE(indices), indices, lock->initial_value, &update));
  loom_aie2p_program_append_register_write32(&builder->array, update.address,
                                             update.value);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_stream_ordinal(
    const loom_aie2p_array_plan_t* plan, loom_xdna_tile_kind_t tile_kind,
    loom_xdna_stream_direction_t direction, loom_xdna_stream_port_t port,
    uint8_t channel, uint16_t* out_ordinal) {
  const loom_xdna_stream_port_range_t* range = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_array_stream_port_range(
      plan->family, tile_kind, direction, port, &range));
  if (channel >= range->count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P stream port %u channel %u exceeds %u channels", (unsigned)port,
        (unsigned)channel, (unsigned)range->count);
  }
  *out_ordinal = range->ordinal + channel;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_accumulate_stream_route(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_array_route_plan_t* route) {
  const loom_xdna_tile_facts_t* tile = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(builder->plan->family,
                                                  route->coordinate, &tile));
  uint16_t source_ordinal = 0;
  uint16_t destination_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_stream_ordinal(
      builder->plan, tile->kind, LOOM_XDNA_STREAM_DIRECTION_SLAVE,
      route->source_port, route->source_channel, &source_ordinal));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_stream_ordinal(
      builder->plan, tile->kind, LOOM_XDNA_STREAM_DIRECTION_MASTER,
      route->destination_port, route->destination_channel,
      &destination_ordinal));

  const uint16_t master_indices[] = {destination_ordinal};
  loom_aie2p_register_update_t master = {0};
  loom_aie2p_register_update_t field = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan, loom_aie2p_program_stream_master_enable_key(tile->kind),
      route->coordinate, IREE_ARRAYSIZE(master_indices), master_indices, 1,
      &master));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan,
      loom_aie2p_program_stream_master_configuration_key(tile->kind),
      route->coordinate, IREE_ARRAYSIZE(master_indices), master_indices,
      source_ordinal, &field));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_merge_register_update(&field, &master));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_accumulate_route_update(builder, &master));

  const uint16_t slave_indices[] = {source_ordinal};
  loom_aie2p_register_update_t slave = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan, loom_aie2p_program_stream_slave_enable_key(tile->kind),
      route->coordinate, IREE_ARRAYSIZE(slave_indices), slave_indices, 1,
      &slave));
  return loom_aie2p_program_accumulate_route_update(builder, &slave);
}

static const char* loom_aie2p_program_shim_mux_key(uint8_t south_channel) {
  switch (south_channel) {
    case 2:
      return "shim_noc.mux_config.south2";
    case 3:
      return "shim_noc.mux_config.south3";
    case 6:
      return "shim_noc.mux_config.south6";
    case 7:
      return "shim_noc.mux_config.south7";
    default:
      return NULL;
  }
}

static const char* loom_aie2p_program_shim_demux_key(uint8_t south_channel) {
  switch (south_channel) {
    case 2:
      return "shim_noc.demux_config.south2";
    case 3:
      return "shim_noc.demux_config.south3";
    case 4:
      return "shim_noc.demux_config.south4";
    case 5:
      return "shim_noc.demux_config.south5";
    default:
      return NULL;
  }
}

static iree_status_t loom_aie2p_program_accumulate_shim_mux_route(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_array_route_plan_t* route) {
  const char* key = NULL;
  if (route->source_port == LOOM_XDNA_STREAM_PORT_DMA &&
      route->destination_port == LOOM_XDNA_STREAM_PORT_NORTH) {
    key = loom_aie2p_program_shim_mux_key(route->destination_channel);
  } else if (route->source_port == LOOM_XDNA_STREAM_PORT_NORTH &&
             route->destination_port == LOOM_XDNA_STREAM_PORT_DMA) {
    key = loom_aie2p_program_shim_demux_key(route->source_channel);
  }
  if (key == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P shim mux route has unsupported ports %u[%u] -> %u[%u]",
        (unsigned)route->source_port, (unsigned)route->source_channel,
        (unsigned)route->destination_port,
        (unsigned)route->destination_channel);
  }
  loom_aie2p_register_update_t update = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan, key, route->coordinate, 0, NULL,
      LOOM_AIE2P_SHIM_MUX_DMA_SELECTION, &update));
  update.requires_mask = true;
  return loom_aie2p_program_accumulate_route_update(builder, &update);
}

static iree_status_t loom_aie2p_program_build_routes(
    loom_aie2p_array_program_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->plan->route_count; ++i) {
    const loom_aie2p_array_route_plan_t* route = &builder->plan->routes[i];
    switch (route->switch_kind) {
      case LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH:
        IREE_RETURN_IF_ERROR(
            loom_aie2p_program_accumulate_stream_route(builder, route));
        break;
      case LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX:
        IREE_RETURN_IF_ERROR(
            loom_aie2p_program_accumulate_shim_mux_route(builder, route));
        break;
      default:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "unknown AIE2P route switch kind %u",
                                (unsigned)route->switch_kind);
    }
  }
  for (iree_host_size_t i = 0; i < builder->route_update_count; ++i) {
    const loom_aie2p_register_update_t* update = &builder->route_updates[i];
    // Stream-switch writes describe the complete supported register value.
    // Shim mux registers share selectors not necessarily owned by one route.
    if (update->requires_mask) {
      loom_aie2p_program_append_register_mask_write32(
          &builder->array, update->address, update->mask, update->value);
    } else {
      loom_aie2p_program_append_register_write32(
          &builder->array, update->address, update->value);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_encode_field(const char* key,
                                                     int64_t value,
                                                     uint32_t* inout_word) {
  const loom_xdna_register_field_id_t field_id =
      loom_aie2p_program_required_field(key);
  uint32_t register_bits = 0;
  IREE_RETURN_IF_ERROR(
      loom_xdna_register_field_encode(field_id, value, &register_bits));
  *inout_word |= register_bits;
  return iree_ok_status();
}

static const loom_aie2p_array_channel_slot_t*
loom_aie2p_program_find_channel_slot(const loom_aie2p_array_plan_t* plan,
                                     uint32_t channel_index, uint32_t slot) {
  for (iree_host_size_t i = 0; i < plan->channel_slot_count; ++i) {
    const loom_aie2p_array_channel_slot_t* candidate = &plan->channel_slots[i];
    if (candidate->channel_index == channel_index && candidate->slot == slot) {
      return candidate;
    }
  }
  IREE_ASSERT_UNREACHABLE("planned AIE2P channel slot must exist");
  return NULL;
}

static const loom_aie2p_array_lock_plan_t* loom_aie2p_program_find_channel_lock(
    const loom_aie2p_array_plan_t* plan, uint32_t channel_index,
    bool consumer_ready) {
  for (iree_host_size_t i = 0; i < plan->lock_count; ++i) {
    const loom_aie2p_array_lock_plan_t* candidate = &plan->locks[i];
    if (candidate->channel_index == channel_index &&
        candidate->consumer_ready == consumer_ready) {
      return candidate;
    }
  }
  IREE_ASSERT_UNREACHABLE("planned AIE2P channel lock must exist");
  return NULL;
}

static const loom_aie2p_array_dma_plan_t* loom_aie2p_program_find_shim_dma(
    const loom_aie2p_array_plan_t* plan, uint32_t channel_index) {
  for (iree_host_size_t i = 0; i < plan->dma_channel_count; ++i) {
    const loom_aie2p_array_dma_plan_t* candidate = &plan->dma_channels[i];
    if (candidate->channel_index == channel_index && candidate->shim_side) {
      return candidate;
    }
  }
  IREE_ASSERT_UNREACHABLE("planned AIE2P shim DMA must exist");
  return NULL;
}

static iree_status_t loom_aie2p_program_compute_dma_buffer_descriptor_address(
    const loom_aie2p_array_plan_t* plan, loom_xdna_tile_coordinate_t coordinate,
    uint16_t buffer_descriptor, uint32_t* out_address) {
  const uint16_t indices[] = {buffer_descriptor};
  loom_aie2p_register_update_t update = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      plan, "compute_memory.dma.bd.word0.buffer_length", coordinate,
      IREE_ARRAYSIZE(indices), indices, 0, &update));
  *out_address = update.address;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_build_compute_dma_descriptor(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_array_dma_plan_t* dma, uint32_t slot_ordinal) {
  const loom_aie2p_array_plan_t* plan = builder->plan;
  const loom_aie2p_array_channel_t* channel =
      &plan->channels[dma->channel_index];
  const loom_aie2p_array_channel_slot_t* slot =
      loom_aie2p_program_find_channel_slot(plan, dma->channel_index,
                                           slot_ordinal);
  const loom_aie2p_array_lock_plan_t* credit_lock =
      loom_aie2p_program_find_channel_lock(plan, dma->channel_index, false);
  const loom_aie2p_array_lock_plan_t* ready_lock =
      loom_aie2p_program_find_channel_lock(plan, dma->channel_index, true);
  const loom_aie2p_array_lock_plan_t* acquire_lock =
      dma->direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY
          ? credit_lock
          : ready_lock;
  const loom_aie2p_array_lock_plan_t* release_lock =
      dma->direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY
          ? ready_lock
          : credit_lock;
  // Compute DMA descriptors address the engine tile's own local memory. Core
  // loads use the separately planned tile aperture (for example 0x70000 for
  // self-memory), which is not a valid DMA descriptor address.
  IREE_ASSERT_EQ(slot->owner.column, dma->coordinate.column);
  IREE_ASSERT_EQ(slot->owner.row, dma->coordinate.row);
  const uint32_t local_address = slot->owner_offset;
  const loom_xdna_tile_facts_t* tile = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_array_tile_facts(plan->family, dma->coordinate, &tile));
  if (local_address % tile->dma.address_alignment != 0 ||
      channel->record_byte_length % tile->dma.transfer_length_granularity !=
          0 ||
      (uint64_t)local_address + channel->record_byte_length >
          tile->dma.address_maximum) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P compute DMA address or length violates engine alignment");
  }

  const uint16_t buffer_descriptor =
      dma->buffer_descriptor_start + (uint16_t)slot_ordinal;
  uint32_t descriptor_address = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_compute_dma_buffer_descriptor_address(
      plan, dma->coordinate, buffer_descriptor, &descriptor_address));
  uint32_t* words = loom_aie2p_program_append_register_block_write32(
      &builder->array, descriptor_address,
      LOOM_AIE2P_COMPUTE_DMA_BUFFER_DESCRIPTOR_WORD_COUNT);
  memset(words, 0,
         LOOM_AIE2P_COMPUTE_DMA_BUFFER_DESCRIPTOR_WORD_COUNT * sizeof(*words));
  const uint32_t encoded_length =
      channel->record_byte_length / tile->dma.transfer_length_granularity -
      tile->dma.transfer_length_offset;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word0.base_address",
      local_address >> tile->dma.address_encoding_shift, &words[0]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word0.buffer_length", encoded_length, &words[0]));
  const uint16_t next_buffer_descriptor =
      dma->buffer_descriptor_start +
      (uint16_t)((slot_ordinal + 1) % dma->buffer_descriptor_count);
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_encode_field("compute_memory.dma.bd.word5.next_bd",
                                      next_buffer_descriptor, &words[5]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word5.use_next_bd", 1, &words[5]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word5.valid_bd", 1, &words[5]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word5.lock_release_value", 1, &words[5]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word5.lock_release_id", release_lock->lock_id,
      &words[5]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word5.lock_acquire_enable", 1, &words[5]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word5.lock_acquire_value", -1, &words[5]));
  return loom_aie2p_program_encode_field(
      "compute_memory.dma.bd.word5.lock_acquire_id", acquire_lock->lock_id,
      &words[5]);
}

static iree_status_t loom_aie2p_program_build_compute_dma(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_array_dma_plan_t* dma) {
  for (uint32_t slot = 0; slot < dma->buffer_descriptor_count; ++slot) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_build_compute_dma_descriptor(builder, dma, slot));
  }
  const uint16_t indices[] = {dma->dma_channel};
  loom_aie2p_register_update_t queue = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan, loom_aie2p_program_compute_dma_queue_key(dma->direction),
      dma->coordinate, IREE_ARRAYSIZE(indices), indices,
      dma->buffer_descriptor_start, &queue));
  loom_aie2p_program_append_register_write32(&builder->array, queue.address,
                                             queue.value);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_shim_dma_buffer_descriptor_address(
    const loom_aie2p_array_plan_t* plan, loom_xdna_tile_coordinate_t coordinate,
    uint16_t buffer_descriptor, uint32_t* out_address) {
  const uint16_t indices[] = {buffer_descriptor};
  loom_aie2p_register_update_t update = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      plan, "shim_noc.dma.bd.word0.buffer_length", coordinate,
      IREE_ARRAYSIZE(indices), indices, 0, &update));
  *out_address = update.address;
  return iree_ok_status();
}

static void loom_aie2p_program_append_relocation(
    loom_aie2p_array_program_builder_t* builder, uint32_t target_record_index,
    uint32_t binding_ordinal, int64_t addend, uint32_t transfer_byte_length,
    const loom_xdna_dma_facts_t* dma_facts) {
  IREE_ASSERT_LT(builder->relocation_count, builder->relocation_capacity);
  builder->relocations[builder->relocation_count++] =
      (loom_aie2p_program_relocation_t){
          .target_record_index = target_record_index,
          .target_word_index = 1,
          .binding_ordinal = binding_ordinal,
          .kind = LOOM_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS,
          .field_byte_width = 8,
          .addend = addend,
          .minimum_value = 0,
          .maximum_value = dma_facts->address_maximum - transfer_byte_length,
          .required_alignment = dma_facts->address_alignment,
      };
}

static iree_status_t loom_aie2p_program_build_shim_dma_descriptor(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_array_binding_plan_t* binding_plan,
    const loom_aie2p_array_dma_plan_t* dma) {
  const loom_aie2p_array_channel_t* channel =
      &builder->plan->channels[binding_plan->channel_index];
  const loom_xdna_tile_facts_t* tile = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(builder->plan->family,
                                                  dma->coordinate, &tile));
  if (channel->record_byte_length % tile->dma.transfer_length_granularity !=
      0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P shim DMA length violates engine alignment");
  }
  uint32_t descriptor_address = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_shim_dma_buffer_descriptor_address(
      builder->plan, dma->coordinate, dma->buffer_descriptor_start,
      &descriptor_address));
  const uint32_t target_record_index = (uint32_t)builder->control.record_count;
  uint32_t* words = loom_aie2p_program_append_register_block_write32(
      &builder->control, descriptor_address,
      LOOM_AIE2P_SHIM_DMA_BUFFER_DESCRIPTOR_WORD_COUNT);
  memset(words, 0,
         LOOM_AIE2P_SHIM_DMA_BUFFER_DESCRIPTOR_WORD_COUNT * sizeof(*words));
  const uint32_t encoded_length =
      channel->record_byte_length / tile->dma.transfer_length_granularity -
      tile->dma.transfer_length_offset;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "shim_noc.dma.bd.word0.buffer_length", encoded_length, &words[0]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "shim_noc.dma.bd.word4.burst_length",
      LOOM_AIE2P_SHIM_DMA_BURST_LENGTH_ENCODING, &words[4]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "shim_noc.dma.bd.word5.axi_cache", LOOM_AIE2P_SHIM_DMA_AXI_CACHE_ENCODING,
      &words[5]));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_field(
      "shim_noc.dma.bd.word7.valid_bd", 1, &words[7]));

  const uint64_t addend =
      (uint64_t)binding_plan->partition_lane * channel->record_byte_length;
  if (addend > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P binding partition offset overflows");
  }
  const uint32_t binding_ordinal =
      builder->plan->bindings[binding_plan->binding_index].ordinal;
  loom_aie2p_program_append_relocation(builder, target_record_index,
                                       binding_ordinal, (int64_t)addend,
                                       channel->record_byte_length, &tile->dma);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_build_shim_dma_queue(
    loom_aie2p_array_program_builder_t* builder,
    const loom_aie2p_array_dma_plan_t* dma) {
  const uint16_t indices[] = {dma->dma_channel};
  const bool issues_completion =
      dma->direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY;
  if (issues_completion) {
    uint8_t controller_id = 0;
    IREE_RETURN_IF_ERROR(loom_xdna_array_controller_id(
        builder->plan->family, dma->coordinate, &controller_id));
    loom_aie2p_register_update_t controller = {0};
    IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
        builder->plan,
        loom_aie2p_program_shim_dma_control_controller_key(dma->direction),
        dma->coordinate, IREE_ARRAYSIZE(indices), indices, controller_id,
        &controller));
    loom_aie2p_program_append_register_mask_write32(
        &builder->control, controller.address,
        LOOM_AIE2P_TASK_COMPLETION_CONTROLLER_MASK, controller.value);
  }

  loom_aie2p_register_update_t queue = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
      builder->plan,
      loom_aie2p_program_shim_dma_queue_start_key(dma->direction),
      dma->coordinate, IREE_ARRAYSIZE(indices), indices,
      dma->buffer_descriptor_start, &queue));
  if (issues_completion) {
    loom_aie2p_register_update_t token = {0};
    IREE_RETURN_IF_ERROR(loom_aie2p_program_resolve_field_update(
        builder->plan,
        loom_aie2p_program_shim_dma_queue_token_key(dma->direction),
        dma->coordinate, IREE_ARRAYSIZE(indices), indices, 1, &token));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_merge_register_update(&token, &queue));
  }
  loom_aie2p_program_append_register_write32(&builder->control, queue.address,
                                             queue.value);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_build_control(
    loom_aie2p_array_program_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->plan->binding_plan_count; ++i) {
    const loom_aie2p_array_binding_plan_t* binding_plan =
        &builder->plan->binding_plans[i];
    const loom_aie2p_array_dma_plan_t* dma = loom_aie2p_program_find_shim_dma(
        builder->plan, binding_plan->channel_index);
    IREE_RETURN_IF_ERROR(loom_aie2p_program_build_shim_dma_descriptor(
        builder, binding_plan, dma));
    IREE_RETURN_IF_ERROR(loom_aie2p_program_build_shim_dma_queue(builder, dma));
  }
  for (iree_host_size_t i = 0; i < builder->plan->binding_plan_count; ++i) {
    const loom_aie2p_array_binding_plan_t* binding_plan =
        &builder->plan->binding_plans[i];
    if (binding_plan->direction !=
        LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY) {
      continue;
    }
    loom_aie2p_program_append_dma_task_wait(
        &builder->control, binding_plan->shim_coordinate,
        binding_plan->direction, binding_plan->dma_channel);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_count_storage(
    const loom_aie2p_array_plan_t* plan,
    iree_host_size_t* out_array_record_capacity,
    iree_host_size_t* out_array_word_capacity,
    iree_host_size_t* out_control_record_capacity,
    iree_host_size_t* out_control_word_capacity,
    iree_host_size_t* out_route_update_capacity) {
  iree_host_size_t compute_dma_count = 0;
  iree_host_size_t compute_buffer_descriptor_count = 0;
  for (iree_host_size_t i = 0; i < plan->dma_channel_count; ++i) {
    const loom_aie2p_array_dma_plan_t* dma = &plan->dma_channels[i];
    if (!dma->shim_side) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_program_add_capacity(1, &compute_dma_count));
      IREE_RETURN_IF_ERROR(loom_aie2p_program_add_capacity(
          dma->buffer_descriptor_count, &compute_buffer_descriptor_count));
    }
  }

  iree_host_size_t array_record_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_scaled_capacity(
      plan->worker_plan_count, 4, &array_record_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_capacity(plan->lock_count,
                                                       &array_record_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_scaled_capacity(
      plan->route_count, 2, &array_record_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_capacity(
      compute_buffer_descriptor_count, &array_record_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_capacity(compute_dma_count,
                                                       &array_record_capacity));

  iree_host_size_t array_word_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_scaled_capacity(
      compute_buffer_descriptor_count,
      LOOM_AIE2P_COMPUTE_DMA_BUFFER_DESCRIPTOR_WORD_COUNT,
      &array_word_capacity));

  iree_host_size_t control_record_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_scaled_capacity(
      plan->binding_plan_count, 4, &control_record_capacity));
  iree_host_size_t control_word_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_scaled_capacity(
      plan->binding_plan_count,
      LOOM_AIE2P_SHIM_DMA_BUFFER_DESCRIPTOR_WORD_COUNT,
      &control_word_capacity));
  iree_host_size_t route_update_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_add_scaled_capacity(
      plan->route_count, 2, &route_update_capacity));

  *out_array_record_capacity = array_record_capacity;
  *out_array_word_capacity = array_word_capacity;
  *out_control_record_capacity = control_record_capacity;
  *out_control_word_capacity = control_word_capacity;
  *out_route_update_capacity = route_update_capacity;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_build_array(
    loom_aie2p_array_program_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->plan->worker_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_append_core_reset(
        builder, builder->plan->worker_plans[i].coordinate));
  }
  for (iree_host_size_t i = 0; i < builder->plan->lock_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_append_lock_initialization(
        builder, &builder->plan->locks[i]));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_build_routes(builder));
  for (iree_host_size_t i = 0; i < builder->plan->dma_channel_count; ++i) {
    const loom_aie2p_array_dma_plan_t* dma = &builder->plan->dma_channels[i];
    if (!dma->shim_side) {
      IREE_RETURN_IF_ERROR(loom_aie2p_program_build_compute_dma(builder, dma));
    }
  }
  for (iree_host_size_t i = 0; i < builder->plan->worker_plan_count; ++i) {
    loom_aie2p_program_append_tile_program_load(&builder->array, (uint32_t)i);
  }
  for (iree_host_size_t i = 0; i < builder->plan->worker_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_append_masked_field(
        builder->plan, &builder->array, "core.control.reset",
        builder->plan->worker_plans[i].coordinate, 0, NULL, 0));
  }
  for (iree_host_size_t i = 0; i < builder->plan->worker_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_append_masked_field(
        builder->plan, &builder->array, "core.control.enable",
        builder->plan->worker_plans[i].coordinate, 0, NULL, 1));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_program_build(
    const loom_aie2p_array_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_array_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_aie2p_array_program_t){0};
  if (plan->family == NULL ||
      plan->family->architecture != LOOM_XDNA_ARCHITECTURE_AIE2P) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P program requires an AIE2P array plan");
  }
  if (plan->worker_plan_count > UINT32_MAX ||
      plan->binding_plan_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P program cardinality exceeds its ABI");
  }

  iree_host_size_t array_record_capacity = 0;
  iree_host_size_t array_word_capacity = 0;
  iree_host_size_t control_record_capacity = 0;
  iree_host_size_t control_word_capacity = 0;
  iree_host_size_t route_update_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_count_storage(
      plan, &array_record_capacity, &array_word_capacity,
      &control_record_capacity, &control_word_capacity,
      &route_update_capacity));

  loom_aie2p_array_program_builder_t builder = {.plan = plan};
  builder.array.record_capacity = array_record_capacity;
  builder.array.word_capacity = array_word_capacity;
  builder.control.record_capacity = control_record_capacity;
  builder.control.word_capacity = control_word_capacity;
  builder.relocation_capacity = plan->binding_plan_count;
  builder.route_update_capacity = route_update_capacity;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, builder.array.record_capacity, sizeof(*builder.array.records),
      (void**)&builder.array.records));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, builder.array.word_capacity, sizeof(*builder.array.words),
      (void**)&builder.array.words));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, builder.control.record_capacity, sizeof(*builder.control.records),
      (void**)&builder.control.records));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, builder.control.word_capacity, sizeof(*builder.control.words),
      (void**)&builder.control.words));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, builder.relocation_capacity, sizeof(*builder.relocations),
      (void**)&builder.relocations));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, builder.route_update_capacity, sizeof(*builder.route_updates),
      (void**)&builder.route_updates));

  IREE_RETURN_IF_ERROR(loom_aie2p_program_build_array(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_build_control(&builder));
  *out_program = (loom_aie2p_array_program_t){
      .array_records = builder.array.records,
      .array_record_count = builder.array.record_count,
      .control_records = builder.control.records,
      .control_record_count = builder.control.record_count,
      .relocations = builder.relocations,
      .relocation_count = builder.relocation_count,
      .tile_program_count = (uint32_t)plan->worker_plan_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_record_byte_length(
    const loom_aie2p_program_record_t* record,
    iree_host_size_t* out_byte_length) {
  iree_host_size_t byte_length = 0;
  switch (record->type) {
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      byte_length = 16;
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      byte_length = 20;
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32:
      if (!iree_host_size_checked_mul_add(
              16, record->value.register_block_write32.word_count,
              sizeof(uint32_t), &byte_length)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "AIE2P block-write record overflows");
      }
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
      byte_length = 12;
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
      byte_length = 16;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AIE2P program record type %u",
                              (unsigned)record->type);
  }
  if (byte_length > LOOM_XDNA_ELF_MAX_PROGRAM_RECORD_SIZE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P program record exceeds the payload ABI");
  }
  *out_byte_length = byte_length;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_payload_byte_length(
    const loom_aie2p_program_record_t* records, iree_host_size_t record_count,
    iree_host_size_t header_byte_length, iree_host_size_t* out_byte_length) {
  if (record_count > LOOM_XDNA_ELF_MAX_PROGRAM_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P program has too many records");
  }
  iree_host_size_t byte_length = header_byte_length;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    iree_host_size_t record_byte_length = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_program_record_byte_length(
        &records[i], &record_byte_length));
    if (!iree_host_size_checked_add(byte_length, record_byte_length,
                                    &byte_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P program payload overflows");
    }
  }
  if (byte_length > LOOM_XDNA_ELF_MAX_PROGRAM_PAYLOAD_SIZE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P program payload exceeds its ABI");
  }
  *out_byte_length = byte_length;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_encode_record(
    const loom_aie2p_program_record_t* record,
    uint32_t first_tile_program_header_ordinal, uint32_t tile_program_count,
    iree_byte_span_t storage) {
  iree_host_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_record_byte_length(record, &byte_length));
  if (storage.data_length < byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P program record storage is too small");
  }
  const loom_xdna_elf_program_record_header_t header = {
      .type = (uint16_t)record->type,
      .flags = 0,
      .byte_length = (uint32_t)byte_length,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_program_record_header(
      &header, iree_make_byte_span(storage.data,
                                   LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE)));
  switch (record->type) {
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      iree_unaligned_store_le_u32(storage.data + 8,
                                  record->value.register_write32.address);
      iree_unaligned_store_le_u32(storage.data + 12,
                                  record->value.register_write32.value);
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      iree_unaligned_store_le_u32(storage.data + 8,
                                  record->value.register_mask_write32.address);
      iree_unaligned_store_le_u32(storage.data + 12,
                                  record->value.register_mask_write32.mask);
      iree_unaligned_store_le_u32(storage.data + 16,
                                  record->value.register_mask_write32.value);
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32: {
      const loom_aie2p_program_register_block_write32_t* block =
          &record->value.register_block_write32;
      if (block->word_count > UINT32_MAX ||
          (block->word_count != 0 && block->words == NULL)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P block-write record is malformed");
      }
      iree_unaligned_store_le_u32(storage.data + 8, block->address);
      iree_unaligned_store_le_u32(storage.data + 12,
                                  (uint32_t)block->word_count);
      for (iree_host_size_t i = 0; i < block->word_count; ++i) {
        iree_unaligned_store_le_u32(storage.data + 16 + i * sizeof(uint32_t),
                                    block->words[i]);
      }
      break;
    }
    case LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD: {
      const uint32_t tile_program_index =
          record->value.tile_program_load.tile_program_index;
      if (tile_program_index >= tile_program_count ||
          tile_program_index > UINT32_MAX - first_tile_program_header_ordinal) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P tile program reference is invalid");
      }
      iree_unaligned_store_le_u32(
          storage.data + 8,
          first_tile_program_header_ordinal + tile_program_index);
      break;
    }
    case LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT: {
      const loom_aie2p_program_dma_task_wait_t* wait =
          &record->value.dma_task_wait;
      if (wait->coordinate.column > UINT8_MAX ||
          wait->coordinate.row > UINT8_MAX || wait->column_count == 0 ||
          wait->row_count == 0) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P DMA task wait range is invalid");
      }
      uint8_t direction = 0;
      switch (wait->direction) {
        case LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY:
          direction = 0;
          break;
        case LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM:
          direction = 1;
          break;
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "AIE2P DMA task wait direction is invalid");
      }
      storage.data[8] = (uint8_t)wait->coordinate.column;
      storage.data[9] = (uint8_t)wait->coordinate.row;
      storage.data[10] = direction;
      storage.data[11] = wait->dma_channel;
      storage.data[12] = wait->column_count;
      storage.data[13] = wait->row_count;
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("validated AIE2P program record type");
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_encode_records(
    const loom_aie2p_program_record_t* records, iree_host_size_t record_count,
    uint32_t first_tile_program_header_ordinal, uint32_t tile_program_count,
    iree_host_size_t header_byte_length, uint32_t* record_offsets,
    iree_byte_span_t storage) {
  iree_host_size_t offset = header_byte_length;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    iree_host_size_t record_byte_length = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_program_record_byte_length(
        &records[i], &record_byte_length));
    if (record_offsets != NULL) record_offsets[i] = (uint32_t)offset;
    IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_record(
        &records[i], first_tile_program_header_ordinal, tile_program_count,
        iree_make_byte_span(storage.data + offset, record_byte_length)));
    offset += record_byte_length;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_program_encode(
    const loom_aie2p_array_program_t* program,
    uint32_t first_tile_program_header_ordinal,
    uint32_t control_program_header_ordinal, iree_arena_allocator_t* arena,
    loom_aie2p_encoded_array_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_aie2p_encoded_array_program_t){0};
  if ((program->array_record_count != 0 && program->array_records == NULL) ||
      (program->control_record_count != 0 &&
       program->control_records == NULL) ||
      (program->relocation_count != 0 && program->relocations == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P program storage is incomplete");
  }
  if (program->tile_program_count >
      UINT32_MAX - first_tile_program_header_ordinal) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P tile program-header range overflows");
  }

  iree_host_size_t array_byte_length = 0;
  iree_host_size_t control_byte_length = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_payload_byte_length(
      program->array_records, program->array_record_count,
      LOOM_XDNA_ELF_ARRAY_HEADER_SIZE, &array_byte_length));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_payload_byte_length(
      program->control_records, program->control_record_count,
      LOOM_XDNA_ELF_CONTROL_HEADER_SIZE, &control_byte_length));
  uint8_t* array_payload = NULL;
  uint8_t* control_payload = NULL;
  uint32_t* control_record_offsets = NULL;
  loom_xdna_elf_relocation_record_t* relocations = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, array_byte_length, (void**)&array_payload));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, control_byte_length,
                                           (void**)&control_payload));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, program->control_record_count, sizeof(*control_record_offsets),
      (void**)&control_record_offsets));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_allocate_array(
      arena, program->relocation_count, sizeof(*relocations),
      (void**)&relocations));
  memset(array_payload, 0, array_byte_length);
  memset(control_payload, 0, control_byte_length);

  const loom_xdna_elf_array_header_t array_header = {
      .abi_major = LOOM_XDNA_ELF_PROGRAM_ABI_MAJOR,
      .abi_minor = LOOM_XDNA_ELF_PROGRAM_ABI_MINOR,
      .record_count = (uint32_t)program->array_record_count,
      .byte_length = (uint32_t)array_byte_length,
      .flags = 0,
      .first_tile_program_header_ordinal = first_tile_program_header_ordinal,
      .tile_program_header_count = program->tile_program_count,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_array_header(
      &array_header,
      iree_make_byte_span(array_payload, LOOM_XDNA_ELF_ARRAY_HEADER_SIZE)));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_records(
      program->array_records, program->array_record_count,
      first_tile_program_header_ordinal, program->tile_program_count,
      LOOM_XDNA_ELF_ARRAY_HEADER_SIZE, NULL,
      iree_make_byte_span(array_payload, array_byte_length)));

  const loom_xdna_elf_control_header_t control_header = {
      .abi_major = LOOM_XDNA_ELF_PROGRAM_ABI_MAJOR,
      .abi_minor = LOOM_XDNA_ELF_PROGRAM_ABI_MINOR,
      .record_count = (uint32_t)program->control_record_count,
      .byte_length = (uint32_t)control_byte_length,
      .flags = 0,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_control_header(
      &control_header,
      iree_make_byte_span(control_payload, LOOM_XDNA_ELF_CONTROL_HEADER_SIZE)));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_encode_records(
      program->control_records, program->control_record_count,
      first_tile_program_header_ordinal, program->tile_program_count,
      LOOM_XDNA_ELF_CONTROL_HEADER_SIZE, control_record_offsets,
      iree_make_byte_span(control_payload, control_byte_length)));

  for (iree_host_size_t i = 0; i < program->relocation_count; ++i) {
    const loom_aie2p_program_relocation_t* source = &program->relocations[i];
    if (source->target_record_index >= program->control_record_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P relocation record target is invalid");
    }
    const loom_aie2p_program_record_t* target =
        &program->control_records[source->target_record_index];
    if (target->type != LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P relocation target is not a block-write record");
    }
    const uint64_t target_byte_offset =
        (uint64_t)source->target_word_index * sizeof(uint32_t);
    const uint64_t block_byte_length =
        (uint64_t)target->value.register_block_write32.word_count *
        sizeof(uint32_t);
    if ((source->field_byte_width != 4 && source->field_byte_width != 8) ||
        target_byte_offset + source->field_byte_width > block_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P relocation field exceeds its block");
    }
    const uint64_t payload_byte_offset =
        (uint64_t)control_record_offsets[source->target_record_index] + 16 +
        target_byte_offset;
    if (payload_byte_offset > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P relocation payload offset overflows");
    }
    relocations[i] = (loom_xdna_elf_relocation_record_t){
        .target_program_header_ordinal = control_program_header_ordinal,
        .target_byte_offset = (uint32_t)payload_byte_offset,
        .binding_ordinal = source->binding_ordinal,
        .kind = source->kind,
        .field_byte_width = source->field_byte_width,
        .flags = 0,
        .addend = source->addend,
        .minimum_value = source->minimum_value,
        .maximum_value = source->maximum_value,
        .required_alignment = source->required_alignment,
    };
  }

  *out_program = (loom_aie2p_encoded_array_program_t){
      .array_payload =
          iree_make_const_byte_span(array_payload, array_byte_length),
      .control_payload =
          iree_make_const_byte_span(control_payload, control_byte_length),
      .relocations = relocations,
      .relocation_count = program->relocation_count,
  };
  return iree_ok_status();
}
