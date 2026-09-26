// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/array_report.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/storage.h"
#include "loom/util/fact_table.h"

typedef struct loom_aie2p_array_report_tile_usage_t {
  // Worker-private storage bytes on the selected tile.
  uint32_t worker_storage_byte_count;
  // Channel storage bytes on the selected tile.
  uint32_t channel_storage_byte_count;
  // Local-memory high-water extent relative to the tile's local base.
  uint32_t local_memory_byte_count;
  // Largest byte occupancy among the tile's local-memory banks.
  uint32_t maximum_bank_storage_byte_count;
} loom_aie2p_array_report_tile_usage_t;

static bool loom_aie2p_array_report_coordinate_equal(
    loom_xdna_tile_coordinate_t lhs, loom_xdna_tile_coordinate_t rhs) {
  return lhs.column == rhs.column && lhs.row == rhs.row;
}

static bool loom_aie2p_array_report_storage_present(
    const loom_aie2p_array_channel_storage_plan_t* storage) {
  return storage->owner.column != UINT16_MAX;
}

static bool loom_aie2p_array_report_storage_equal(
    const loom_aie2p_array_channel_storage_plan_t* lhs,
    const loom_aie2p_array_channel_storage_plan_t* rhs) {
  return loom_aie2p_array_report_coordinate_equal(lhs->owner, rhs->owner) &&
         lhs->owner_offset == rhs->owner_offset;
}

static uint32_t loom_aie2p_array_report_overlap_byte_count(
    uint32_t offset, uint32_t byte_count, uint32_t range_offset,
    uint32_t range_byte_count) {
  const uint64_t end = (uint64_t)offset + byte_count;
  const uint64_t range_end = (uint64_t)range_offset + range_byte_count;
  const uint64_t overlap_start =
      iree_max((uint64_t)offset, (uint64_t)range_offset);
  const uint64_t overlap_end = iree_min(end, range_end);
  return overlap_end > overlap_start ? (uint32_t)(overlap_end - overlap_start)
                                     : 0;
}

static uint32_t loom_aie2p_array_report_channel_storage_byte_count(
    const loom_aie2p_array_plan_t* plan, uint32_t channel_index) {
  uint32_t byte_count = 0;
  for (iree_host_size_t i = 0; i < plan->channel_slot_count; ++i) {
    const loom_aie2p_array_channel_slot_t* slot = &plan->channel_slots[i];
    if (slot->channel_index != channel_index) {
      continue;
    }
    const bool has_sender =
        loom_aie2p_array_report_storage_present(&slot->sender_storage);
    const bool has_receiver =
        loom_aie2p_array_report_storage_present(&slot->receiver_storage);
    if (has_sender &&
        plan->channels[channel_index].source_channel_index == channel_index) {
      byte_count += slot->byte_length;
    }
    if (has_receiver &&
        (!has_sender || !loom_aie2p_array_report_storage_equal(
                            &slot->sender_storage, &slot->receiver_storage))) {
      byte_count += slot->byte_length;
    }
  }
  return byte_count;
}

static void loom_aie2p_array_report_accumulate_channel_storage(
    const loom_aie2p_array_plan_t* plan, iree_host_size_t slot_index,
    loom_xdna_tile_coordinate_t coordinate, uint32_t range_offset,
    uint32_t range_byte_count, uint32_t* byte_count,
    uint32_t* range_overlap_byte_count, uint32_t* high_water_byte_count) {
  const loom_aie2p_array_channel_slot_t* slot =
      &plan->channel_slots[slot_index];
  const loom_aie2p_array_channel_storage_plan_t* storages[] = {
      &slot->sender_storage,
      &slot->receiver_storage,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(storages); ++i) {
    const loom_aie2p_array_channel_storage_plan_t* storage = storages[i];
    if ((i == 0 && plan->channels[slot->channel_index].source_channel_index !=
                       slot->channel_index) ||
        !loom_aie2p_array_report_coordinate_equal(storage->owner, coordinate) ||
        !loom_aie2p_array_report_storage_present(storage) ||
        (i != 0 &&
         loom_aie2p_array_report_storage_present(&slot->sender_storage) &&
         loom_aie2p_array_report_storage_equal(&slot->sender_storage,
                                               storage))) {
      continue;
    }
    if (byte_count != NULL) {
      *byte_count += slot->byte_length;
    }
    if (range_overlap_byte_count != NULL) {
      *range_overlap_byte_count += loom_aie2p_array_report_overlap_byte_count(
          storage->owner_offset, slot->byte_length, range_offset,
          range_byte_count);
    }
    if (high_water_byte_count != NULL) {
      const uint32_t allocation_end = storage->owner_offset + slot->byte_length;
      *high_water_byte_count = iree_max(*high_water_byte_count, allocation_end);
    }
  }
}

static void loom_aie2p_array_report_query_tile_usage(
    const loom_aie2p_array_plan_t* plan, loom_xdna_tile_coordinate_t coordinate,
    const loom_xdna_tile_memory_facts_t* memory,
    loom_aie2p_array_report_tile_usage_t* out_usage) {
  *out_usage = (loom_aie2p_array_report_tile_usage_t){0};
  uint32_t high_water = memory->local_base;
  for (iree_host_size_t i = 0; i < plan->worker_storage_count; ++i) {
    const loom_aie2p_array_worker_storage_plan_t* storage =
        &plan->worker_storage[i];
    if (!loom_aie2p_array_report_coordinate_equal(
            plan->workers[storage->worker_index].coordinate, coordinate)) {
      continue;
    }
    out_usage->worker_storage_byte_count += storage->byte_length;
    high_water =
        iree_max(high_water, storage->owner_offset + storage->byte_length);
  }
  for (iree_host_size_t i = 0; i < plan->worker_plan_count; ++i) {
    const loom_aie2p_array_worker_plan_t* worker = &plan->worker_plans[i];
    if (!loom_aie2p_array_report_coordinate_equal(worker->coordinate,
                                                  coordinate)) {
      continue;
    }
    out_usage->worker_storage_byte_count += worker->fold_state.byte_length;
    high_water = iree_max(high_water, worker->fold_state.owner_offset +
                                          worker->fold_state.byte_length);
  }
  for (iree_host_size_t i = 0; i < plan->channel_slot_count; ++i) {
    loom_aie2p_array_report_accumulate_channel_storage(
        plan, i, coordinate, /*range_offset=*/0,
        /*range_byte_count=*/0, &out_usage->channel_storage_byte_count,
        /*range_overlap_byte_count=*/NULL, &high_water);
  }
  out_usage->local_memory_byte_count = high_water - memory->local_base;

  const uint32_t bank_byte_count =
      memory->bank_count != 0 ? memory->local_capacity / memory->bank_count : 0;
  for (uint8_t bank = 0; bank < memory->bank_count; ++bank) {
    const uint32_t bank_offset = memory->local_base + bank * bank_byte_count;
    uint32_t occupied_byte_count = 0;
    for (iree_host_size_t i = 0; i < plan->worker_storage_count; ++i) {
      const loom_aie2p_array_worker_storage_plan_t* storage =
          &plan->worker_storage[i];
      if (!loom_aie2p_array_report_coordinate_equal(
              plan->workers[storage->worker_index].coordinate, coordinate)) {
        continue;
      }
      occupied_byte_count += loom_aie2p_array_report_overlap_byte_count(
          storage->owner_offset, storage->byte_length, bank_offset,
          bank_byte_count);
    }
    for (iree_host_size_t i = 0; i < plan->worker_plan_count; ++i) {
      const loom_aie2p_array_worker_plan_t* worker = &plan->worker_plans[i];
      if (!loom_aie2p_array_report_coordinate_equal(worker->coordinate,
                                                    coordinate)) {
        continue;
      }
      occupied_byte_count += loom_aie2p_array_report_overlap_byte_count(
          worker->fold_state.owner_offset, worker->fold_state.byte_length,
          bank_offset, bank_byte_count);
    }
    for (iree_host_size_t i = 0; i < plan->channel_slot_count; ++i) {
      loom_aie2p_array_report_accumulate_channel_storage(
          plan, i, coordinate, bank_offset, bank_byte_count,
          /*byte_count=*/NULL, &occupied_byte_count,
          /*high_water_byte_count=*/NULL);
    }
    out_usage->maximum_bank_storage_byte_count = iree_max(
        out_usage->maximum_bank_storage_byte_count, occupied_byte_count);
  }
}

static const loom_aie2p_array_endpoint_t* loom_aie2p_array_report_base_endpoint(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_endpoint_t* endpoint) {
  if (endpoint->binding_view_source_endpoint_index != UINT32_MAX) {
    return &plan->endpoints[endpoint->binding_view_source_endpoint_index];
  }
  return endpoint;
}

static loom_target_compile_report_pipeline_endpoint_t
loom_aie2p_array_report_endpoint(const loom_aie2p_array_plan_t* plan,
                                 uint32_t endpoint_index) {
  const loom_aie2p_array_endpoint_t* endpoint =
      loom_aie2p_array_report_base_endpoint(plan,
                                            &plan->endpoints[endpoint_index]);
  return (loom_target_compile_report_pipeline_endpoint_t){
      .owner = endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING
                   ? LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_BINDING
                   : LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_WORKER,
      .owner_index = endpoint->owner_index,
      .port = endpoint->port,
  };
}

static iree_string_view_t loom_aie2p_array_report_transport_name(
    loom_aie2p_array_channel_transport_t transport) {
  switch (transport) {
    case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA:
      return IREE_SV("external-dma");
    case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY:
      return IREE_SV("neighbor-memory");
    case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA:
      return IREE_SV("routed-dma");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_aie2p_array_report_fold_kind_name(
    loom_combining_kind_t kind) {
  static const iree_string_view_t names[LOOM_COMBINING_KIND_COUNT_] = {
      [LOOM_COMBINING_KIND_ADDI] = IREE_SVL("addi"),
      [LOOM_COMBINING_KIND_ADDF] = IREE_SVL("addf"),
      [LOOM_COMBINING_KIND_MULI] = IREE_SVL("muli"),
      [LOOM_COMBINING_KIND_MULF] = IREE_SVL("mulf"),
      [LOOM_COMBINING_KIND_MINSI] = IREE_SVL("minsi"),
      [LOOM_COMBINING_KIND_MAXSI] = IREE_SVL("maxsi"),
      [LOOM_COMBINING_KIND_MINUI] = IREE_SVL("minui"),
      [LOOM_COMBINING_KIND_MAXUI] = IREE_SVL("maxui"),
      [LOOM_COMBINING_KIND_ANDI] = IREE_SVL("andi"),
      [LOOM_COMBINING_KIND_ORI] = IREE_SVL("ori"),
      [LOOM_COMBINING_KIND_XORI] = IREE_SVL("xori"),
      [LOOM_COMBINING_KIND_MINIMUMF] = IREE_SVL("minimumf"),
      [LOOM_COMBINING_KIND_MAXIMUMF] = IREE_SVL("maximumf"),
      [LOOM_COMBINING_KIND_MINNUMF] = IREE_SVL("minnumf"),
      [LOOM_COMBINING_KIND_MAXNUMF] = IREE_SVL("maxnumf"),
  };
  return loom_combining_kind_is_valid(kind) ? names[kind] : IREE_SV("unknown");
}

static void loom_aie2p_array_report_worker_record_counts(
    const loom_aie2p_array_plan_t* plan, uint32_t worker_index,
    loom_target_compile_report_pipeline_worker_row_t* row) {
  row->ring_state_count = plan->worker_plans[worker_index].port_count;
  for (iree_host_size_t i = 0; i < plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &plan->channels[i];
    const loom_aie2p_array_endpoint_t* sender =
        &plan->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &plan->endpoints[channel->receiver_endpoint_index];
    if (receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
        receiver->owner_index == worker_index) {
      ++row->input_channel_count;
      row->input_record_count = channel->record_count;
    }
    if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
        sender->owner_index == worker_index) {
      ++row->output_channel_count;
      row->output_record_count = channel->record_count;
    }
  }
}

static void loom_aie2p_array_report_channel_storage(
    const loom_module_t* module, const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_channel_t* channel,
    loom_target_compile_report_pipeline_storage_t* out_storage) {
  const loom_aie2p_array_endpoint_t* external_endpoint = NULL;
  const loom_aie2p_array_endpoint_t* sender =
      &plan->endpoints[channel->sender_endpoint_index];
  const loom_aie2p_array_endpoint_t* receiver =
      &plan->endpoints[channel->receiver_endpoint_index];
  if (loom_aie2p_array_report_base_endpoint(plan, sender)->owner_kind ==
      LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
    external_endpoint = sender;
  } else if (loom_aie2p_array_report_base_endpoint(plan, receiver)
                 ->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
    external_endpoint = receiver;
  }
  if (external_endpoint == NULL) {
    return;
  }
  external_endpoint =
      loom_aie2p_array_report_base_endpoint(plan, external_endpoint);

  loom_value_fact_storage_schema_t schema = {0};
  if (!loom_encoding_query_type_storage_schema(
          /*context=*/NULL, module, external_endpoint->message_type, &schema) ||
      schema.static_spec_encoding_id == 0) {
    return;
  }
  const loom_encoding_t* encoding =
      loom_module_encoding(module, schema.static_spec_encoding_id);
  const loom_encoding_record_layout_t* record_layout = NULL;
  if (encoding == NULL ||
      !loom_encoding_query_static_record_layout(
          module, schema.static_spec_encoding_id, &record_layout)) {
    return;
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  loom_value_fact_address_layout_t address_layout = {0};
  iree_string_view_t address_layout_name = IREE_SV("unknown");
  if (loom_encoding_query_type_address_layout(
          /*context=*/NULL, module, external_endpoint->message_type,
          stride_storage, IREE_ARRAYSIZE(stride_storage), &address_layout)) {
    if (address_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
      address_layout_name = IREE_SV("dense");
    } else if (address_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
      address_layout_name = IREE_SV("strided");
    }
  }
  *out_storage = (loom_target_compile_report_pipeline_storage_t){
      .schema_name = loom_string_table_get(&module->strings, encoding->name_id),
      .address_layout = address_layout_name,
      .transform = IREE_SV("none"),
      .schema_logical_element_count =
          record_layout->geometry.logical_element_count,
      .schema_record_byte_count = record_layout->geometry.storage_byte_count,
      .schema_required_alignment = record_layout->geometry.required_alignment,
      .schema_records_per_channel_record =
          channel->record_byte_length /
          record_layout->geometry.storage_byte_count,
  };
}

static void loom_aie2p_array_report_channel_transfer(
    const loom_aie2p_array_plan_t* plan, uint32_t channel_index,
    loom_target_compile_report_pipeline_transfer_t* out_transfer) {
  *out_transfer = (loom_target_compile_report_pipeline_transfer_t){
      .binding_ordinal = UINT32_MAX,
      .partition_lane_count = 1,
  };
  for (iree_host_size_t i = 0; i < plan->binding_plan_count; ++i) {
    const loom_aie2p_array_binding_plan_t* binding_plan =
        &plan->binding_plans[i];
    if (binding_plan->channel_index != channel_index) {
      continue;
    }
    *out_transfer = (loom_target_compile_report_pipeline_transfer_t){
        .binding_ordinal = plan->bindings[binding_plan->binding_index].ordinal,
        .partition_lane = binding_plan->partition_lane,
        .partition_lane_count = binding_plan->partition_lane_count,
        .binding_byte_offset = binding_plan->binding_byte_offset,
        .binding_span_byte_count = binding_plan->binding_span_byte_length,
        .task_byte_count = binding_plan->transfer_byte_length,
        .task_repeat_count = binding_plan->task_repeat_count,
        .activation_byte_count = (uint64_t)binding_plan->transfer_byte_length *
                                 binding_plan->task_repeat_count,
    };
    return;
  }
}

static void loom_aie2p_array_report_channel_resources(
    const loom_aie2p_array_plan_t* plan, uint32_t channel_index,
    loom_target_compile_report_pipeline_channel_row_t* row) {
  for (iree_host_size_t i = 0; i < plan->lock_count; ++i) {
    row->hardware_lock_count += plan->locks[i].channel_index == channel_index;
  }
  for (iree_host_size_t i = 0; i < plan->dma_channel_count; ++i) {
    if (plan->dma_channels[i].channel_index != channel_index) {
      continue;
    }
    ++row->dma_channel_count;
    row->dma_buffer_descriptor_count +=
        plan->dma_channels[i].buffer_descriptor_count;
  }
  for (iree_host_size_t i = 0; i < plan->route_count; ++i) {
    row->route_count += plan->routes[i].channel_index == channel_index;
  }
}

iree_status_t loom_aie2p_array_report_record(
    const loom_module_t* module, iree_string_view_t root_name,
    const loom_aie2p_array_plan_t* plan, const loom_aie2p_xdna_tile_t* tiles,
    loom_target_compile_report_t* report,
    iree_arena_allocator_t* scratch_arena) {
  if (report == NULL) {
    return iree_ok_status();
  }

  const bool wants_rows = loom_target_compile_report_wants_details(
      report, LOOM_TARGET_COMPILE_REPORT_DETAIL_PIPELINE_PLAN_ROWS);
  loom_target_compile_report_pipeline_worker_row_t* worker_rows = NULL;
  loom_target_compile_report_pipeline_channel_row_t* channel_rows = NULL;
  if (wants_rows) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, plan->worker_count,
                                  sizeof(*worker_rows), (void**)&worker_rows));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, plan->channel_count, sizeof(*channel_rows),
        (void**)&channel_rows));
  }

  loom_target_compile_report_pipeline_plan_summary_t summary = {
      .root_name = root_name,
      .realization = IREE_SV("spatial-program"),
      .group_count = (uint32_t)plan->group_count,
      .worker_count = (uint32_t)plan->worker_count,
      .binding_count = plan->binding_slot_count,
      .channel_count = (uint32_t)plan->channel_count,
      .channel_slot_count = (uint32_t)plan->channel_slot_count,
      .hardware_lock_count = (uint32_t)plan->lock_count,
      .dma_channel_count = (uint32_t)plan->dma_channel_count,
      .route_count = (uint32_t)plan->route_count,
  };
  for (iree_host_size_t i = 0; i < plan->worker_storage_count; ++i) {
    summary.worker_storage_byte_count += plan->worker_storage[i].byte_length;
  }
  for (iree_host_size_t i = 0; i < plan->worker_plan_count; ++i) {
    summary.worker_storage_byte_count +=
        plan->worker_plans[i].fold_state.byte_length;
  }
  for (iree_host_size_t i = 0; i < plan->channel_count; ++i) {
    summary.channel_storage_byte_count +=
        loom_aie2p_array_report_channel_storage_byte_count(plan, (uint32_t)i);
  }
  for (iree_host_size_t i = 0; i < plan->dma_channel_count; ++i) {
    summary.dma_buffer_descriptor_count +=
        plan->dma_channels[i].buffer_descriptor_count;
  }
  for (iree_host_size_t i = 0; i < plan->binding_plan_count; ++i) {
    summary.external_dma_byte_count +=
        (uint64_t)plan->binding_plans[i].transfer_byte_length *
        plan->binding_plans[i].task_repeat_count;
  }

  bool has_code_headroom = false;
  bool has_local_headroom = false;
  for (iree_host_size_t i = 0; i < plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &plan->workers[i];
    const loom_aie2p_leaf_realization_t* realization =
        &tiles[i].contribution->realization;
    const loom_xdna_tile_facts_t* tile_facts =
        loom_xdna_array_tile_facts(plan->family, worker->coordinate);
    loom_aie2p_array_report_tile_usage_t usage = {0};
    loom_aie2p_array_report_query_tile_usage(plan, worker->coordinate,
                                             &tile_facts->memory, &usage);

    const uint32_t code_byte_count = (uint32_t)realization->code.byte_length;
    const uint32_t code_headroom =
        tile_facts->memory.program_capacity - code_byte_count;
    const uint32_t local_headroom =
        tile_facts->memory.local_capacity - usage.local_memory_byte_count;
    summary.worker_code_byte_count += code_byte_count;
    summary.maximum_worker_code_byte_count =
        iree_max(summary.maximum_worker_code_byte_count, code_byte_count);
    summary.minimum_worker_code_headroom_byte_count =
        has_code_headroom
            ? iree_min(summary.minimum_worker_code_headroom_byte_count,
                       code_headroom)
            : code_headroom;
    has_code_headroom = true;
    summary.maximum_tile_local_memory_byte_count =
        iree_max(summary.maximum_tile_local_memory_byte_count,
                 usage.local_memory_byte_count);
    summary.minimum_tile_local_memory_headroom_byte_count =
        has_local_headroom
            ? iree_min(summary.minimum_tile_local_memory_headroom_byte_count,
                       local_headroom)
            : local_headroom;
    has_local_headroom = true;
    if (usage.maximum_bank_storage_byte_count >=
        summary.maximum_bank_storage_byte_count) {
      summary.maximum_bank_storage_byte_count =
          usage.maximum_bank_storage_byte_count;
      summary.bank_storage_capacity_byte_count =
          tile_facts->memory.bank_count != 0
              ? tile_facts->memory.local_capacity /
                    tile_facts->memory.bank_count
              : 0;
    }

    if (wants_rows) {
      worker_rows[i] = (loom_target_compile_report_pipeline_worker_row_t){
          .worker_index = (uint32_t)i,
          .group_index = worker->group_index,
          .lane = worker->lane,
          .flags = LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_RESIDENT |
                   LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_PLACED,
          .entry_name = loom_low_diagnostic_symbol_name(module, worker->entry),
          .placement =
              {
                  .rank = 2,
                  .x = worker->coordinate.column,
                  .y = worker->coordinate.row,
              },
          .finite_loop_trip_count = worker->fold_record_count,
          .code_byte_count = code_byte_count,
          .code_capacity_byte_count = tile_facts->memory.program_capacity,
          .worker_storage_byte_count = usage.worker_storage_byte_count,
          .channel_storage_byte_count = usage.channel_storage_byte_count,
          .local_memory_byte_count = usage.local_memory_byte_count,
          .local_memory_capacity_byte_count = tile_facts->memory.local_capacity,
          .maximum_bank_storage_byte_count =
              usage.maximum_bank_storage_byte_count,
          .bank_storage_capacity_byte_count =
              tile_facts->memory.bank_count != 0
                  ? tile_facts->memory.local_capacity /
                        tile_facts->memory.bank_count
                  : 0,
      };
      if (worker->fold_record_count != 0) {
        worker_rows[i].flags |=
            LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_FOLDED;
        worker_rows[i].fold_kind =
            loom_aie2p_array_report_fold_kind_name(worker->fold_kind);
      }
      loom_aie2p_array_report_worker_record_counts(plan, (uint32_t)i,
                                                   &worker_rows[i]);
    }
  }

  for (iree_host_size_t i = 0; i < plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &plan->channels[i];
    const uint64_t activation_byte_count =
        (uint64_t)channel->record_byte_length * channel->record_count;
    if (channel->transport == LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA) {
      summary.routed_dma_byte_count += activation_byte_count;
    }
    if (!wants_rows) {
      continue;
    }
    channel_rows[i] = (loom_target_compile_report_pipeline_channel_row_t){
        .channel_index = (uint32_t)i,
        .transport = loom_aie2p_array_report_transport_name(channel->transport),
        .sender = loom_aie2p_array_report_endpoint(
            plan, channel->sender_endpoint_index),
        .receiver = loom_aie2p_array_report_endpoint(
            plan, channel->receiver_endpoint_index),
        .capacity = channel->capacity,
        .record_count = channel->record_count,
        .record_byte_count = channel->record_byte_length,
        .activation_byte_count = activation_byte_count,
        .local_storage_byte_count =
            loom_aie2p_array_report_channel_storage_byte_count(plan,
                                                               (uint32_t)i),
    };
    loom_aie2p_array_report_channel_resources(plan, (uint32_t)i,
                                              &channel_rows[i]);
    loom_aie2p_array_report_channel_storage(module, plan, channel,
                                            &channel_rows[i].storage);
    loom_aie2p_array_report_channel_transfer(plan, (uint32_t)i,
                                             &channel_rows[i].transfer);
  }

  const loom_target_compile_report_pipeline_plan_t report_plan = {
      .summary = summary,
      .worker_rows = worker_rows,
      .worker_row_count = wants_rows ? plan->worker_count : 0,
      .channel_rows = channel_rows,
      .channel_row_count = wants_rows ? plan->channel_count : 0,
  };
  return loom_target_compile_report_record_pipeline_plan(report, &report_plan);
}
