// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/array/abi_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/array/binding.h"
#include "loom/target/arch/amd/xdna/aie2p/array/channel_resources.h"
#include "loom/target/arch/amd/xdna/aie2p/array/local_memory.h"
#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/array/topology.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/util/fact_table.h"

typedef enum loom_aie2p_array_entity_kind_e {
  LOOM_AIE2P_ARRAY_ENTITY_GROUP = 0,
  LOOM_AIE2P_ARRAY_ENTITY_BINDING = 1,
  LOOM_AIE2P_ARRAY_ENTITY_WORKER = 2,
  LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT = 3,
} loom_aie2p_array_entity_kind_t;

typedef enum loom_aie2p_array_worker_rate_e {
  LOOM_AIE2P_ARRAY_WORKER_RATE_RECORDWISE = 0,
  LOOM_AIE2P_ARRAY_WORKER_RATE_FOLDED = 1,
} loom_aie2p_array_worker_rate_t;

typedef struct loom_aie2p_array_entity_t {
  // Locally defined topology result admitted by array Low verification.
  loom_value_id_t value_id;
  // Logical topology table containing the indexed row.
  loom_aie2p_array_entity_kind_t kind;
  // Row in the table selected by kind.
  uint32_t index;
} loom_aie2p_array_entity_t;

typedef struct loom_aie2p_array_tile_state_t {
  // Local-memory, DMA, lock, and lifecycle allocation state.
  loom_aie2p_array_tile_resources_t resources;
  // Retained shim completion route, or UINT32_MAX before first use.
  uint32_t completion_route_index;
} loom_aie2p_array_tile_state_t;

typedef struct loom_aie2p_array_source_endpoint_t {
  // Physical tile containing the retained sending DMA.
  loom_xdna_tile_coordinate_t coordinate;
  // Direction-local sending DMA channel ordinal.
  uint8_t dma_channel;
  // Uncommitted source transition, or NULL for an allocated multicast source.
  const loom_aie2p_array_compute_endpoint_proposal_t* proposal;
} loom_aie2p_array_source_endpoint_t;

// Builder-row indices assigned by one infallible endpoint commit.
typedef struct loom_aie2p_array_committed_endpoint_t {
  // Appended DMA-plan row.
  uint32_t dma_index;
  // First appended lock-plan row.
  uint32_t credit_lock_index;
} loom_aie2p_array_committed_endpoint_t;

// Result-table extents admitted before arena allocation. Fixed-row tables use
// exact counts; routing uses a topology-derived upper bound until selection.
typedef struct loom_aie2p_array_physical_cardinalities_t {
  // Function-local worker storage rows.
  uint32_t worker_storage_count;
  // Worker read-only data placement rows.
  uint32_t read_only_data_count;
  // Active worker ABI port rows.
  uint32_t worker_port_count;
  // Channel ring slot rows.
  uint32_t channel_slot_count;
  // Hardware lock rows.
  uint32_t lock_count;
  // Compute and shim DMA endpoint rows.
  uint32_t dma_channel_count;
  // Worst-case stream route rows before physical selection.
  uint32_t route_capacity;
  // Host binding patch rows.
  uint32_t binding_plan_count;
} loom_aie2p_array_physical_cardinalities_t;

// Aggregate physical demand accumulated in source-channel order.
typedef struct loom_aie2p_array_physical_demands_t {
  // Compute memory-to-stream DMA channels.
  uint32_t compute_memory_to_stream_channels;
  // Compute stream-to-memory DMA channels.
  uint32_t compute_stream_to_memory_channels;
  // Compute DMA buffer descriptors.
  uint32_t compute_buffer_descriptors;
  // Compute hardware locks.
  uint32_t compute_locks;
  // Shim memory-to-stream DMA channels.
  uint32_t shim_memory_to_stream_channels;
  // Shim stream-to-memory DMA channels.
  uint32_t shim_stream_to_memory_channels;
  // Shim DMA buffer descriptors.
  uint32_t shim_buffer_descriptors;
} loom_aie2p_array_physical_demands_t;

typedef struct loom_aie2p_array_plan_builder_t {
  const loom_module_t* module;
  const loom_op_t* function_op;
  const loom_low_descriptor_set_t* descriptor_set;
  const loom_xdna_array_family_t* family;
  const loom_aie2p_array_leaf_t* leaves;
  iree_host_size_t leaf_count;
  // Caller-owned sink for structured failures in authored array topology.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Whether private construction still satisfies authored target admission.
  bool valid;
  // Whether the Low ABI explicitly fixes the dense binding-table cardinality.
  bool has_explicit_binding_slot_count;
  iree_arena_allocator_t* arena;
  loom_value_fact_table_t facts;
  loom_aie2p_array_plan_t* plan;

  loom_aie2p_array_group_t* groups;
  loom_aie2p_array_binding_t* bindings;
  loom_aie2p_array_worker_t* workers;
  loom_aie2p_array_endpoint_t* endpoints;
  loom_aie2p_array_channel_t* channels;
  loom_aie2p_array_entity_t* entities;
  iree_host_size_t entity_capacity;

  loom_aie2p_array_worker_plan_t* worker_plans;
  loom_aie2p_array_worker_storage_plan_t* worker_storage;
  loom_aie2p_array_read_only_data_plan_t* read_only_data;
  loom_aie2p_array_worker_port_plan_t* worker_ports;
  // Source-resource ordinals mapped to the corresponding physical port row.
  uint32_t* worker_resource_ports;
  // Fold-output ordinals mapped to worker-local resident state ordinals.
  uint32_t* worker_fold_output_states;
  loom_aie2p_array_channel_slot_t* channel_slots;
  loom_aie2p_array_lock_plan_t* locks;
  loom_aie2p_array_dma_plan_t* dma_channels;
  loom_aie2p_array_binding_plan_t* binding_plans;
  // Retained completion resources sharing the binding-plan allocation.
  loom_aie2p_array_completion_route_t* completion_routes;
  loom_aie2p_array_route_builder_t route_builder;

  iree_host_size_t group_cursor;
  iree_host_size_t binding_cursor;
  iree_host_size_t worker_cursor;
  iree_host_size_t endpoint_cursor;
  iree_host_size_t channel_cursor;
  iree_host_size_t worker_storage_cursor;
  iree_host_size_t read_only_data_cursor;
  iree_host_size_t worker_port_cursor;
  iree_host_size_t channel_slot_cursor;
  iree_host_size_t lock_cursor;
  iree_host_size_t dma_channel_cursor;
  iree_host_size_t binding_plan_cursor;

  loom_aie2p_array_tile_state_t* tile_states;
} loom_aie2p_array_plan_builder_t;

static iree_status_t loom_aie2p_array_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static bool loom_aie2p_array_symbol_ref_equal(loom_symbol_ref_t lhs,
                                              loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static const loom_aie2p_array_leaf_t* loom_aie2p_array_find_leaf(
    const loom_aie2p_array_plan_builder_t* builder, loom_symbol_ref_t entry) {
  const loom_aie2p_array_leaf_t* result = NULL;
  for (iree_host_size_t i = 0; i < builder->leaf_count; ++i) {
    if (loom_aie2p_array_symbol_ref_equal(builder->leaves[i].entry, entry)) {
      IREE_ASSERT(result == NULL, "leaf table entries must be unique");
      result = &builder->leaves[i];
    }
  }
  return result;
}

static uint64_t loom_aie2p_array_constant(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id) {
  // Closed array vocabulary admits only constant.u32/u64 as scalar producers.
  // Their generated immediate domains and shared fact callback establish the
  // exact nonnegative value before topology extraction.
  loom_value_fact_uniform_element_t uniform = {0};
  const bool found = loom_value_facts_query_uniform_element(
      &builder->facts.context,
      loom_value_fact_table_lookup(&builder->facts, value_id), &uniform);
  IREE_ASSERT(found, "admitted array constants carry uniform integer facts");
  (void)found;
  return (uint64_t)uniform.element.range_lo;
}

static loom_type_t loom_aie2p_array_result_value_type(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_type_t register_type =
      loom_module_value_type(module, loom_op_results(op)[0]);
  return *loom_type_register_value_type(register_type);
}

// Fibonacci hashing mixes the otherwise nearly sequential SSA value IDs for
// the power-of-two compact entity table.
static uint32_t loom_aie2p_array_hash_value_id(loom_value_id_t value_id) {
  return (uint32_t)value_id * 2654435769u;
}

static iree_host_size_t loom_aie2p_array_entity_slot(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id) {
  iree_host_size_t slot =
      loom_aie2p_array_hash_value_id(value_id) & (builder->entity_capacity - 1);
  while (builder->entities[slot].value_id != LOOM_VALUE_ID_INVALID &&
         builder->entities[slot].value_id != value_id) {
    slot = (slot + 1) & (builder->entity_capacity - 1);
  }
  return slot;
}

static const loom_aie2p_array_entity_t* loom_aie2p_array_lookup_entity(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id) {
  // Empty signatures, isolated single-block bodies, and descriptor operand
  // classes prove every topology reference has an earlier local producer.
  return &builder->entities[loom_aie2p_array_entity_slot(builder, value_id)];
}

static void loom_aie2p_array_define_entity(
    loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id,
    loom_aie2p_array_entity_kind_t kind, uint32_t index) {
  loom_aie2p_array_entity_t* entity =
      &builder->entities[loom_aie2p_array_entity_slot(builder, value_id)];
  *entity = (loom_aie2p_array_entity_t){
      .value_id = value_id,
      .kind = kind,
      .index = index,
  };
}

static void loom_aie2p_array_count_topology(
    loom_aie2p_array_plan_builder_t* builder, const loom_block_t* block) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(builder->descriptor_set, op, &packet);
    if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
      continue;
    }
    switch (packet.descriptor_ordinal) {
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTANT_U32:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTANT_U64:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTRAIN_LOCATION:
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_GROUP:
        ++builder->plan->group_count;
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_BINDING:
        ++builder->plan->binding_count;
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER_FOLD:
        ++builder->plan->worker_count;
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_RECEIVER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_RECEIVER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_RECEIVER:
        ++builder->plan->endpoint_count;
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CHANNEL:
        ++builder->plan->channel_count;
        break;
      default:
        IREE_ASSERT_UNREACHABLE("generated array descriptor ordinal");
        break;
    }
  }
}

static iree_status_t loom_aie2p_array_allocate_topology(
    loom_aie2p_array_plan_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->group_count, sizeof(*builder->groups),
      (void**)&builder->groups));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->binding_count, sizeof(*builder->bindings),
      (void**)&builder->bindings));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->worker_count, sizeof(*builder->workers),
      (void**)&builder->workers));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->endpoint_count,
      sizeof(*builder->endpoints), (void**)&builder->endpoints));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->channel_count, sizeof(*builder->channels),
      (void**)&builder->channels));

  iree_host_size_t entity_count = 0;
  if (!iree_host_size_checked_add(builder->plan->group_count,
                                  builder->plan->binding_count,
                                  &entity_count) ||
      !iree_host_size_checked_add(entity_count, builder->plan->worker_count,
                                  &entity_count) ||
      !iree_host_size_checked_add(entity_count, builder->plan->endpoint_count,
                                  &entity_count) ||
      !iree_host_size_checked_mul(entity_count, 2, &entity_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P array entity table is too large");
  }
  builder->entity_capacity =
      iree_host_size_next_power_of_two(iree_max(8, entity_count));
  if (!iree_host_size_is_power_of_two(builder->entity_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P array entity table is too large");
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->entity_capacity, sizeof(*builder->entities),
      (void**)&builder->entities));
  for (iree_host_size_t i = 0; i < builder->entity_capacity; ++i) {
    builder->entities[i].value_id = LOOM_VALUE_ID_INVALID;
  }
  builder->plan->groups = builder->groups;
  builder->plan->bindings = builder->bindings;
  builder->plan->workers = builder->workers;
  builder->plan->endpoints = builder->endpoints;
  builder->plan->channels = builder->channels;
  return iree_ok_status();
}

static void loom_aie2p_array_extract_group(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  loom_aie2p_array_group_t* group = &builder->groups[builder->group_cursor];
  group->value_id = loom_op_results(op)[0];
  group->lane_count =
      (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[0]);
  loom_aie2p_array_define_entity(builder, group->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_GROUP,
                                 (uint32_t)builder->group_cursor++);
}

static void loom_aie2p_array_extract_binding(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  loom_aie2p_array_binding_t* binding =
      &builder->bindings[builder->binding_cursor];
  const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  binding->value_id = loom_op_results(op)[0];
  binding->ordinal =
      (uint32_t)loom_aie2p_array_array_binding_ordinal(attrs).i64;
  binding->access =
      (loom_aie2p_array_binding_access_t)loom_aie2p_array_array_binding_access(
          attrs)
          .i64;
  if (!builder->has_explicit_binding_slot_count) {
    const uint64_t required_slot_count = (uint64_t)binding->ordinal + 1u;
    builder->plan->binding_slot_count = iree_max(
        builder->plan->binding_slot_count,
        (uint32_t)iree_min(required_slot_count,
                           (uint64_t)LOOM_AIE2P_ARRAY_MAX_BINDING_SLOT_COUNT));
  }
  loom_aie2p_array_define_entity(builder, binding->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_BINDING,
                                 (uint32_t)builder->binding_cursor++);
}

static iree_status_t loom_aie2p_array_extract_worker(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op,
    loom_aie2p_array_worker_rate_t rate) {
  loom_aie2p_array_worker_t* worker = &builder->workers[builder->worker_cursor];
  worker->value_id = loom_op_results(op)[0];
  worker->group_index =
      loom_aie2p_array_lookup_entity(builder, loom_op_operands(op)[0])->index;
  worker->lane =
      (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[1]);
  const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  worker->entry = (rate == LOOM_AIE2P_ARRAY_WORKER_RATE_FOLDED
                       ? loom_aie2p_array_array_worker_fold_entry(attrs)
                       : loom_aie2p_array_array_worker_entry(attrs))
                      .symbol;
  // Array IR can retain declarations until linking. Materialization consumes
  // a local Low body whose resource imports are the resident worker's ABI.
  const loom_symbol_t* entry_symbol =
      &builder->module->symbols.entries[worker->entry.symbol_id];
  const iree_string_view_t entry_name =
      loom_string_table_get(&builder->module->strings, entry_symbol->name_id);
  const loom_op_t* entry_op = entry_symbol->defining_op;
  if (!loom_low_func_def_isa(entry_op)) {
    const loom_diagnostic_param_t params[] = {loom_param_string(entry_name)};
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_XDNA_016,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    builder->valid = false;
    return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
  }
  const loom_func_like_t entry =
      loom_func_like_const_cast(builder->module, entry_op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(entry, &argument_count);
  if (argument_count != 0 || entry_op->result_count != 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(entry_name),
        loom_param_string(IREE_SV("aie2p-resident")),
        loom_param_string(argument_count != 0 ? IREE_SV("register argument")
                                              : IREE_SV("register result")),
        loom_param_u32(argument_count != 0 ? argument_count
                                           : entry_op->result_count),
        loom_param_u32(0),
    };
    const loom_diagnostic_emission_t emission = {
        .op = entry_op,
        .error = LOOM_ERR_TARGET_054,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    builder->valid = false;
    return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
  }
  worker->leaf = loom_aie2p_array_find_leaf(builder, worker->entry);
  IREE_ASSERT(worker->leaf != NULL,
              "verified resident worker must select a core leaf");
  worker->fold_record_count = 0;
  worker->fold_output_port = 0;
  worker->fold_output_count = 0;
  worker->fold_kind = LOOM_COMBINING_KIND_ADDI;
  worker->fold_fast_math_flags = 0;
  worker->active_endpoint_count = 0;
  if (rate == LOOM_AIE2P_ARRAY_WORKER_RATE_FOLDED) {
    worker->fold_record_count =
        (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[2]);
    if (worker->fold_record_count == 0) {
      const loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(worker->fold_record_count),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 2)),
      };
      const loom_diagnostic_emission_t emission = {
          .op = op,
          .error = LOOM_ERR_XDNA_001,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      builder->valid = false;
      return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
    }
    worker->fold_output_port =
        (uint32_t)loom_aie2p_array_array_worker_fold_output_port(attrs).i64;
    worker->fold_output_count =
        (uint32_t)loom_aie2p_array_array_worker_fold_output_count(attrs).i64;
    worker->fold_kind =
        (loom_combining_kind_t)loom_aie2p_array_array_worker_fold_kind(attrs)
            .i64;
    worker->fold_fast_math_flags =
        (uint8_t)loom_aie2p_array_array_worker_fold_fast_math(attrs).i64;
  }
  worker->coordinate = (loom_xdna_tile_coordinate_t){
      .column = UINT16_MAX,
      .row = UINT16_MAX,
  };
  loom_aie2p_array_define_entity(builder, worker->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_WORKER,
                                 (uint32_t)builder->worker_cursor++);
  return iree_ok_status();
}

static void loom_aie2p_array_extract_endpoint(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op,
    loom_aie2p_array_endpoint_direction_t direction) {
  loom_aie2p_array_endpoint_t* endpoint =
      &builder->endpoints[builder->endpoint_cursor];
  endpoint->value_id = loom_op_results(op)[0];
  endpoint->direction = direction;
  const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  endpoint->port =
      (uint32_t)(direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND
                     ? loom_aie2p_array_array_sender_port(attrs)
                     : loom_aie2p_array_array_receiver_port(attrs))
          .i64;
  endpoint->message_type =
      loom_aie2p_array_result_value_type(builder->module, op);
  endpoint->first_channel_index = UINT32_MAX;
  endpoint->channel_use_count = 0;
  endpoint->worker_resource_ordinal = UINT32_MAX;
  endpoint->binding_view_source_endpoint_index = UINT32_MAX;
  endpoint->binding_view_record_count = 0;
  endpoint->binding_byte_offset = 0;
  endpoint->binding_view_partitioned = false;
  endpoint->partition_lane = 0;
  endpoint->partition_lane_count = 1;

  const loom_value_id_t owner_value = loom_op_operands(op)[0];
  const loom_aie2p_array_entity_t* owner =
      loom_aie2p_array_lookup_entity(builder, owner_value);
  endpoint->owner_kind = owner->kind == LOOM_AIE2P_ARRAY_ENTITY_BINDING
                             ? LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING
                             : LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER;
  endpoint->owner_index = owner->index;
  loom_aie2p_array_define_entity(builder, endpoint->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
                                 (uint32_t)builder->endpoint_cursor++);
}

static void loom_aie2p_array_extract_binding_view(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op,
    bool partitioned) {
  loom_aie2p_array_endpoint_t* endpoint =
      &builder->endpoints[builder->endpoint_cursor];
  endpoint->value_id = loom_op_results(op)[0];
  endpoint->binding_view_source_endpoint_index =
      loom_aie2p_array_lookup_entity(builder, loom_op_operands(op)[0])->index;
  const loom_aie2p_array_endpoint_t* source =
      &builder->endpoints[endpoint->binding_view_source_endpoint_index];
  endpoint->direction = source->direction;
  endpoint->owner_kind = source->owner_kind;
  endpoint->owner_index = source->owner_index;
  endpoint->port = source->port;
  endpoint->message_type =
      loom_aie2p_array_result_value_type(builder->module, op);
  endpoint->first_channel_index = UINT32_MAX;
  endpoint->channel_use_count = 0;
  endpoint->worker_resource_ordinal = UINT32_MAX;
  endpoint->binding_byte_offset =
      loom_aie2p_array_constant(builder, loom_op_operands(op)[1]);
  endpoint->binding_view_record_count = 0;
  endpoint->binding_view_partitioned = partitioned;
  endpoint->partition_lane = 0;
  endpoint->partition_lane_count = 1;
  if (partitioned) {
    endpoint->partition_lane =
        (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[2]);
    endpoint->partition_lane_count =
        (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[3]);
  }
  loom_aie2p_array_define_entity(builder, endpoint->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
                                 (uint32_t)builder->endpoint_cursor++);
}

static void loom_aie2p_array_extract_channel(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  const uint32_t channel_index = (uint32_t)builder->channel_cursor;
  loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  channel->value_id = loom_op_results(op)[0];
  channel->source_channel_index = channel_index;
  channel->binding_plan_index = UINT32_MAX;
  channel->first_channel_slot = UINT32_MAX;
  channel->sender_dma_index = UINT32_MAX;
  channel->sender_endpoint_index =
      loom_aie2p_array_lookup_entity(builder, loom_op_operands(op)[0])->index;
  channel->receiver_endpoint_index =
      loom_aie2p_array_lookup_entity(builder, loom_op_operands(op)[1])->index;
  channel->capacity =
      (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[2]);
  channel->record_count =
      (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[3]);
  channel->record_byte_length = 0;
  channel->neighbor_receiver_load_address_base = 0;
  channel->resource_flags = 0;
  ++builder->channel_cursor;
}

static iree_status_t loom_aie2p_array_extract_location(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  const uint32_t worker_index =
      loom_aie2p_array_lookup_entity(builder, loom_op_operands(op)[0])->index;
  loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
  if (worker->coordinate.column != UINT16_MAX) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(worker_index),
        loom_param_u32(2),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_XDNA_020,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    builder->valid = false;
    return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
  }
  const uint32_t column =
      (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[1]);
  const uint32_t row =
      (uint32_t)loom_aie2p_array_constant(builder, loom_op_operands(op)[2]);
  if (column >= builder->family->column_count ||
      row >= builder->family->row_count) {
    builder->valid = false;
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(column),
        loom_param_u32(row),
        loom_param_u32(builder->family->column_count),
        loom_param_u32(builder->family->row_count),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_XDNA_013,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
  }
  const loom_xdna_tile_coordinate_t coordinate = {
      .column = (uint16_t)column,
      .row = (uint16_t)row,
  };
  const loom_xdna_tile_facts_t* tile_facts =
      loom_xdna_array_tile_facts(builder->family, coordinate);
  if (tile_facts->kind != LOOM_XDNA_TILE_KIND_COMPUTE) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(worker_index),
        loom_param_u32(column),
        loom_param_u32(row),
        loom_param_string(IREE_SV("the coordinate is not a compute tile")),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_XDNA_021,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    builder->valid = false;
    return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
  }
  for (iree_host_size_t i = 0; i < builder->worker_cursor; ++i) {
    const loom_aie2p_array_worker_t* other = &builder->workers[i];
    if (i == worker_index || other->coordinate.column != coordinate.column ||
        other->coordinate.row != coordinate.row) {
      continue;
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(worker_index),
        loom_param_u32(column),
        loom_param_u32(row),
        loom_param_string(
            IREE_SV("another resident worker already occupies the tile")),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_XDNA_021,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    builder->valid = false;
    return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
  }
  worker->coordinate = coordinate;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_topology(
    loom_aie2p_array_plan_builder_t* builder, const loom_block_t* block) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (!builder->valid) {
      break;
    }
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(builder->descriptor_set, op, &packet);
    if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
      continue;
    }
    switch (packet.descriptor_ordinal) {
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTANT_U32:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTANT_U64:
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_GROUP: {
        loom_aie2p_array_extract_group(builder, op);
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_BINDING: {
        loom_aie2p_array_extract_binding(builder, op);
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_worker(
            builder, op, LOOM_AIE2P_ARRAY_WORKER_RATE_RECORDWISE));
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER_FOLD: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_worker(
            builder, op, LOOM_AIE2P_ARRAY_WORKER_RATE_FOLDED));
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER: {
        loom_aie2p_array_extract_endpoint(
            builder, op, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND);
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_RECEIVER: {
        loom_aie2p_array_extract_endpoint(
            builder, op, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE);
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_RECEIVER: {
        loom_aie2p_array_extract_binding_view(builder, op, false);
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_RECEIVER: {
        loom_aie2p_array_extract_binding_view(builder, op, true);
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CHANNEL: {
        loom_aie2p_array_extract_channel(builder, op);
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTRAIN_LOCATION: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_location(builder, op));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE("generated array descriptor ordinal");
        break;
    }
  }
  return iree_ok_status();
}

static loom_aie2p_array_tile_state_t* loom_aie2p_array_tile_state(
    loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate) {
  const iree_host_size_t index =
      (iree_host_size_t)coordinate.row * builder->family->column_count +
      coordinate.column;
  return &builder->tile_states[index];
}

static bool loom_aie2p_array_coordinates_equal(
    loom_xdna_tile_coordinate_t lhs, loom_xdna_tile_coordinate_t rhs) {
  return lhs.column == rhs.column && lhs.row == rhs.row;
}

static bool loom_aie2p_array_try_compute_endpoint(
    loom_aie2p_array_plan_builder_t* builder,
    const loom_aie2p_array_source_endpoint_t* source_endpoint,
    const loom_aie2p_array_compute_endpoint_request_t* request,
    loom_aie2p_array_compute_endpoint_proposal_t* out_proposal) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, request->coordinate);
  loom_aie2p_array_compute_endpoint_request_t candidate = *request;
  const loom_aie2p_array_compute_endpoint_proposal_t* predecessor = NULL;
  if (source_endpoint != NULL &&
      loom_aie2p_array_coordinates_equal(source_endpoint->coordinate,
                                         request->coordinate)) {
    candidate.flags =
        LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_REQUEST_FLAG_REQUIRE_LOOPBACK;
    candidate.loopback_source_dma_channel = source_endpoint->dma_channel;
    predecessor = source_endpoint->proposal;
  }
  return loom_aie2p_array_channel_resources_propose_compute(
             &state->resources, predecessor, &candidate, out_proposal) ==
         LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE;
}

// Selects a complete compute DMA endpoint whose memory and locks are directly
// visible to |worker_coordinate|. The worker's own tile remains first. Visible
// worker-free tiles precede occupied tiles so resident workers retain their
// local DMA capacity. A selected proposal retains every cursor transition and
// load-window base needed by infallible commit.
static bool loom_aie2p_array_select_compute_endpoint(
    loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t worker_coordinate,
    loom_aie2p_array_dma_direction_t direction, uint8_t record_count,
    uint32_t record_byte_length,
    const loom_aie2p_array_source_endpoint_t* source_endpoint,
    loom_aie2p_array_compute_endpoint_proposal_t* out_proposal) {
  loom_aie2p_array_tile_state_t* worker_state =
      loom_aie2p_array_tile_state(builder, worker_coordinate);
  loom_aie2p_array_compute_endpoint_request_t request = {
      .coordinate = worker_coordinate,
      .load_address_base =
          worker_state->resources.facts->memory.local_load_base,
      .record_count = record_count,
      .record_byte_length = record_byte_length,
      .direction = direction,
  };
  if (loom_aie2p_array_try_compute_endpoint(builder, source_endpoint, &request,
                                            out_proposal)) {
    return true;
  }

  for (uint8_t worker_pass = 0; worker_pass < 2; ++worker_pass) {
    const bool select_worker_tiles = worker_pass != 0;
    for (uint8_t i = 0; i < worker_state->resources.facts->memory.window_count;
         ++i) {
      const loom_xdna_address_window_t* window =
          &builder->family->address_windows
               [worker_state->resources.facts->memory.window_start + i];
      if (window->owner_kind != LOOM_XDNA_TILE_KIND_COMPUTE ||
          (window->owner_column_delta == 0 && window->owner_row_delta == 0)) {
        continue;
      }
      const int32_t candidate_column =
          (int32_t)worker_coordinate.column + window->owner_column_delta;
      const int32_t candidate_row =
          (int32_t)worker_coordinate.row + window->owner_row_delta;
      if (candidate_column < 0 ||
          candidate_column >= builder->family->column_count ||
          candidate_row < 0 || candidate_row >= builder->family->row_count) {
        continue;
      }
      request.coordinate = (loom_xdna_tile_coordinate_t){
          .column = (uint16_t)candidate_column,
          .row = (uint16_t)candidate_row,
      };
      request.load_address_base = window->base;
      const loom_aie2p_array_tile_state_t* candidate_state =
          loom_aie2p_array_tile_state(builder, request.coordinate);
      const bool candidate_has_worker =
          iree_any_bit_set(candidate_state->resources.flags,
                           LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_WORKER);
      if (candidate_has_worker != select_worker_tiles ||
          candidate_state->resources.facts->kind !=
              LOOM_XDNA_TILE_KIND_COMPUTE) {
        continue;
      }
      if (loom_aie2p_array_try_compute_endpoint(builder, source_endpoint,
                                                &request, out_proposal)) {
        return true;
      }
    }
  }
  return false;
}

static uint32_t loom_aie2p_array_append_lock_pair(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_endpoint_direction_t ring_endpoint_direction,
    const loom_aie2p_array_ring_resource_proposal_t* proposal) {
  const uint32_t credit_lock_index = (uint32_t)builder->lock_cursor;
  builder->locks[builder->lock_cursor++] = (loom_aie2p_array_lock_plan_t){
      .channel_index = channel_index,
      .coordinate = proposal->coordinate,
      .lock_id = proposal->credit_lock,
      .initial_value = (int8_t)proposal->record_count,
      .ring_endpoint_direction = ring_endpoint_direction,
      .consumer_ready = 0,
  };
  builder->locks[builder->lock_cursor++] = (loom_aie2p_array_lock_plan_t){
      .channel_index = channel_index,
      .coordinate = proposal->coordinate,
      .lock_id = proposal->ready_lock,
      .initial_value = 0,
      .ring_endpoint_direction = ring_endpoint_direction,
      .consumer_ready = 1,
  };
  return credit_lock_index;
}

static loom_aie2p_array_committed_endpoint_t
loom_aie2p_array_commit_compute_endpoint(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_endpoint_direction_t ring_endpoint_direction,
    const loom_aie2p_array_compute_endpoint_proposal_t* proposal) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, proposal->ring.coordinate);
  loom_aie2p_array_channel_resources_commit_compute(proposal,
                                                    &state->resources);
  const uint32_t credit_lock_index = loom_aie2p_array_append_lock_pair(
      builder, channel_index, ring_endpoint_direction, &proposal->ring);
  const uint32_t dma_index = (uint32_t)builder->dma_channel_cursor;
  loom_aie2p_array_dma_flags_t flags = 0;
  if (iree_any_bit_set(
          proposal->flags,
          LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_PROPOSAL_FLAG_STARTS_DMA_SERVICE)) {
    flags |= LOOM_AIE2P_ARRAY_DMA_FLAG_SERVICE_TILE_LIFECYCLE;
  }
  builder->dma_channels[builder->dma_channel_cursor++] =
      (loom_aie2p_array_dma_plan_t){
          .channel_index = channel_index,
          .coordinate = proposal->ring.coordinate,
          .direction = proposal->direction,
          .dma_channel = proposal->dma_channel,
          .flags = flags,
          .buffer_descriptor_start = proposal->buffer_descriptor_start,
          .buffer_descriptor_count = proposal->buffer_descriptor_count,
          .credit_lock_index = credit_lock_index,
      };
  return (loom_aie2p_array_committed_endpoint_t){
      .dma_index = dma_index,
      .credit_lock_index = credit_lock_index,
  };
}

static uint32_t loom_aie2p_array_commit_neighbor_resources(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_ring_resource_proposal_t* proposal) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, proposal->coordinate);
  loom_aie2p_array_channel_resources_commit_ring(proposal, &state->resources);
  return loom_aie2p_array_append_lock_pair(
      builder, channel_index, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
      proposal);
}

static uint32_t loom_aie2p_array_commit_shim_endpoint(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_shim_endpoint_proposal_t* proposal) {
  loom_aie2p_array_tile_resources_t* resources =
      &loom_aie2p_array_tile_state(builder, proposal->coordinate)->resources;
  loom_aie2p_array_channel_resources_commit_shim(proposal, resources);
  const uint32_t dma_index = (uint32_t)builder->dma_channel_cursor;
  builder->dma_channels[builder->dma_channel_cursor++] =
      (loom_aie2p_array_dma_plan_t){
          .channel_index = channel_index,
          .coordinate = proposal->coordinate,
          .direction = proposal->direction,
          .dma_channel = proposal->dma_channel,
          .flags = LOOM_AIE2P_ARRAY_DMA_FLAG_SHIM,
          .buffer_descriptor_start = proposal->buffer_descriptor_start,
          .buffer_descriptor_count = proposal->buffer_descriptor_count,
          .credit_lock_index = UINT32_MAX,
      };
  return dma_index;
}

static bool loom_aie2p_array_select_shim_endpoint(
    loom_aie2p_array_plan_builder_t* builder, uint16_t preferred_column,
    loom_aie2p_array_dma_direction_t direction, uint16_t descriptor_count,
    loom_aie2p_array_shim_endpoint_proposal_t* out_proposal) {
  for (uint16_t distance = 0; distance < builder->family->column_count;
       ++distance) {
    const int candidates[2] = {
        (int)preferred_column - (int)distance,
        (int)preferred_column + (int)distance,
    };
    const int candidate_count = distance == 0 ? 1 : 2;
    for (int i = 0; i < candidate_count; ++i) {
      if (candidates[i] < 0 || candidates[i] >= builder->family->column_count) {
        continue;
      }
      const loom_xdna_tile_coordinate_t coordinate = {
          .column = (uint16_t)candidates[i],
          .row = 0,
      };
      loom_aie2p_array_tile_state_t* state =
          loom_aie2p_array_tile_state(builder, coordinate);
      if (loom_aie2p_array_channel_resources_propose_shim(
              &state->resources, coordinate, direction, descriptor_count,
              out_proposal) == LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t loom_aie2p_array_reject_channel_resources(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate, iree_string_view_t reason) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(channel_index),
      loom_param_u32(worker_coordinate.column),
      loom_param_u32(worker_coordinate.row),
      loom_param_u32(channel->capacity),
      loom_param_u32(channel->record_byte_length),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_value_def_op(loom_value_table_const_value(
          &builder->module->values, channel->value_id)),
      .error = LOOM_ERR_XDNA_006,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  builder->valid = false;
  return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
}

static loom_xdna_tile_coordinate_t loom_aie2p_array_channel_worker_coordinate(
    const loom_aie2p_array_plan_builder_t* builder,
    const loom_aie2p_array_channel_t* channel,
    loom_aie2p_array_endpoint_direction_t direction) {
  const uint32_t endpoint_index =
      direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND
          ? channel->sender_endpoint_index
          : channel->receiver_endpoint_index;
  const loom_aie2p_array_endpoint_t* endpoint =
      &builder->endpoints[endpoint_index];
  return builder->workers[endpoint->owner_index].coordinate;
}

static iree_status_t loom_aie2p_array_admit_compute_endpoint_cardinality(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_dma_direction_t direction,
    loom_xdna_tile_coordinate_t worker_coordinate,
    loom_aie2p_array_physical_demands_t* demands,
    loom_aie2p_array_physical_cardinalities_t* cardinalities) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const loom_xdna_tile_resource_totals_t* totals =
      &loom_xdna_array_tile_kind_facts(builder->family,
                                       LOOM_XDNA_TILE_KIND_COMPUTE)
           ->array_resources;
  uint32_t* directional_channel_count =
      direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
          ? &demands->compute_memory_to_stream_channels
          : &demands->compute_stream_to_memory_channels;
  if (*directional_channel_count == totals->dma_channel_count_per_direction) {
    const iree_string_view_t reason =
        direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
            ? IREE_SV(
                  "the array has no remaining compute memory-to-stream DMA "
                  "channel")
            : IREE_SV(
                  "the array has no remaining compute stream-to-memory DMA "
                  "channel");
    return loom_aie2p_array_reject_channel_resources(builder, channel_index,
                                                     worker_coordinate, reason);
  }
  if (totals->dma_buffer_descriptor_count -
          demands->compute_buffer_descriptors <
      channel->capacity) {
    return loom_aie2p_array_reject_channel_resources(
        builder, channel_index, worker_coordinate,
        IREE_SV("the array has too few remaining compute DMA buffer "
                "descriptors for the channel ring"));
  }
  if (totals->lock_count - demands->compute_locks < 2u) {
    return loom_aie2p_array_reject_channel_resources(
        builder, channel_index, worker_coordinate,
        IREE_SV("the array has fewer than two remaining compute "
                "synchronization locks"));
  }
  ++*directional_channel_count;
  demands->compute_buffer_descriptors += channel->capacity;
  demands->compute_locks += 2u;
  ++cardinalities->dma_channel_count;
  cardinalities->lock_count += 2u;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_admit_shim_endpoint_cardinality(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_dma_direction_t direction,
    loom_xdna_tile_coordinate_t worker_coordinate,
    loom_aie2p_array_physical_demands_t* demands,
    loom_aie2p_array_physical_cardinalities_t* cardinalities) {
  const loom_xdna_tile_resource_totals_t* totals =
      &loom_xdna_array_tile_kind_facts(builder->family,
                                       LOOM_XDNA_TILE_KIND_SHIM_NOC)
           ->array_resources;
  uint32_t* directional_channel_count =
      direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
          ? &demands->shim_memory_to_stream_channels
          : &demands->shim_stream_to_memory_channels;
  if (*directional_channel_count == totals->dma_channel_count_per_direction) {
    const iree_string_view_t reason =
        direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
            ? IREE_SV(
                  "the array has no remaining shim memory-to-stream DMA "
                  "channel")
            : IREE_SV(
                  "the array has no remaining shim stream-to-memory DMA "
                  "channel");
    return loom_aie2p_array_reject_channel_resources(builder, channel_index,
                                                     worker_coordinate, reason);
  }
  if (demands->shim_buffer_descriptors == totals->dma_buffer_descriptor_count) {
    return loom_aie2p_array_reject_channel_resources(
        builder, channel_index, worker_coordinate,
        IREE_SV("the array has no remaining shim DMA buffer descriptor"));
  }
  ++*directional_channel_count;
  ++demands->shim_buffer_descriptors;
  ++cardinalities->dma_channel_count;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_admit_neighbor_cardinality(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate,
    loom_aie2p_array_physical_demands_t* demands,
    loom_aie2p_array_physical_cardinalities_t* cardinalities) {
  const loom_xdna_tile_resource_totals_t* totals =
      &loom_xdna_array_tile_kind_facts(builder->family,
                                       LOOM_XDNA_TILE_KIND_COMPUTE)
           ->array_resources;
  if (totals->lock_count - demands->compute_locks < 2u) {
    return loom_aie2p_array_reject_channel_resources(
        builder, channel_index, worker_coordinate,
        IREE_SV("the array has fewer than two remaining compute "
                "synchronization locks"));
  }
  demands->compute_locks += 2u;
  cardinalities->lock_count += 2u;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_reject_worker_local_storage(
    loom_aie2p_array_plan_builder_t* builder, uint32_t worker_index,
    iree_string_view_t purpose, uint64_t requested_bytes,
    uint64_t alignment_bytes) {
  const loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
  const loom_aie2p_array_tile_state_t* tile_state =
      loom_aie2p_array_tile_state(builder, worker->coordinate);
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(worker_index),
      loom_param_u32(worker->coordinate.column),
      loom_param_u32(worker->coordinate.row),
      loom_param_string(purpose),
      loom_param_u64(requested_bytes),
      loom_param_u64(alignment_bytes),
      loom_param_u32(tile_state->resources.facts->memory.local_capacity),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_value_def_op(
          loom_module_value(builder->module, worker->value_id)),
      .error = LOOM_ERR_XDNA_033,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  builder->valid = false;
  return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_plan_workers(
    loom_aie2p_array_plan_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &builder->workers[i];
    const loom_aie2p_array_leaf_t* leaf = worker->leaf;
    const loom_low_function_requirements_t* requirements = &leaf->requirements;
    builder->worker_plans[i] = (loom_aie2p_array_worker_plan_t){
        .worker_index = (uint32_t)i,
        .coordinate = worker->coordinate,
        .first_port = (uint32_t)builder->worker_port_cursor,
        .first_storage = (uint32_t)builder->worker_storage_cursor,
    };
    builder->worker_port_cursor += worker->active_endpoint_count;

    loom_aie2p_array_tile_state_t* tile_state =
        loom_aie2p_array_tile_state(builder, worker->coordinate);
    for (uint32_t space_index = 0; space_index < LOOM_STORAGE_SPACE_COUNT_;
         ++space_index) {
      const loom_storage_space_t storage_space =
          (loom_storage_space_t)space_index;
      const loom_low_storage_layout_requirement_t requirement =
          loom_low_storage_layout_requirement(&requirements->storage_layout,
                                              storage_space);
      if (requirement.byte_length == 0) {
        continue;
      }
      loom_aie2p_array_local_memory_proposal_t proposal;
      if (!loom_aie2p_array_local_memory_propose_worker(
              tile_state->resources.facts, tile_state->resources.bank_cursors,
              requirement.byte_length, requirement.minimum_alignment,
              &proposal)) {
        iree_string_view_t storage_space_name = iree_string_view_empty();
        (void)loom_low_storage_space_set_names(
            loom_low_storage_space_set_for(storage_space), 1,
            &storage_space_name);
        return loom_aie2p_array_reject_worker_local_storage(
            builder, (uint32_t)i, storage_space_name, requirement.byte_length,
            requirement.minimum_alignment);
      }
      loom_aie2p_array_local_memory_commit(
          tile_state->resources.facts, &proposal,
          tile_state->resources.bank_cursors, &tile_state->resources.next_bank);
      const uint32_t load_address =
          tile_state->resources.facts->memory.local_load_base +
          proposal.owner_offset;
      builder->worker_storage[builder->worker_storage_cursor++] =
          (loom_aie2p_array_worker_storage_plan_t){
              .worker_index = (uint32_t)i,
              .storage_space = storage_space,
              .owner_offset = proposal.owner_offset,
              .load_address = load_address,
              .byte_length = (uint32_t)requirement.byte_length,
          };
    }
  }

  // Immutable leaf data is persistent worker state. Reserve it after every
  // authored storage domain and before generated fold and channel storage.
  for (iree_host_size_t worker_index = 0;
       worker_index < builder->plan->worker_count; ++worker_index) {
    const loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
    builder->worker_plans[worker_index].first_read_only_data =
        (uint32_t)builder->read_only_data_cursor;
    const loom_low_function_requirements_t* requirements =
        &worker->leaf->requirements;
    loom_aie2p_array_tile_state_t* tile_state =
        loom_aie2p_array_tile_state(builder, worker->coordinate);
    for (iree_host_size_t requirement_ordinal = 0;
         requirement_ordinal < requirements->read_only_data_count;
         ++requirement_ordinal) {
      const loom_low_read_only_data_requirement_t* requirement =
          &requirements->read_only_data[requirement_ordinal];
      uint32_t owner_offset = 0;
      if (requirement->contents.data_length != 0) {
        loom_aie2p_array_local_memory_proposal_t proposal;
        if (!loom_aie2p_array_local_memory_propose_worker(
                tile_state->resources.facts, tile_state->resources.bank_cursors,
                requirement->contents.data_length,
                requirement->minimum_alignment, &proposal)) {
          const loom_symbol_t* symbol =
              &builder->module->symbols.entries[requirement->symbol.symbol_id];
          return loom_aie2p_array_reject_worker_local_storage(
              builder, (uint32_t)worker_index,
              loom_string_table_get(&builder->module->strings, symbol->name_id),
              requirement->contents.data_length,
              requirement->minimum_alignment);
        }
        loom_aie2p_array_local_memory_commit(tile_state->resources.facts,
                                             &proposal,
                                             tile_state->resources.bank_cursors,
                                             &tile_state->resources.next_bank);
        owner_offset = proposal.owner_offset;
      }
      builder->read_only_data[builder->read_only_data_cursor++] =
          (loom_aie2p_array_read_only_data_plan_t){
              .worker_index = (uint32_t)worker_index,
              .requirement_ordinal = (uint32_t)requirement_ordinal,
              .owner_offset = owner_offset,
              .load_address =
                  tile_state->resources.facts->memory.local_load_base +
                  owner_offset,
              .byte_length = (uint32_t)requirement->contents.data_length,
          };
    }
  }
  return iree_ok_status();
}

// Indexed output extents retained while private fold allocations are formed.
typedef struct loom_aie2p_array_fold_geometry_t {
  // Byte lengths in the worker's contiguous folded-output port order.
  uint32_t* output_byte_lengths;
  // At least one output exceeds one register-backed accumulator tile.
  bool needs_private_state;
} loom_aie2p_array_fold_geometry_t;

// Classifies all canonical output channels in one pass, then reserves private
// state before channel rings consume memory. Narrow folds keep their registers.
static iree_status_t loom_aie2p_array_plan_fold_states(
    loom_aie2p_array_plan_builder_t* builder) {
  const uint32_t fragment_byte_length = 64 * sizeof(float);
  iree_host_size_t output_count = 0;
  for (uint32_t i = 0; i < builder->plan->worker_count; ++i) {
    output_count += builder->workers[i].fold_output_count;
  }
  if (output_count == 0) {
    return iree_ok_status();
  }
  loom_aie2p_array_fold_geometry_t* geometries = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->worker_count, sizeof(*geometries),
      (void**)&geometries));
  uint32_t* output_byte_lengths = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, output_count, sizeof(*output_byte_lengths),
      (void**)&output_byte_lengths));
  iree_host_size_t first_output = 0;
  for (uint32_t i = 0; i < builder->plan->worker_count; ++i) {
    geometries[i] = (loom_aie2p_array_fold_geometry_t){
        .output_byte_lengths = output_byte_lengths + first_output,
    };
    first_output += builder->workers[i].fold_output_count;
  }
  for (uint32_t i = 0; i < builder->plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &builder->channels[i];
    const loom_aie2p_array_endpoint_t* sender =
        &builder->endpoints[channel->sender_endpoint_index];
    if (channel->source_channel_index != i ||
        sender->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      continue;
    }
    const loom_aie2p_array_worker_t* worker =
        &builder->workers[sender->owner_index];
    if (worker->fold_record_count == 0) {
      continue;
    }
    loom_aie2p_array_fold_geometry_t* geometry =
        &geometries[sender->owner_index];
    geometry->output_byte_lengths[sender->port - worker->fold_output_port] =
        channel->record_byte_length;
    geometry->needs_private_state |=
        channel->record_byte_length > fragment_byte_length;
  }
  for (uint32_t worker_index = 0; worker_index < builder->plan->worker_count;
       ++worker_index) {
    const loom_aie2p_array_fold_geometry_t* geometry =
        &geometries[worker_index];
    if (!geometry->needs_private_state) {
      continue;
    }
    const loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
    const uint32_t* lengths = geometry->output_byte_lengths;

    uint64_t byte_length = 0;
    uint32_t span_count = 0;
    for (uint32_t i = 0; i < worker->fold_output_count; ++i) {
      byte_length = iree_align_uint64(
          byte_length, LOOM_AIE2P_ARRAY_CHANNEL_STORAGE_ALIGNMENT);
      byte_length += lengths[i];
      span_count += (lengths[i] >= fragment_byte_length) +
                    (lengths[i] % fragment_byte_length != 0);
    }
    loom_aie2p_array_tile_state_t* tile_state =
        loom_aie2p_array_tile_state(builder, worker->coordinate);
    loom_aie2p_array_local_memory_proposal_t proposal;
    if (!loom_aie2p_array_local_memory_propose_worker(
            tile_state->resources.facts, tile_state->resources.bank_cursors,
            byte_length, LOOM_AIE2P_ARRAY_CHANNEL_STORAGE_ALIGNMENT,
            &proposal)) {
      return loom_aie2p_array_reject_worker_local_storage(
          builder, worker_index, IREE_SV("private fold"), byte_length,
          LOOM_AIE2P_ARRAY_CHANNEL_STORAGE_ALIGNMENT);
    }
    loom_aie2p_array_fold_span_t* spans = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, span_count, sizeof(*spans), (void**)&spans));
    loom_aie2p_array_local_memory_commit(tile_state->resources.facts, &proposal,
                                         tile_state->resources.bank_cursors,
                                         &tile_state->resources.next_bank);
    loom_aie2p_array_fold_state_plan_t* state =
        &builder->worker_plans[worker_index].fold_state;
    state->byte_length = (uint32_t)byte_length;
    state->span_count = span_count;
    state->owner_offset = proposal.owner_offset;
    state->load_address = tile_state->resources.facts->memory.local_load_base +
                          proposal.owner_offset;
    state->spans = spans;
    uint32_t state_byte_offset = 0;
    uint32_t span_index = 0;
    for (uint32_t i = 0; i < worker->fold_output_count; ++i) {
      state_byte_offset = (uint32_t)iree_align_uint64(
          state_byte_offset, LOOM_AIE2P_ARRAY_CHANNEL_STORAGE_ALIGNMENT);
      const uint32_t repeat_count = lengths[i] / fragment_byte_length;
      const uint32_t repeated_byte_length = repeat_count * fragment_byte_length;
      if (repeat_count != 0) {
        spans[span_index++] = (loom_aie2p_array_fold_span_t){
            .output_index = i,
            .output_byte_offset = 0,
            .state_byte_offset = state_byte_offset,
            .byte_length = fragment_byte_length,
            .repeat_count = repeat_count,
        };
      }
      if (repeated_byte_length != lengths[i]) {
        spans[span_index++] = (loom_aie2p_array_fold_span_t){
            .output_index = i,
            .output_byte_offset = repeated_byte_length,
            .state_byte_offset = state_byte_offset + repeated_byte_length,
            .byte_length = lengths[i] - repeated_byte_length,
            .repeat_count = 1,
        };
      }
      state_byte_offset += lengths[i];
    }
  }
  return iree_ok_status();
}

static void loom_aie2p_array_bind_worker_port(
    loom_aie2p_array_plan_builder_t* builder,
    const loom_aie2p_array_endpoint_t* endpoint, uint32_t channel_index,
    uint32_t credit_lock_index) {
  if (endpoint->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
    return;
  }
  loom_aie2p_array_worker_plan_t* worker_plan =
      &builder->worker_plans[endpoint->owner_index];
  const uint32_t port_index =
      worker_plan->first_port + worker_plan->port_count++;
  if (endpoint->worker_resource_ordinal != UINT32_MAX) {
    builder->worker_resource_ports[worker_plan->first_port +
                                   endpoint->worker_resource_ordinal] =
        port_index;
  }
  builder->worker_ports[port_index] = (loom_aie2p_array_worker_port_plan_t){
      .worker_index = endpoint->owner_index,
      .port = endpoint->port,
      .direction = endpoint->direction,
      .channel_index = channel_index,
      .credit_lock_index = credit_lock_index,
      .resident_state_ordinal = endpoint->worker_resource_ordinal,
  };
}

// Returns the canonical channel owning a shared sender, or NULL when this
// channel owns its sender-side physical resources.
static const loom_aie2p_array_channel_t* loom_aie2p_array_source_channel(
    const loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  return channel->source_channel_index == channel_index
             ? NULL
             : &builder->channels[channel->source_channel_index];
}

// Returns one offset from the retained same-tile source-major allocation
// sequence. All records have identical size and alignment, so the sequence is
// exactly the historical sender/receiver interleave with only its labels
// permuted. Relabeling here preserves emitted plans without rerunning
// placement.
static uint32_t loom_aie2p_array_composed_ring_offset(
    const loom_aie2p_array_ring_resource_proposal_t* sender,
    const loom_aie2p_array_ring_resource_proposal_t* receiver,
    uint16_t sequence_index) {
  return sequence_index < sender->record_count
             ? sender->owner_offsets[sequence_index]
             : receiver->owner_offsets[sequence_index - sender->record_count];
}

static void loom_aie2p_array_plan_channel_slots(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver,
    const loom_aie2p_array_ring_resource_proposal_t* sender_proposal,
    const loom_aie2p_array_ring_resource_proposal_t* receiver_proposal) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const bool composed_same_tile_rings =
      channel->transport == LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA &&
      sender_proposal != NULL && receiver_proposal != NULL &&
      loom_aie2p_array_coordinates_equal(sender_proposal->coordinate,
                                         receiver_proposal->coordinate);
  for (uint32_t slot = 0; slot < channel->capacity; ++slot) {
    loom_aie2p_array_channel_slot_t* channel_slot =
        &builder->channel_slots[builder->channel_slot_cursor++];
    *channel_slot = (loom_aie2p_array_channel_slot_t){
        .channel_index = channel_index,
        .slot = slot,
        .byte_length = channel->record_byte_length,
        .sender_storage =
            {
                .owner = {UINT16_MAX, UINT16_MAX},
                .owner_offset = UINT32_MAX,
                .load_address = UINT32_MAX,
            },
        .receiver_storage =
            {
                .owner = {UINT16_MAX, UINT16_MAX},
                .owner_offset = UINT32_MAX,
                .load_address = UINT32_MAX,
            },
    };

    if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const loom_aie2p_array_channel_t* source_channel =
          loom_aie2p_array_source_channel(builder, channel_index);
      if (source_channel != NULL) {
        const uint32_t source_slot_index =
            source_channel->first_channel_slot + slot;
        IREE_ASSERT_LT(source_slot_index, builder->channel_slot_cursor);
        channel_slot->sender_storage =
            builder->channel_slots[source_slot_index].sender_storage;
      } else {
        const uint32_t owner_offset =
            composed_same_tile_rings
                ? loom_aie2p_array_composed_ring_offset(
                      sender_proposal, receiver_proposal, (uint16_t)(slot * 2u))
                : sender_proposal->owner_offsets[slot];
        channel_slot->sender_storage.owner = sender_proposal->coordinate;
        channel_slot->sender_storage.owner_offset = owner_offset;
        channel_slot->sender_storage.load_address =
            sender_proposal->load_address_base + owner_offset;
      }
    }
    if (receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const bool aliases_sender =
          channel->transport ==
          LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY;
      const uint32_t owner_offset =
          aliases_sender             ? channel_slot->sender_storage.owner_offset
          : composed_same_tile_rings ? loom_aie2p_array_composed_ring_offset(
                                           sender_proposal, receiver_proposal,
                                           (uint16_t)(slot * 2u + 1u))
                                     : receiver_proposal->owner_offsets[slot];
      channel_slot->receiver_storage.owner =
          aliases_sender ? channel_slot->sender_storage.owner
                         : receiver_proposal->coordinate;
      channel_slot->receiver_storage.owner_offset = owner_offset;
      channel_slot->receiver_storage.load_address =
          (aliases_sender ? channel->neighbor_receiver_load_address_base
                          : receiver_proposal->load_address_base) +
          owner_offset;
    }
  }
}

static uint32_t loom_aie2p_array_plan_completion_route(
    loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, coordinate);
  if (state->completion_route_index != UINT32_MAX) {
    return state->completion_route_index;
  }

  const loom_xdna_stream_port_range_t* source =
      loom_xdna_array_stream_port_range(
          builder->family, LOOM_XDNA_TILE_KIND_SHIM_NOC,
          LOOM_XDNA_STREAM_DIRECTION_SLAVE, LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
  const loom_xdna_stream_port_range_t* destination =
      loom_xdna_array_stream_port_range(
          builder->family, LOOM_XDNA_TILE_KIND_SHIM_NOC,
          LOOM_XDNA_STREAM_DIRECTION_MASTER, LOOM_XDNA_STREAM_PORT_SOUTH);

  // TileControl0 -> South0 is the native completion endpoint. It is the only
  // packet demand on this shim; circuit DMA routes occupy different ports.
  // Assign the first packet identity, rule and arbiter/master-select tuple
  // from its unused packet resources and retain them for every egress DMA.
  const uint32_t index = (uint32_t)builder->plan->completion_route_count++;
  builder->completion_routes[index] = (loom_aie2p_array_completion_route_t){
      .coordinate = coordinate,
      .source_ordinal = source->ordinal,
      .destination_ordinal = destination->ordinal,
      .packet_id = 0,
      .arbiter = 0,
      .master_select = 0,
      .rule_slot = 0,
  };
  state->completion_route_index = index;
  return index;
}

static iree_status_t loom_aie2p_array_diagnose_route_capacity(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index) {
  builder->valid = false;
  const loom_diagnostic_param_t params[] = {loom_param_u32(channel_index)};
  const loom_diagnostic_emission_t emission = {
      .op = loom_value_def_op(loom_value_table_const_value(
          &builder->module->values, builder->channels[channel_index].value_id)),
      .error = LOOM_ERR_XDNA_007,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_admit_dma_record_length(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index) {
  loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const loom_aie2p_array_endpoint_t* sender =
      &builder->endpoints[channel->sender_endpoint_index];
  const loom_aie2p_array_endpoint_t* receiver =
      &builder->endpoints[channel->receiver_endpoint_index];
  const loom_aie2p_array_endpoint_t* worker_endpoint =
      sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER ? sender
                                                                   : receiver;
  const loom_xdna_tile_coordinate_t worker_coordinate =
      builder->workers[worker_endpoint->owner_index].coordinate;
  const loom_xdna_dma_facts_t* dma_facts =
      &loom_xdna_array_tile_facts(builder->family, worker_coordinate)->dma;
  const uint64_t scaled_length =
      channel->record_byte_length / dma_facts->transfer_length_granularity;
  const uint64_t minimum_units =
      iree_max(1u, dma_facts->transfer_length_offset);
  const uint64_t maximum_units =
      (uint64_t)dma_facts->maximum_encoded_transfer_length +
      dma_facts->transfer_length_offset;
  if (channel->record_byte_length % dma_facts->transfer_length_granularity ==
          0 &&
      scaled_length >= minimum_units && scaled_length <= maximum_units) {
    channel->encoded_dma_record_length =
        (uint32_t)(scaled_length - dma_facts->transfer_length_offset);
    return iree_ok_status();
  }

  builder->valid = false;
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(channel_index),
      loom_param_u32(channel->record_byte_length),
      loom_param_u64(minimum_units * dma_facts->transfer_length_granularity),
      loom_param_u64(maximum_units * dma_facts->transfer_length_granularity),
      loom_param_u32(dma_facts->transfer_length_granularity),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_value_def_op(loom_value_table_const_value(
          &builder->module->values, channel->value_id)),
      .error = LOOM_ERR_XDNA_017,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_diagnose_binding_transfer(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* binding_endpoint,
    const loom_aie2p_array_endpoint_t* base_binding_endpoint,
    iree_string_view_t reason) {
  loom_diagnostic_param_t source_type =
      loom_param_type(base_binding_endpoint->message_type);
  loom_diagnostic_param_t byte_offset =
      loom_param_u64(binding_endpoint->binding_byte_offset);
  if (binding_endpoint->binding_view_source_endpoint_index == UINT32_MAX) {
    source_type = loom_param_with_field_ref(
        source_type,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0));
  } else {
    byte_offset = loom_param_with_field_ref(
        byte_offset,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 1));
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(channel_index),
      loom_param_with_field_ref(
          loom_param_type(binding_endpoint->message_type),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      source_type,
      byte_offset,
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .module = builder->module,
      .op = loom_value_def_op(loom_value_table_const_value(
          &builder->module->values, binding_endpoint->value_id)),
      .error = LOOM_ERR_XDNA_032,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  builder->valid = false;
  return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
}

// Admits every host-facing transfer before any physical resource cursor moves.
// Ingress binding multicast retains one canonical patch row, matching its
// shared shim DMA. Egress branches each target a distinct binding and retain a
// distinct patch even when their worker-side compute DMA is shared.
static iree_status_t loom_aie2p_array_admit_binding_transfers(
    loom_aie2p_array_plan_builder_t* builder) {
  const loom_xdna_dma_facts_t* shim_dma_facts =
      &loom_xdna_array_tile_kind_facts(builder->family,
                                       LOOM_XDNA_TILE_KIND_SHIM_NOC)
           ->dma;
  for (iree_host_size_t i = 0; i < builder->plan->channel_count; ++i) {
    const uint32_t channel_index = (uint32_t)i;
    loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
    if (channel->transport != LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA) {
      continue;
    }
    const loom_aie2p_array_endpoint_t* sender =
        &builder->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &builder->endpoints[channel->receiver_endpoint_index];
    const bool ingress = iree_any_bit_set(
        channel->resource_flags,
        LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_COMPUTE_STREAM_TO_MEMORY);
    const loom_aie2p_array_endpoint_t* binding_endpoint =
        ingress ? sender : receiver;
    const loom_aie2p_array_endpoint_t* base_binding_endpoint =
        loom_aie2p_array_topology_base_endpoint(builder->plan,
                                                binding_endpoint);
    const bool owns_shim = iree_any_bit_set(
        channel->resource_flags,
        LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_SHIM_MEMORY_TO_STREAM |
            LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_SHIM_STREAM_TO_MEMORY);
    if (!owns_shim) {
      const loom_aie2p_array_channel_t* source_channel =
          loom_aie2p_array_source_channel(builder, channel_index);
      IREE_ASSERT_NE(source_channel->binding_plan_index, UINT32_MAX);
      channel->binding_plan_index = source_channel->binding_plan_index;
      continue;
    }

    IREE_ASSERT_LT(builder->binding_plan_cursor,
                   builder->plan->binding_plan_count);
    channel->binding_plan_index = (uint32_t)builder->binding_plan_cursor++;
    loom_aie2p_array_binding_plan_t* binding_plan =
        &builder->binding_plans[channel->binding_plan_index];
    *binding_plan = (loom_aie2p_array_binding_plan_t){
        .binding_index = base_binding_endpoint->owner_index,
        .channel_index = channel_index,
        .dma_index = UINT32_MAX,
        .partition_lane = binding_endpoint->partition_lane,
        .partition_lane_count = binding_endpoint->partition_lane_count,
        .completion_route_index = UINT32_MAX,
    };
    iree_string_view_t reason = iree_string_view_empty();
    if (!loom_aie2p_array_resolve_binding_transfer(
            builder->module, &builder->facts, builder->family,
            base_binding_endpoint->message_type, binding_endpoint->message_type,
            binding_endpoint->binding_byte_offset,
            binding_endpoint->binding_view_partitioned,
            binding_endpoint->partition_lane, channel->record_byte_length,
            channel->record_count, shim_dma_facts, binding_plan, &reason)) {
      return loom_aie2p_array_diagnose_binding_transfer(
          builder, channel_index, binding_endpoint, base_binding_endpoint,
          reason);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_external_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver) {
  loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const bool ingress = iree_any_bit_set(
      channel->resource_flags,
      LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_COMPUTE_STREAM_TO_MEMORY);
  const loom_aie2p_array_endpoint_t* worker_endpoint =
      ingress ? receiver : sender;
  const loom_aie2p_array_worker_t* worker =
      &builder->workers[worker_endpoint->owner_index];

  const loom_aie2p_array_dma_direction_t compute_direction =
      ingress ? LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY
              : LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM;
  const loom_aie2p_array_dma_direction_t shim_direction =
      ingress ? LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
              : LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY;
  const loom_aie2p_array_channel_t* source_channel =
      loom_aie2p_array_source_channel(builder, channel_index);

  const bool owns_compute_dma = iree_any_bit_set(
      channel->resource_flags,
      LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_COMPUTE_MEMORY_TO_STREAM |
          LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_COMPUTE_STREAM_TO_MEMORY);
  loom_aie2p_array_compute_endpoint_proposal_t compute_proposal = {0};
  if (owns_compute_dma) {
    if (!loom_aie2p_array_select_compute_endpoint(
            builder, worker->coordinate, compute_direction,
            (uint8_t)channel->capacity, channel->record_byte_length,
            /*source_endpoint=*/NULL, &compute_proposal)) {
      const iree_string_view_t reason =
          ingress ? IREE_SV(
                        "no visible compute tile can host the receiving DMA "
                        "endpoint, ring, and locks")
                  : IREE_SV(
                        "no visible compute tile can host the sending DMA "
                        "endpoint, ring, and locks");
      return loom_aie2p_array_reject_channel_resources(
          builder, channel_index, worker->coordinate, reason);
    }
  }

  const loom_aie2p_array_dma_plan_t* shared_compute_dma =
      owns_compute_dma
          ? NULL
          : &builder->dma_channels[source_channel->sender_dma_index];
  const uint16_t preferred_shim_column =
      owns_compute_dma ? compute_proposal.ring.coordinate.column
                       : shared_compute_dma->coordinate.column;
  const bool owns_shim_dma = iree_any_bit_set(
      channel->resource_flags,
      LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_SHIM_MEMORY_TO_STREAM |
          LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_SHIM_STREAM_TO_MEMORY);
  loom_aie2p_array_shim_endpoint_proposal_t shim_proposal = {0};
  if (owns_shim_dma) {
    if (!loom_aie2p_array_select_shim_endpoint(
            builder, preferred_shim_column, shim_direction,
            /*descriptor_count=*/1, &shim_proposal)) {
      const iree_string_view_t reason =
          ingress ? IREE_SV(
                        "no shim tile has a free memory-to-stream DMA channel "
                        "and buffer descriptor")
                  : IREE_SV(
                        "no shim tile has a free stream-to-memory DMA channel "
                        "and buffer descriptor");
      return loom_aie2p_array_reject_channel_resources(
          builder, channel_index, worker->coordinate, reason);
    }
  }

  loom_aie2p_array_committed_endpoint_t committed_compute = {
      .dma_index =
          source_channel ? source_channel->sender_dma_index : UINT32_MAX,
      .credit_lock_index = UINT32_MAX,
  };
  if (owns_compute_dma) {
    committed_compute = loom_aie2p_array_commit_compute_endpoint(
        builder, channel_index, worker_endpoint->direction, &compute_proposal);
  }
  const uint32_t shim_dma_index =
      owns_shim_dma ? loom_aie2p_array_commit_shim_endpoint(
                          builder, channel_index, &shim_proposal)
                    : source_channel->sender_dma_index;
  loom_aie2p_array_dma_plan_t* compute_dma =
      &builder->dma_channels[committed_compute.dma_index];
  loom_aie2p_array_dma_plan_t* shim_dma =
      &builder->dma_channels[shim_dma_index];
  if (!ingress) {
    channel->sender_dma_index = committed_compute.dma_index;
  }
  if (ingress) {
    channel->sender_dma_index = shim_dma_index;
  }
  loom_aie2p_array_plan_channel_slots(
      builder, channel_index, sender, receiver,
      owns_compute_dma && !ingress ? &compute_proposal.ring : NULL,
      owns_compute_dma && ingress ? &compute_proposal.ring : NULL);

  const bool routed =
      ingress ? loom_aie2p_array_route_ingress(
                    &builder->route_builder, channel_index,
                    channel->source_channel_index, shim_dma->coordinate,
                    shim_dma->dma_channel, compute_dma->coordinate,
                    compute_dma->dma_channel)
              : loom_aie2p_array_route_egress(
                    &builder->route_builder, channel_index,
                    channel->source_channel_index, compute_dma->coordinate,
                    compute_dma->dma_channel, shim_dma->coordinate,
                    shim_dma->dma_channel);
  if (!routed) {
    return loom_aie2p_array_diagnose_route_capacity(builder, channel_index);
  }

  if (owns_shim_dma) {
    loom_aie2p_array_binding_plan_t* binding_plan =
        &builder->binding_plans[channel->binding_plan_index];
    binding_plan->dma_index = shim_dma_index;
    binding_plan->completion_route_index =
        ingress ? UINT32_MAX
                : loom_aie2p_array_plan_completion_route(builder,
                                                         shim_dma->coordinate);
  }
  if (owns_compute_dma) {
    loom_aie2p_array_bind_worker_port(builder, worker_endpoint, channel_index,
                                      committed_compute.credit_lock_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_neighbor_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const loom_xdna_tile_coordinate_t owner =
      builder->workers[sender->owner_index].coordinate;
  loom_aie2p_array_tile_state_t* owner_state =
      loom_aie2p_array_tile_state(builder, owner);
  loom_aie2p_array_ring_resource_proposal_t proposal;
  const loom_aie2p_array_channel_resource_failure_t failure =
      loom_aie2p_array_channel_resources_propose_neighbor(
          &owner_state->resources, owner,
          owner_state->resources.facts->memory.local_load_base,
          (uint8_t)channel->capacity, channel->record_byte_length, &proposal);
  if (failure != LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE) {
    const iree_string_view_t reason =
        failure == LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_RING_STORAGE
            ? IREE_SV(
                  "the selected neighbor-memory owner cannot fit the "
                  "channel ring")
            : IREE_SV(
                  "the selected neighbor-memory owner has fewer than two "
                  "free synchronization locks");
    return loom_aie2p_array_reject_channel_resources(builder, channel_index,
                                                     owner, reason);
  }
  const uint32_t credit_lock_index = loom_aie2p_array_commit_neighbor_resources(
      builder, channel_index, &proposal);
  loom_aie2p_array_plan_channel_slots(builder, channel_index, sender, receiver,
                                      &proposal,
                                      /*receiver_proposal=*/NULL);
  loom_aie2p_array_bind_worker_port(builder, sender, channel_index,
                                    credit_lock_index);
  loom_aie2p_array_bind_worker_port(builder, receiver, channel_index,
                                    credit_lock_index);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_routed_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver) {
  loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const loom_aie2p_array_worker_t* sender_worker =
      &builder->workers[sender->owner_index];
  const loom_aie2p_array_worker_t* receiver_worker =
      &builder->workers[receiver->owner_index];
  const loom_aie2p_array_channel_t* source_channel =
      loom_aie2p_array_source_channel(builder, channel_index);
  const bool owns_sender_dma = iree_any_bit_set(
      channel->resource_flags,
      LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_COMPUTE_MEMORY_TO_STREAM);
  loom_aie2p_array_compute_endpoint_proposal_t sender_proposal;
  loom_aie2p_array_source_endpoint_t source_endpoint;
  if (owns_sender_dma) {
    if (!loom_aie2p_array_select_compute_endpoint(
            builder, sender_worker->coordinate,
            LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM,
            (uint8_t)channel->capacity, channel->record_byte_length,
            /*source_endpoint=*/NULL, &sender_proposal)) {
      return loom_aie2p_array_reject_channel_resources(
          builder, channel_index, sender_worker->coordinate,
          IREE_SV("no visible compute tile can host the sending DMA endpoint, "
                  "ring, and locks"));
    }
    source_endpoint = (loom_aie2p_array_source_endpoint_t){
        .coordinate = sender_proposal.ring.coordinate,
        .dma_channel = sender_proposal.dma_channel,
        .proposal = &sender_proposal,
    };
  } else {
    const loom_aie2p_array_dma_plan_t* source_dma =
        &builder->dma_channels[source_channel->sender_dma_index];
    source_endpoint = (loom_aie2p_array_source_endpoint_t){
        .coordinate = source_dma->coordinate,
        .dma_channel = source_dma->dma_channel,
    };
  }
  loom_aie2p_array_compute_endpoint_proposal_t receiver_proposal;
  if (!loom_aie2p_array_select_compute_endpoint(
          builder, receiver_worker->coordinate,
          LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY,
          (uint8_t)channel->capacity, channel->record_byte_length,
          &source_endpoint, &receiver_proposal)) {
    return loom_aie2p_array_reject_channel_resources(
        builder, channel_index, receiver_worker->coordinate,
        IREE_SV("no visible compute tile can host the receiving DMA endpoint, "
                "ring, locks, and source connection"));
  }

  loom_aie2p_array_committed_endpoint_t committed_sender = {
      .dma_index =
          source_channel ? source_channel->sender_dma_index : UINT32_MAX,
      .credit_lock_index = UINT32_MAX,
  };
  if (owns_sender_dma) {
    committed_sender = loom_aie2p_array_commit_compute_endpoint(
        builder, channel_index, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
        &sender_proposal);
  }
  const loom_aie2p_array_committed_endpoint_t committed_receiver =
      loom_aie2p_array_commit_compute_endpoint(
          builder, channel_index, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
          &receiver_proposal);
  channel->sender_dma_index = committed_sender.dma_index;
  loom_aie2p_array_dma_plan_t* sender_dma =
      &builder->dma_channels[committed_sender.dma_index];
  loom_aie2p_array_dma_plan_t* receiver_dma =
      &builder->dma_channels[committed_receiver.dma_index];
  loom_aie2p_array_plan_channel_slots(
      builder, channel_index, sender, receiver,
      owns_sender_dma ? &sender_proposal.ring : NULL, &receiver_proposal.ring);
  if (!loom_aie2p_array_route_workers(
          &builder->route_builder, channel_index, channel->source_channel_index,
          sender_dma->coordinate, sender_dma->dma_channel,
          receiver_dma->coordinate, receiver_dma->dma_channel)) {
    return loom_aie2p_array_diagnose_route_capacity(builder, channel_index);
  }
  if (owns_sender_dma) {
    loom_aie2p_array_bind_worker_port(builder, sender, channel_index,
                                      committed_sender.credit_lock_index);
  }
  loom_aie2p_array_bind_worker_port(builder, receiver, channel_index,
                                    committed_receiver.credit_lock_index);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_channels(
    loom_aie2p_array_plan_builder_t* builder) {
  for (iree_host_size_t i = 0;
       builder->valid && i < builder->plan->channel_count; ++i) {
    loom_aie2p_array_channel_t* channel = &builder->channels[i];
    const loom_aie2p_array_endpoint_t* sender =
        &builder->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &builder->endpoints[channel->receiver_endpoint_index];
    channel->first_channel_slot = (uint32_t)builder->channel_slot_cursor;
    switch (channel->transport) {
      case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_external_channel(
            builder, (uint32_t)i, sender, receiver));
        break;
      }
      case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_neighbor_channel(
            builder, (uint32_t)i, sender, receiver));
        break;
      }
      case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_routed_channel(
            builder, (uint32_t)i, sender, receiver));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE("validated AIE2P channel transport");
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_initialize_tile_states(
    loom_aie2p_array_plan_builder_t* builder) {
  const iree_host_size_t tile_count =
      (iree_host_size_t)builder->family->column_count *
      builder->family->row_count;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, tile_count, sizeof(*builder->tile_states),
      (void**)&builder->tile_states));
  memset(builder->tile_states, 0, tile_count * sizeof(*builder->tile_states));
  iree_host_size_t bank_cursor_count = 0;
  for (uint16_t row = 0; row < builder->family->row_count; ++row) {
    for (uint16_t column = 0; column < builder->family->column_count;
         ++column) {
      loom_aie2p_array_tile_state_t* state = loom_aie2p_array_tile_state(
          builder, (loom_xdna_tile_coordinate_t){column, row});
      state->resources.facts = loom_xdna_array_tile_facts(
          builder->family, (loom_xdna_tile_coordinate_t){column, row});
      bank_cursor_count += state->resources.facts->memory.bank_count;
      state->completion_route_index = UINT32_MAX;
    }
  }
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    loom_aie2p_array_tile_state(builder, builder->workers[i].coordinate)
        ->resources.flags |= LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_WORKER;
  }
  uint32_t* bank_cursors = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, bank_cursor_count, sizeof(*bank_cursors),
      (void**)&bank_cursors));
  memset(bank_cursors, 0, bank_cursor_count * sizeof(*bank_cursors));
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_array_tile_state_t* state = &builder->tile_states[i];
    if (state->resources.facts->kind != LOOM_XDNA_TILE_KIND_SHIM_NOC) {
      state->resources.bank_cursors = bank_cursors;
      bank_cursors += state->resources.facts->memory.bank_count;
    }
  }

  return iree_ok_status();
}

// Admits aggregate physical demand and retains exact fixed-row counts plus a
// bounded route capacity before any arena allocation or tile cursor mutation.
// Generated family validation proves these u32 accumulators cover every
// physically admissible plan.
static iree_status_t loom_aie2p_array_admit_physical_cardinalities(
    loom_aie2p_array_plan_builder_t* builder,
    loom_aie2p_array_physical_cardinalities_t* out_cardinalities) {
  *out_cardinalities = (loom_aie2p_array_physical_cardinalities_t){0};
  loom_aie2p_array_physical_demands_t demands = {0};
  const uint32_t maximum_route_length =
      (uint32_t)builder->family->column_count + builder->family->row_count;
  for (iree_host_size_t i = 0; i < builder->plan->channel_count; ++i) {
    const uint32_t channel_index = (uint32_t)i;
    const loom_aie2p_array_channel_t* channel =
        &builder->channels[channel_index];
    if (channel->transport !=
        LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_array_admit_dma_record_length(builder, channel_index));
      if (!builder->valid) {
        return iree_ok_status();
      }
    }

    const bool owns_compute_memory_to_stream = iree_any_bit_set(
        channel->resource_flags,
        LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_COMPUTE_MEMORY_TO_STREAM);
    const bool owns_compute_stream_to_memory = iree_any_bit_set(
        channel->resource_flags,
        LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_COMPUTE_STREAM_TO_MEMORY);
    const bool owns_shim_memory_to_stream = iree_any_bit_set(
        channel->resource_flags,
        LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_SHIM_MEMORY_TO_STREAM);
    const bool owns_shim_stream_to_memory = iree_any_bit_set(
        channel->resource_flags,
        LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_SHIM_STREAM_TO_MEMORY);
    const bool owns_neighbor_ring = iree_any_bit_set(
        channel->resource_flags,
        LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FLAG_OWNS_NEIGHBOR_RING);
    const loom_xdna_tile_coordinate_t sender_coordinate =
        owns_compute_memory_to_stream || owns_shim_stream_to_memory ||
                owns_neighbor_ring
            ? loom_aie2p_array_channel_worker_coordinate(
                  builder, channel, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND)
            : (loom_xdna_tile_coordinate_t){0};
    const loom_xdna_tile_coordinate_t receiver_coordinate =
        owns_compute_stream_to_memory || owns_shim_memory_to_stream
            ? loom_aie2p_array_channel_worker_coordinate(
                  builder, channel, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE)
            : (loom_xdna_tile_coordinate_t){0};

    if (owns_compute_memory_to_stream) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_admit_compute_endpoint_cardinality(
          builder, channel_index,
          LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM, sender_coordinate,
          &demands, out_cardinalities));
    }
    if (builder->valid && owns_compute_stream_to_memory) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_admit_compute_endpoint_cardinality(
          builder, channel_index,
          LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY, receiver_coordinate,
          &demands, out_cardinalities));
    }
    if (builder->valid && owns_shim_memory_to_stream) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_admit_shim_endpoint_cardinality(
          builder, channel_index,
          LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM, receiver_coordinate,
          &demands, out_cardinalities));
    }
    if (builder->valid && owns_shim_stream_to_memory) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_admit_shim_endpoint_cardinality(
          builder, channel_index,
          LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY, sender_coordinate,
          &demands, out_cardinalities));
    }
    if (builder->valid && owns_neighbor_ring) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_admit_neighbor_cardinality(
          builder, channel_index, sender_coordinate, &demands,
          out_cardinalities));
    }
    if (!builder->valid) {
      return iree_ok_status();
    }

    out_cardinalities->channel_slot_count += channel->capacity;
    if (channel->transport !=
        LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY) {
      out_cardinalities->route_capacity += maximum_route_length;
    }
    if (owns_shim_memory_to_stream || owns_shim_stream_to_memory) {
      ++out_cardinalities->binding_plan_count;
    }
  }

  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &builder->workers[i];
    out_cardinalities->worker_port_count += worker->active_endpoint_count;
    const loom_low_function_requirements_t* requirements =
        &worker->leaf->requirements;
    for (uint32_t space = 0; space < LOOM_STORAGE_SPACE_COUNT_; ++space) {
      out_cardinalities->worker_storage_count +=
          loom_low_storage_layout_requirement(&requirements->storage_layout,
                                              (loom_storage_space_t)space)
              .byte_length != 0;
    }
    out_cardinalities->read_only_data_count +=
        (uint32_t)requirements->read_only_data_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_allocate_physical_plan(
    loom_aie2p_array_plan_builder_t* builder) {
  loom_aie2p_array_physical_cardinalities_t cardinalities;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_admit_physical_cardinalities(builder, &cardinalities));
  if (!builder->valid) {
    return iree_ok_status();
  }

  builder->plan->worker_plan_count = builder->plan->worker_count;
  builder->plan->worker_storage_count = cardinalities.worker_storage_count;
  builder->plan->read_only_data_count = cardinalities.read_only_data_count;
  builder->plan->worker_port_count = cardinalities.worker_port_count;
  builder->plan->channel_slot_count = cardinalities.channel_slot_count;
  builder->plan->lock_count = cardinalities.lock_count;
  builder->plan->dma_channel_count = cardinalities.dma_channel_count;
  builder->plan->route_count = cardinalities.route_capacity;
  builder->plan->binding_plan_count = cardinalities.binding_plan_count;

  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->worker_plan_count,
      sizeof(*builder->worker_plans), (void**)&builder->worker_plans));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->worker_storage_count,
      sizeof(*builder->worker_storage), (void**)&builder->worker_storage));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->read_only_data_count,
      sizeof(*builder->read_only_data), (void**)&builder->read_only_data));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->worker_port_count,
      sizeof(*builder->worker_ports) + sizeof(*builder->worker_resource_ports) +
          sizeof(*builder->worker_fold_output_states),
      (void**)&builder->worker_ports));
  // The u32-aligned arrays share one allocation. Physical ports retain channel
  // order; the trailing index arrays retain source resource and folded-output
  // order.
  if (builder->plan->worker_port_count != 0) {
    builder->worker_resource_ports =
        (uint32_t*)(builder->worker_ports + builder->plan->worker_port_count);
    builder->worker_fold_output_states =
        builder->worker_resource_ports + builder->plan->worker_port_count;
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->channel_slot_count,
      sizeof(*builder->channel_slots), (void**)&builder->channel_slots));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->lock_count, sizeof(*builder->locks),
      (void**)&builder->locks));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->dma_channel_count,
      sizeof(*builder->dma_channels), (void**)&builder->dma_channels));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_builder_initialize(
      builder->family, builder->plan->channel_count, builder->plan->route_count,
      builder->arena, &builder->route_builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->binding_plan_count,
      sizeof(*builder->binding_plans) + sizeof(*builder->completion_routes),
      (void**)&builder->binding_plans));
  // At most one completion route is needed per retained egress binding plan.
  // The trailing rows share the binding allocation and require no additional
  // alignment.
  if (builder->plan->binding_plan_count != 0) {
    builder->completion_routes =
        (loom_aie2p_array_completion_route_t*)(builder->binding_plans +
                                               builder->plan
                                                   ->binding_plan_count);
  }

  builder->plan->worker_plans = builder->worker_plans;
  builder->plan->worker_storage = builder->worker_storage;
  builder->plan->read_only_data = builder->read_only_data;
  builder->plan->worker_ports = builder->worker_ports;
  builder->plan->worker_resource_ports = builder->worker_resource_ports;
  builder->plan->worker_fold_output_states = builder->worker_fold_output_states;
  builder->plan->channel_slots = builder->channel_slots;
  builder->plan->locks = builder->locks;
  builder->plan->dma_channels = builder->dma_channels;
  builder->plan->routes = builder->route_builder.routes;
  builder->plan->binding_plans = builder->binding_plans;
  builder->plan->completion_routes = builder->completion_routes;
  return loom_aie2p_array_initialize_tile_states(builder);
}

static void loom_aie2p_array_finalize_worker_port_states(
    loom_aie2p_array_plan_builder_t* builder) {
  for (iree_host_size_t worker_index = 0;
       worker_index < builder->plan->worker_count; ++worker_index) {
    const loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
    loom_aie2p_array_worker_plan_t* worker_plan =
        &builder->worker_plans[worker_index];
    uint32_t next_state_ordinal =
        (uint32_t)worker->leaf->requirements.resource_count;
    for (uint32_t port_ordinal = 0; port_ordinal < worker_plan->port_count;
         ++port_ordinal) {
      loom_aie2p_array_worker_port_plan_t* port =
          &builder->worker_ports[worker_plan->first_port + port_ordinal];
      const bool is_fold_output =
          port->direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND &&
          port->port >= worker->fold_output_port &&
          port->port - worker->fold_output_port < worker->fold_output_count;
      if (is_fold_output) {
        if (port->resident_state_ordinal == UINT32_MAX) {
          port->resident_state_ordinal = next_state_ordinal++;
        }
        builder
            ->worker_fold_output_states[worker_plan->first_port + port->port -
                                        worker->fold_output_port] =
            port->resident_state_ordinal;
      }
    }
    worker_plan->resident_state_count = next_state_ordinal;
  }
}

static void loom_aie2p_array_finalize_physical_counts(
    loom_aie2p_array_plan_builder_t* builder) {
  builder->plan->worker_storage_count = builder->worker_storage_cursor;
  builder->plan->read_only_data_count = builder->read_only_data_cursor;
  builder->plan->worker_port_count = builder->worker_port_cursor;
  builder->plan->channel_slot_count = builder->channel_slot_cursor;
  builder->plan->lock_count = builder->lock_cursor;
  builder->plan->dma_channel_count = builder->dma_channel_cursor;
  builder->plan->route_count = builder->route_builder.route_count;
  builder->plan->binding_plan_count = builder->binding_plan_cursor;
}

iree_status_t loom_aie2p_array_plan_build(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_aie2p_array_leaf_t* leaves, iree_host_size_t leaf_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_aie2p_array_plan_t* out_plan, bool* out_valid) {
  *out_plan = (loom_aie2p_array_plan_t){0};
  *out_valid = false;
  const loom_func_like_t function =
      loom_func_like_const_cast(module, function_op);
  loom_region_t* body = loom_func_like_body(function);

  loom_aie2p_array_plan_t plan = {
      .function_op = function_op,
      .family = loom_xdna_npu2_array_family(),
  };
  const loom_aie2p_array_abi_layout_t abi_layout =
      loom_aie2p_array_abi_layout_from_verified(function_op);
  plan.binding_slot_count = abi_layout.binding_slot_count;
  loom_aie2p_array_plan_builder_t builder = {
      .module = module,
      .function_op = function_op,
      .descriptor_set = loom_aie2p_array_descriptor_set(),
      .family = loom_xdna_npu2_array_family(),
      .leaves = leaves,
      .leaf_count = leaf_count,
      .diagnostic_emitter = diagnostic_emitter,
      .valid = true,
      .has_explicit_binding_slot_count = abi_layout.has_binding_slot_count,
      .arena = arena,
      .plan = &plan,
  };
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&builder.facts, arena,
                                                        module->values.count));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(&builder.facts, module, function));
  const loom_block_t* block = loom_region_entry_block(body);
  loom_aie2p_array_count_topology(&builder, block);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_topology(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_topology(&builder, block));
  if (!builder.valid) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_validate(
      module, &builder.facts, diagnostic_emitter, arena, &plan, builder.workers,
      builder.endpoints, builder.channels, &builder.valid));
  if (!builder.valid) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_plan(&builder));
  if (!builder.valid) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_admit_binding_transfers(&builder));
  if (!builder.valid) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_workers(&builder));
  if (!builder.valid) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_fold_states(&builder));
  if (!builder.valid) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channels(&builder));
  if (builder.valid) {
    loom_aie2p_array_finalize_worker_port_states(&builder);
    loom_aie2p_array_finalize_physical_counts(&builder);
    *out_plan = plan;
    *out_valid = true;
  }
  return iree_ok_status();
}
