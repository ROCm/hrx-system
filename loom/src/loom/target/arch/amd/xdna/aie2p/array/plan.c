// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/array/binding.h"
#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/array/topology.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT = 64,
};

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

typedef uint8_t loom_aie2p_array_tile_state_flags_t;
enum loom_aie2p_array_tile_state_flag_bits_e {
  LOOM_AIE2P_ARRAY_TILE_STATE_FLAG_HAS_WORKER = 1u << 0,
  LOOM_AIE2P_ARRAY_TILE_STATE_FLAG_HAS_DMA_SERVICE = 1u << 1,
};

typedef struct loom_aie2p_array_entity_t {
  // Locally defined topology result admitted by array Low verification.
  loom_value_id_t value_id;
  // Logical topology table containing the indexed row.
  loom_aie2p_array_entity_kind_t kind;
  // Row in the table selected by kind.
  uint32_t index;
} loom_aie2p_array_entity_t;

typedef struct loom_aie2p_array_tile_state_t {
  const loom_xdna_tile_facts_t* facts;
  // Memory allocation and shim packet allocation occupy disjoint tile kinds.
  union {
    // Next free byte in each local-memory bank on compute and memory tiles.
    uint32_t* bank_cursors;
    // Retained route on a shim, or UINT32_MAX before its first egress binding.
    uint32_t completion_route_index;
  } allocation;
  uint16_t next_buffer_descriptor;
  uint8_t next_memory_to_stream_channel;
  uint8_t next_stream_to_memory_channel;
  uint8_t next_lock;
  uint8_t next_bank;
  // Worker occupancy and ownership of service-tile lifecycle programming.
  loom_aie2p_array_tile_state_flags_t flags;
} loom_aie2p_array_tile_state_t;

typedef struct loom_aie2p_array_source_endpoint_t {
  // Retained sending DMA whose stream must reach the selected receiver.
  const loom_aie2p_array_dma_plan_t* dma;
  // Storage and locks not yet charged to a newly selected sending endpoint.
  struct {
    // Byte length of each pending ring record.
    uint32_t record_byte_length;
    // Pending record count, or zero when reusing an allocated multicast source.
    uint32_t record_count;
  } pending_ring;
} loom_aie2p_array_source_endpoint_t;

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

static iree_status_t loom_aie2p_array_exact_u32(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id,
    const char* purpose, uint32_t* out_value) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  int64_t value = 0;
  if (!loom_value_facts_query_all_equal_element(
          &builder->facts.context,
          loom_value_fact_table_lookup(&builder->facts, value_id),
          &element_facts) ||
      !loom_value_facts_as_exact_i64(element_facts, &value) || value < 0 ||
      value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array %s must resolve to one exact non-negative u32 fact",
        purpose);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
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

static iree_status_t loom_aie2p_array_dimension_value(
    const loom_aie2p_array_plan_builder_t* builder, loom_type_t type,
    iree_host_size_t dimension, uint32_t* out_value) {
  if (!loom_type_dim_is_dynamic_at(type, dimension)) {
    const int64_t value = loom_type_dim_static_size_at(type, dimension);
    if (value < 0 || value > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P channel tile dimension is out of range");
    }
    *out_value = (uint32_t)value;
    return iree_ok_status();
  }
  return loom_aie2p_array_exact_u32(builder,
                                    loom_type_dim_value_id_at(type, dimension),
                                    "channel tile dimension", out_value);
}

static iree_status_t loom_aie2p_array_element_count(
    const loom_aie2p_array_plan_builder_t* builder, loom_type_t type,
    uint64_t* out_count) {
  uint64_t count = 1;
  for (iree_host_size_t i = 0; i < loom_type_rank(type); ++i) {
    uint32_t dimension = 0;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_array_dimension_value(builder, type, i, &dimension));
    if (dimension == 0 ||
        !iree_checked_mul_u64(count, (uint64_t)dimension, &count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P channel tile must have a non-empty representable shape");
    }
  }
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_record_byte_length(
    const loom_aie2p_array_plan_builder_t* builder, loom_type_t type,
    uint32_t* out_byte_length) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_element_count(builder, type, &element_count));
  const int32_t element_bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  uint64_t bit_length = 0;
  if (!iree_checked_mul_u64(element_count, (uint64_t)element_bit_width,
                            &bit_length) ||
      (bit_length & 7u) != 0 || bit_length / 8u > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P channel tile must have a representable whole-byte footprint");
  }
  *out_byte_length = (uint32_t)(bit_length / 8u);
  return iree_ok_status();
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
        .error = LOOM_ERR_TARGET_125,
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
          .error = LOOM_ERR_TARGET_083,
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
  endpoint->binding_view_source_endpoint_index = UINT32_MAX;
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
  endpoint->binding_byte_offset =
      loom_aie2p_array_constant(builder, loom_op_operands(op)[1]);
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

static iree_status_t loom_aie2p_array_extract_channel(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  const uint32_t channel_index = (uint32_t)builder->channel_cursor;
  loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  channel->value_id = loom_op_results(op)[0];
  channel->source_channel_index = channel_index;
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
  const loom_type_t message_type =
      loom_aie2p_array_result_value_type(builder->module, op);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_record_byte_length(
      builder, message_type, &channel->record_byte_length));
  ++builder->channel_cursor;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_location(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  const uint32_t worker_index =
      loom_aie2p_array_lookup_entity(builder, loom_op_operands(op)[0])->index;
  loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
  if (worker->coordinate.column != UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P worker has multiple locations");
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
        .error = LOOM_ERR_TARGET_120,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
  }
  worker->coordinate = (loom_xdna_tile_coordinate_t){
      .column = (uint16_t)column,
      .row = (uint16_t)row,
  };
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
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_channel(builder, op));
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

static bool loom_aie2p_array_try_allocate_storage_in_bank(
    loom_aie2p_array_tile_state_t* state, uint8_t bank, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_owner_offset) {
  const uint32_t bank_capacity =
      state->facts->memory.local_capacity / state->facts->memory.bank_count;
  const uint32_t bank_begin = bank * bank_capacity;
  const uint32_t bank_end = bank_begin + bank_capacity;
  uint64_t cursor = bank_begin + state->allocation.bank_cursors[bank];
  if (!iree_checked_align_u64(cursor, alignment, &cursor)) {
    return false;
  }
  const uint64_t owner_end = cursor + byte_length;
  if (owner_end > bank_end) {
    // Keep single-bank record placement stable. Larger records consume every
    // intervening bank prefix and cannot cross an already occupied prefix.
    if (byte_length <= bank_capacity || cursor >= bank_end ||
        owner_end > state->facts->memory.local_capacity) {
      return false;
    }
    const uint8_t last_bank = (uint8_t)((owner_end - 1u) / bank_capacity);
    for (uint8_t i = bank + 1u; i <= last_bank; ++i) {
      if (state->allocation.bank_cursors[i] != 0) {
        return false;
      }
    }
    for (uint8_t i = bank; i < last_bank; ++i) {
      state->allocation.bank_cursors[i] = bank_capacity;
    }
    state->allocation.bank_cursors[last_bank] =
        (uint32_t)owner_end - last_bank * bank_capacity;
  } else {
    state->allocation.bank_cursors[bank] = (uint32_t)owner_end - bank_begin;
  }
  *out_owner_offset = (uint32_t)cursor;
  return true;
}

static bool loom_aie2p_array_try_allocate_channel_storage(
    loom_aie2p_array_tile_state_t* state, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_owner_offset) {
  for (uint8_t attempt = 0; attempt < state->facts->memory.bank_count;
       ++attempt) {
    const uint8_t bank = (uint8_t)((state->next_bank + attempt) %
                                   state->facts->memory.bank_count);
    if (!loom_aie2p_array_try_allocate_storage_in_bank(
            state, bank, byte_length, alignment, out_owner_offset)) {
      continue;
    }
    state->next_bank = (uint8_t)((bank + 1u) % state->facts->memory.bank_count);
    return true;
  }
  return false;
}

static iree_status_t loom_aie2p_array_allocate_channel_storage(
    loom_aie2p_array_tile_state_t* state, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_owner_offset) {
  if (loom_aie2p_array_try_allocate_channel_storage(
          state, byte_length, alignment, out_owner_offset)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "AIE2P tile local memory banks are exhausted");
}

// Packs persistent worker storage from high banks without perturbing the
// round-robin cursor used by channel rings. Keeping the two allocation classes
// independent gives equivalent workers stable ring addresses while separating
// long-lived state from the first banks selected for streaming traffic.
static iree_status_t loom_aie2p_array_allocate_worker_storage(
    loom_aie2p_array_tile_state_t* state, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_owner_offset) {
  const uint8_t bank_count = state->facts->memory.bank_count;
  for (uint8_t attempt = 0; attempt < bank_count; ++attempt) {
    const uint8_t bank = (uint8_t)(bank_count - attempt - 1u);
    if (!loom_aie2p_array_try_allocate_storage_in_bank(
            state, bank, byte_length, alignment, out_owner_offset)) {
      continue;
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "AIE2P tile local memory banks are exhausted");
}

static iree_status_t loom_aie2p_array_allocate_lock_pair(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_endpoint_direction_t ring_endpoint_direction,
    int8_t credit_count, uint32_t* out_credit_lock_index) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, coordinate);
  if ((uint32_t)state->next_lock + 2u > state->facts->lock_count ||
      credit_count < state->facts->lock_value_minimum ||
      credit_count > state->facts->lock_value_maximum) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel lock resources are exhausted");
  }
  const uint32_t credit_lock_index = (uint32_t)builder->lock_cursor;
  const uint8_t credit_lock = state->next_lock++;
  const uint8_t ready_lock = state->next_lock++;
  builder->locks[builder->lock_cursor++] = (loom_aie2p_array_lock_plan_t){
      .channel_index = channel_index,
      .coordinate = coordinate,
      .lock_id = credit_lock,
      .initial_value = credit_count,
      .ring_endpoint_direction = ring_endpoint_direction,
      .consumer_ready = 0,
  };
  builder->locks[builder->lock_cursor++] = (loom_aie2p_array_lock_plan_t){
      .channel_index = channel_index,
      .coordinate = coordinate,
      .lock_id = ready_lock,
      .initial_value = 0,
      .ring_endpoint_direction = ring_endpoint_direction,
      .consumer_ready = 1,
  };
  *out_credit_lock_index = credit_lock_index;
  return iree_ok_status();
}

static bool loom_aie2p_array_can_allocate_dma(
    const loom_aie2p_array_tile_state_t* state,
    loom_aie2p_array_dma_direction_t direction, uint32_t descriptor_count) {
  const uint8_t next_channel =
      direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
          ? state->next_memory_to_stream_channel
          : state->next_stream_to_memory_channel;
  return next_channel < state->facts->dma.channel_count_per_direction &&
         descriptor_count <= UINT16_MAX &&
         (uint32_t)state->next_buffer_descriptor + descriptor_count <=
             state->facts->dma.buffer_descriptor_count;
}

static bool loom_aie2p_array_can_allocate_ring_storage(
    const loom_aie2p_array_tile_state_t* state,
    loom_xdna_tile_coordinate_t coordinate, uint32_t record_byte_length,
    uint32_t record_count,
    const loom_aie2p_array_source_endpoint_t* source_endpoint) {
  const uint8_t bank_count = state->facts->memory.bank_count;
  uint32_t bank_cursors[UINT8_MAX + 1u];
  memcpy(bank_cursors, state->allocation.bank_cursors,
         bank_count * sizeof(*bank_cursors));
  loom_aie2p_array_tile_state_t probe = *state;
  probe.allocation.bank_cursors = bank_cursors;
  if (source_endpoint != NULL &&
      source_endpoint->dma->coordinate.column == coordinate.column &&
      source_endpoint->dma->coordinate.row == coordinate.row) {
    for (uint32_t record = 0;
         record < source_endpoint->pending_ring.record_count; ++record) {
      uint32_t owner_offset = 0;
      if (!loom_aie2p_array_try_allocate_channel_storage(
              &probe, source_endpoint->pending_ring.record_byte_length,
              LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT, &owner_offset)) {
        return false;
      }
    }
  }
  for (uint32_t record = 0; record < record_count; ++record) {
    uint32_t owner_offset = 0;
    if (!loom_aie2p_array_try_allocate_channel_storage(
            &probe, record_byte_length, LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT,
            &owner_offset)) {
      return false;
    }
  }
  return true;
}

static bool loom_aie2p_array_can_allocate_channel_endpoint(
    const loom_aie2p_array_tile_state_t* state,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_dma_direction_t direction, uint32_t descriptor_count,
    uint32_t record_byte_length,
    const loom_aie2p_array_source_endpoint_t* source_endpoint) {
  const bool shares_source_tile =
      source_endpoint != NULL &&
      source_endpoint->dma->coordinate.column == coordinate.column &&
      source_endpoint->dma->coordinate.row == coordinate.row;
  if (shares_source_tile && (source_endpoint->dma->dma_channel >=
                                 state->facts->dma.loopback_channel_count ||
                             source_endpoint->dma->dma_channel !=
                                 state->next_stream_to_memory_channel)) {
    return false;
  }
  const uint32_t required_lock_count =
      shares_source_tile && source_endpoint->pending_ring.record_count != 0
          ? 4u
          : 2u;
  return loom_aie2p_array_can_allocate_dma(state, direction,
                                           descriptor_count) &&
         (uint32_t)state->next_lock + required_lock_count <=
             state->facts->lock_count &&
         loom_aie2p_array_can_allocate_ring_storage(
             state, coordinate, record_byte_length, descriptor_count,
             source_endpoint);
}

static iree_status_t loom_aie2p_array_allocate_dma(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_dma_direction_t direction, uint32_t descriptor_count,
    loom_aie2p_array_dma_flags_t flags, uint32_t* out_dma_index) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, coordinate);
  if (!loom_aie2p_array_can_allocate_dma(state, direction, descriptor_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P DMA resources are exhausted");
  }
  uint8_t* next_channel =
      direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
          ? &state->next_memory_to_stream_channel
          : &state->next_stream_to_memory_channel;
  const uint8_t dma_channel = (*next_channel)++;
  const uint16_t buffer_descriptor_start = state->next_buffer_descriptor;
  state->next_buffer_descriptor =
      (uint16_t)(state->next_buffer_descriptor + descriptor_count);
  const uint32_t dma_index = (uint32_t)builder->dma_channel_cursor;
  if (!iree_any_bit_set(flags, LOOM_AIE2P_ARRAY_DMA_FLAG_SHIM) &&
      !iree_any_bit_set(state->flags,
                        LOOM_AIE2P_ARRAY_TILE_STATE_FLAG_HAS_WORKER) &&
      !iree_any_bit_set(state->flags,
                        LOOM_AIE2P_ARRAY_TILE_STATE_FLAG_HAS_DMA_SERVICE)) {
    flags |= LOOM_AIE2P_ARRAY_DMA_FLAG_SERVICE_TILE_LIFECYCLE;
    state->flags |= LOOM_AIE2P_ARRAY_TILE_STATE_FLAG_HAS_DMA_SERVICE;
  }
  builder->dma_channels[builder->dma_channel_cursor++] =
      (loom_aie2p_array_dma_plan_t){
          .channel_index = channel_index,
          .coordinate = coordinate,
          .direction = direction,
          .dma_channel = dma_channel,
          .buffer_descriptor_start = buffer_descriptor_start,
          .buffer_descriptor_count = (uint16_t)descriptor_count,
          .credit_lock_index = UINT32_MAX,
          .flags = flags,
      };
  *out_dma_index = dma_index;
  return iree_ok_status();
}

// Selects a compute DMA endpoint whose memory and locks are directly visible
// to |worker_coordinate|. AIE cores can address the local memory and locks of
// adjacent compute tiles, so a worker with more logical ports than local DMA
// channels can use available engines in that architectural window.
// The worker's own tile remains first to keep compact plans local when it has
// capacity. A routed receiver must also connect to the retained source DMA;
// sharing its tile requires an available same-index loopback pair.
static iree_status_t loom_aie2p_array_select_compute_dma(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate,
    loom_aie2p_array_dma_direction_t direction, uint32_t descriptor_count,
    uint32_t record_byte_length,
    const loom_aie2p_array_source_endpoint_t* source_endpoint,
    uint32_t* out_dma_index) {
  loom_aie2p_array_tile_state_t* worker_state =
      loom_aie2p_array_tile_state(builder, worker_coordinate);
  if (loom_aie2p_array_can_allocate_channel_endpoint(
          worker_state, worker_coordinate, direction, descriptor_count,
          record_byte_length, source_endpoint)) {
    return loom_aie2p_array_allocate_dma(
        builder, channel_index, worker_coordinate, direction, descriptor_count,
        /*flags=*/0, out_dma_index);
  }

  // Preserve the local DMA resources of adjacent workers when a visible
  // unoccupied compute tile can service the endpoint instead.
  for (uint8_t worker_pass = 0; worker_pass < 2; ++worker_pass) {
    const bool select_worker_tiles = worker_pass != 0;
    for (uint8_t i = 0; i < worker_state->facts->memory.window_count; ++i) {
      const loom_xdna_address_window_t* window =
          &builder->family
               ->address_windows[worker_state->facts->memory.window_start + i];
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
      const loom_xdna_tile_coordinate_t candidate = {
          .column = (uint16_t)candidate_column,
          .row = (uint16_t)candidate_row,
      };
      loom_aie2p_array_tile_state_t* candidate_state =
          loom_aie2p_array_tile_state(builder, candidate);
      const bool candidate_has_worker = iree_any_bit_set(
          candidate_state->flags, LOOM_AIE2P_ARRAY_TILE_STATE_FLAG_HAS_WORKER);
      if (candidate_has_worker != select_worker_tiles) {
        continue;
      }
      if (candidate_state->facts->kind != LOOM_XDNA_TILE_KIND_COMPUTE ||
          !loom_aie2p_array_can_allocate_channel_endpoint(
              candidate_state, candidate, direction, descriptor_count,
              record_byte_length, source_endpoint)) {
        continue;
      }
      return loom_aie2p_array_allocate_dma(builder, channel_index, candidate,
                                           direction, descriptor_count,
                                           /*flags=*/0, out_dma_index);
    }
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(channel_index),
      loom_param_u32(worker_coordinate.column),
      loom_param_u32(worker_coordinate.row),
      loom_param_u32(descriptor_count),
      loom_param_u32(record_byte_length),
  };
  const loom_diagnostic_emission_t emission = {
      .op = builder->function_op,
      .error = LOOM_ERR_TARGET_088,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  builder->valid = false;
  return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_select_shim_dma(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    uint16_t preferred_column, loom_aie2p_array_dma_direction_t direction,
    uint32_t descriptor_count, uint32_t* out_dma_index) {
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
      if (!loom_aie2p_array_can_allocate_dma(state, direction,
                                             descriptor_count)) {
        continue;
      }
      return loom_aie2p_array_allocate_dma(
          builder, channel_index, coordinate, direction, descriptor_count,
          LOOM_AIE2P_ARRAY_DMA_FLAG_SHIM, out_dma_index);
    }
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "AIE2P shim DMA resources are exhausted");
}

static iree_status_t loom_aie2p_array_bind_worker_leaf(
    const loom_aie2p_array_plan_builder_t* builder, uint32_t worker_index,
    const loom_low_function_requirements_t* requirements,
    uint32_t* out_port_count) {
  if (requirements->return_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P worker entry must return after one channel firing");
  }
  uint32_t port_count = 0;
  for (iree_host_size_t i = 0; i < builder->plan->endpoint_count; ++i) {
    loom_aie2p_array_endpoint_t* endpoint = &builder->endpoints[i];
    if (endpoint->binding_view_source_endpoint_index == UINT32_MAX &&
        endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
        endpoint->owner_index == worker_index) {
      ++port_count;
      endpoint->worker_resource_ordinal = UINT32_MAX;
      iree_host_size_t import_match_count = 0;
      for (iree_host_size_t j = 0; j < requirements->resource_count; ++j) {
        if ((uint64_t)loom_low_resource_index(requirements->resources[j]) ==
            endpoint->port) {
          ++import_match_count;
          endpoint->worker_resource_ordinal = (uint32_t)j;
        }
      }
      if (import_match_count > 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P worker port must match at most one leaf resource import");
      }
    }
  }
  for (iree_host_size_t i = 0; i < requirements->resource_count; ++i) {
    const uint64_t resource_index =
        (uint64_t)loom_low_resource_index(requirements->resources[i]);
    iree_host_size_t endpoint_match_count = 0;
    for (iree_host_size_t j = 0; j < builder->plan->endpoint_count; ++j) {
      const loom_aie2p_array_endpoint_t* endpoint = &builder->endpoints[j];
      if (endpoint->binding_view_source_endpoint_index == UINT32_MAX &&
          endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
          endpoint->owner_index == worker_index &&
          endpoint->port == resource_index) {
        ++endpoint_match_count;
      }
    }
    if (endpoint_match_count != 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P worker resource must be bound by exactly one topology port");
    }
  }
  *out_port_count = port_count;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_workers(
    loom_aie2p_array_plan_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &builder->workers[i];
    const loom_aie2p_array_leaf_t* leaf = worker->leaf;
    const loom_low_function_requirements_t* requirements = &leaf->requirements;
    uint32_t port_count = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_bind_worker_leaf(
        builder, (uint32_t)i, requirements, &port_count));
    builder->worker_plans[i] = (loom_aie2p_array_worker_plan_t){
        .worker_index = (uint32_t)i,
        .coordinate = worker->coordinate,
        .first_port = (uint32_t)builder->worker_port_cursor,
    };
    builder->worker_port_cursor += port_count;

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
      if (requirement.byte_length > UINT32_MAX ||
          requirement.minimum_alignment > UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "AIE2P worker storage requirement is not representable");
      }
      uint32_t owner_offset = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_worker_storage(
          tile_state, (uint32_t)requirement.byte_length,
          (uint32_t)requirement.minimum_alignment, &owner_offset));
      uint32_t load_address = 0;
      IREE_RETURN_IF_ERROR(loom_xdna_array_form_load_address(
          builder->family, worker->coordinate, LOOM_XDNA_MEMORY_SPACE_DATA,
          worker->coordinate, owner_offset, (uint32_t)requirement.byte_length,
          &load_address));
      builder->worker_storage[builder->worker_storage_cursor++] =
          (loom_aie2p_array_worker_storage_plan_t){
              .worker_index = (uint32_t)i,
              .storage_space = storage_space,
              .owner_offset = owner_offset,
              .load_address = load_address,
              .byte_length = (uint32_t)requirement.byte_length,
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
      byte_length =
          iree_align_uint64(byte_length, LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT);
      byte_length += lengths[i];
      span_count += (lengths[i] >= fragment_byte_length) +
                    (lengths[i] % fragment_byte_length != 0);
    }
    if (byte_length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P worker fold state is not representable");
    }
    loom_aie2p_array_fold_state_plan_t* state =
        &builder->worker_plans[worker_index].fold_state;
    state->byte_length = (uint32_t)byte_length;
    state->span_count = span_count;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_worker_storage(
        loom_aie2p_array_tile_state(builder, worker->coordinate),
        state->byte_length, LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT,
        &state->owner_offset));
    IREE_RETURN_IF_ERROR(loom_xdna_array_form_load_address(
        builder->family, worker->coordinate, LOOM_XDNA_MEMORY_SPACE_DATA,
        worker->coordinate, state->owner_offset, state->byte_length,
        &state->load_address));
    loom_aie2p_array_fold_span_t* spans = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, state->span_count, sizeof(*spans), (void**)&spans));
    state->spans = spans;
    uint32_t state_byte_offset = 0;
    uint32_t span_index = 0;
    for (uint32_t i = 0; i < worker->fold_output_count; ++i) {
      state_byte_offset = (uint32_t)iree_align_uint64(
          state_byte_offset, LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT);
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

static iree_status_t loom_aie2p_array_plan_channel_slots(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver,
    const loom_xdna_tile_coordinate_t* sender_storage_owner,
    const loom_xdna_tile_coordinate_t* receiver_storage_owner) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
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
        const loom_xdna_tile_coordinate_t owner =
            sender_storage_owner
                ? *sender_storage_owner
                : builder->workers[sender->owner_index].coordinate;
        loom_aie2p_array_tile_state_t* owner_state =
            loom_aie2p_array_tile_state(builder, owner);
        uint32_t owner_offset = 0;
        IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_channel_storage(
            owner_state, channel->record_byte_length,
            LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT, &owner_offset));
        channel_slot->sender_storage.owner = owner;
        channel_slot->sender_storage.owner_offset = owner_offset;
        IREE_RETURN_IF_ERROR(loom_xdna_array_form_load_address(
            builder->family, builder->workers[sender->owner_index].coordinate,
            LOOM_XDNA_MEMORY_SPACE_DATA, owner, owner_offset,
            channel->record_byte_length,
            &channel_slot->sender_storage.load_address));
      }
    }
    if (receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const bool aliases_sender =
          channel->transport ==
          LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY;
      const loom_xdna_tile_coordinate_t owner =
          aliases_sender ? channel_slot->sender_storage.owner
          : receiver_storage_owner
              ? *receiver_storage_owner
              : builder->workers[receiver->owner_index].coordinate;
      uint32_t owner_offset = channel_slot->sender_storage.owner_offset;
      if (!aliases_sender) {
        loom_aie2p_array_tile_state_t* owner_state =
            loom_aie2p_array_tile_state(builder, owner);
        IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_channel_storage(
            owner_state, channel->record_byte_length,
            LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT, &owner_offset));
      }
      channel_slot->receiver_storage.owner = owner;
      channel_slot->receiver_storage.owner_offset = owner_offset;
      IREE_RETURN_IF_ERROR(loom_xdna_array_form_load_address(
          builder->family, builder->workers[receiver->owner_index].coordinate,
          LOOM_XDNA_MEMORY_SPACE_DATA, owner, owner_offset,
          channel->record_byte_length,
          &channel_slot->receiver_storage.load_address));
    }
  }
  return iree_ok_status();
}

static uint32_t loom_aie2p_array_plan_completion_route(
    loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, coordinate);
  if (state->allocation.completion_route_index != UINT32_MAX) {
    return state->allocation.completion_route_index;
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
  state->allocation.completion_route_index = index;
  return index;
}

static iree_status_t loom_aie2p_array_diagnose_route_capacity(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index) {
  builder->valid = false;
  const loom_diagnostic_param_t params[] = {loom_param_u32(channel_index)};
  const loom_diagnostic_emission_t emission = {
      .op = loom_value_def_op(loom_value_table_const_value(
          &builder->module->values, builder->channels[channel_index].value_id)),
      .error = LOOM_ERR_TARGET_092,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(builder->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_plan_external_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver) {
  loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const bool ingress =
      loom_aie2p_array_topology_base_endpoint(builder->plan, sender)
          ->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING;
  const loom_aie2p_array_endpoint_t* binding_endpoint =
      ingress ? sender : receiver;
  const loom_aie2p_array_endpoint_t* base_binding_endpoint =
      loom_aie2p_array_topology_base_endpoint(builder->plan, binding_endpoint);
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

  uint32_t compute_dma_index = UINT32_MAX;
  const bool owns_compute_dma = ingress || source_channel == NULL;
  if (owns_compute_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_select_compute_dma(
        builder, channel_index, worker->coordinate, compute_direction,
        channel->capacity, channel->record_byte_length,
        /*source_endpoint=*/NULL, &compute_dma_index));
    if (!builder->valid) {
      return iree_ok_status();
    }
  } else {
    compute_dma_index = source_channel->sender_dma_index;
  }
  loom_aie2p_array_dma_plan_t* compute_dma =
      &builder->dma_channels[compute_dma_index];
  if (!ingress) {
    channel->sender_dma_index = compute_dma_index;
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channel_slots(
      builder, channel_index, sender, receiver,
      ingress ? NULL : &compute_dma->coordinate,
      ingress ? &compute_dma->coordinate : NULL));

  uint32_t shim_dma_index = UINT32_MAX;
  const bool owns_shim_dma = !ingress || source_channel == NULL;
  if (owns_shim_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_select_shim_dma(
        builder, channel_index, compute_dma->coordinate.column, shim_direction,
        /*descriptor_count=*/1, &shim_dma_index));
  } else {
    shim_dma_index = source_channel->sender_dma_index;
  }
  loom_aie2p_array_dma_plan_t* shim_dma =
      &builder->dma_channels[shim_dma_index];
  if (ingress) {
    channel->sender_dma_index = shim_dma_index;
  }

  if (channel->capacity > INT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel capacity exceeds lock range");
  }
  uint32_t credit_lock_index = UINT32_MAX;
  if (owns_compute_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
        builder, channel_index, compute_dma->coordinate,
        worker_endpoint->direction, (int8_t)channel->capacity,
        &credit_lock_index));
    compute_dma->credit_lock_index = credit_lock_index;
  }

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
    loom_aie2p_array_binding_plan_t binding_plan = {
        .binding_index = base_binding_endpoint->owner_index,
        .channel_index = channel_index,
        .dma_index = shim_dma_index,
        .partition_lane = binding_endpoint->partition_lane,
        .partition_lane_count = binding_endpoint->partition_lane_count,
        .completion_route_index = ingress
                                      ? UINT32_MAX
                                      : loom_aie2p_array_plan_completion_route(
                                            builder, shim_dma->coordinate),
    };
    const loom_xdna_tile_facts_t* shim_tile =
        loom_xdna_array_tile_facts(builder->family, shim_dma->coordinate);
    IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_binding_transfer(
        builder->module, &builder->facts, builder->family,
        base_binding_endpoint->message_type, binding_endpoint->message_type,
        binding_endpoint->binding_byte_offset,
        binding_endpoint->binding_view_partitioned,
        binding_endpoint->partition_lane, channel->record_byte_length,
        channel->record_count, &shim_tile->dma, &binding_plan));
    builder->binding_plans[builder->binding_plan_cursor++] = binding_plan;
  }
  if (owns_compute_dma) {
    loom_aie2p_array_bind_worker_port(builder, worker_endpoint, channel_index,
                                      credit_lock_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_neighbor_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channel_slots(
      builder, channel_index, sender, receiver,
      /*sender_storage_owner=*/NULL, /*receiver_storage_owner=*/NULL));
  const loom_xdna_tile_coordinate_t owner =
      builder->workers[sender->owner_index].coordinate;
  if (channel->capacity > INT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel capacity exceeds lock range");
  }
  uint32_t credit_lock_index = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
      builder, channel_index, owner, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
      (int8_t)channel->capacity, &credit_lock_index));
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
  const bool owns_sender_dma = source_channel == NULL;
  uint32_t sender_dma_index = UINT32_MAX;
  if (owns_sender_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_select_compute_dma(
        builder, channel_index, sender_worker->coordinate,
        LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM, channel->capacity,
        channel->record_byte_length, /*source_endpoint=*/NULL,
        &sender_dma_index));
    if (!builder->valid) {
      return iree_ok_status();
    }
  } else {
    sender_dma_index = source_channel->sender_dma_index;
  }
  channel->sender_dma_index = sender_dma_index;
  loom_aie2p_array_dma_plan_t* sender_dma =
      &builder->dma_channels[sender_dma_index];
  const loom_aie2p_array_source_endpoint_t source_endpoint = {
      .dma = sender_dma,
      .pending_ring =
          {
              .record_byte_length = channel->record_byte_length,
              .record_count = owns_sender_dma ? channel->capacity : 0,
          },
  };
  uint32_t receiver_dma_index = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_select_compute_dma(
      builder, channel_index, receiver_worker->coordinate,
      LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY, channel->capacity,
      channel->record_byte_length, &source_endpoint, &receiver_dma_index));
  if (!builder->valid) {
    return iree_ok_status();
  }
  loom_aie2p_array_dma_plan_t* receiver_dma =
      &builder->dma_channels[receiver_dma_index];
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channel_slots(
      builder, channel_index, sender, receiver, &sender_dma->coordinate,
      &receiver_dma->coordinate));

  if (channel->capacity > INT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel capacity exceeds lock range");
  }
  uint32_t sender_credit_lock_index = UINT32_MAX;
  if (owns_sender_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
        builder, channel_index, sender_dma->coordinate,
        LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND, (int8_t)channel->capacity,
        &sender_credit_lock_index));
    sender_dma->credit_lock_index = sender_credit_lock_index;
  }
  uint32_t receiver_credit_lock_index = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
      builder, channel_index, receiver_dma->coordinate,
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE, (int8_t)channel->capacity,
      &receiver_credit_lock_index));
  receiver_dma->credit_lock_index = receiver_credit_lock_index;
  if (!loom_aie2p_array_route_workers(
          &builder->route_builder, channel_index, channel->source_channel_index,
          sender_dma->coordinate, sender_dma->dma_channel,
          receiver_dma->coordinate, receiver_dma->dma_channel)) {
    return loom_aie2p_array_diagnose_route_capacity(builder, channel_index);
  }
  if (owns_sender_dma) {
    loom_aie2p_array_bind_worker_port(builder, sender, channel_index,
                                      sender_credit_lock_index);
  }
  loom_aie2p_array_bind_worker_port(builder, receiver, channel_index,
                                    receiver_credit_lock_index);
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
      state->facts = loom_xdna_array_tile_facts(
          builder->family, (loom_xdna_tile_coordinate_t){column, row});
      bank_cursor_count += state->facts->memory.bank_count;
    }
  }
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    loom_aie2p_array_tile_state(builder, builder->workers[i].coordinate)
        ->flags |= LOOM_AIE2P_ARRAY_TILE_STATE_FLAG_HAS_WORKER;
  }
  uint32_t* bank_cursors = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, bank_cursor_count, sizeof(*bank_cursors),
      (void**)&bank_cursors));
  memset(bank_cursors, 0, bank_cursor_count * sizeof(*bank_cursors));
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_array_tile_state_t* state = &builder->tile_states[i];
    if (state->facts->kind == LOOM_XDNA_TILE_KIND_SHIM_NOC) {
      state->allocation.completion_route_index = UINT32_MAX;
    } else {
      state->allocation.bank_cursors = bank_cursors;
      bank_cursors += state->facts->memory.bank_count;
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_allocate_physical_plan(
    loom_aie2p_array_plan_builder_t* builder) {
  uint64_t channel_slot_count = 0;
  iree_host_size_t external_channel_count = 0;
  iree_host_size_t neighbor_channel_count = 0;
  iree_host_size_t routed_channel_count = 0;
  for (iree_host_size_t i = 0; i < builder->plan->channel_count; ++i) {
    if (!iree_checked_add_u64(channel_slot_count, builder->channels[i].capacity,
                              &channel_slot_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P channel slot count overflowed");
    }
    if (builder->channels[i].transport ==
        LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA) {
      ++external_channel_count;
    } else if (builder->channels[i].transport ==
               LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY) {
      ++neighbor_channel_count;
    } else {
      ++routed_channel_count;
    }
  }
  if (channel_slot_count > UINT32_MAX ||
      builder->plan->endpoint_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P channel slot or endpoint count is too large");
  }

  uint64_t worker_storage_count = 0;
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_aie2p_array_leaf_t* leaf = builder->workers[i].leaf;
    const loom_low_function_requirements_t* requirements = &leaf->requirements;
    for (uint32_t space = 0; space < LOOM_STORAGE_SPACE_COUNT_; ++space) {
      worker_storage_count +=
          loom_low_storage_layout_requirement(&requirements->storage_layout,
                                              (loom_storage_space_t)space)
              .byte_length != 0;
    }
  }
  uint64_t lock_count = 0;
  uint64_t dma_channel_count = 0;
  uint64_t route_count = 0;
  if (!iree_checked_mul_u64(routed_channel_count, 2u, &lock_count) ||
      !iree_checked_add_u64(lock_count, external_channel_count, &lock_count) ||
      !iree_checked_add_u64(lock_count, neighbor_channel_count, &lock_count) ||
      !iree_checked_mul_u64(lock_count, 2u, &lock_count) ||
      !iree_checked_add_u64(external_channel_count, routed_channel_count,
                            &dma_channel_count) ||
      !iree_checked_mul_u64(dma_channel_count, 2u, &dma_channel_count) ||
      !iree_checked_add_u64(external_channel_count, routed_channel_count,
                            &route_count) ||
      !iree_checked_mul_u64(
          route_count,
          (uint64_t)builder->family->column_count + builder->family->row_count,
          &route_count) ||
      worker_storage_count > IREE_HOST_SIZE_MAX || lock_count > UINT32_MAX ||
      dma_channel_count > UINT32_MAX || route_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P physical plan count overflowed");
  }

  builder->plan->worker_plan_count = builder->plan->worker_count;
  builder->plan->worker_storage_count = (iree_host_size_t)worker_storage_count;
  builder->plan->worker_port_count = builder->plan->endpoint_count;
  builder->plan->channel_slot_count = (iree_host_size_t)channel_slot_count;
  builder->plan->lock_count = (iree_host_size_t)lock_count;
  builder->plan->dma_channel_count = (iree_host_size_t)dma_channel_count;
  builder->plan->route_count = (iree_host_size_t)route_count;
  builder->plan->binding_plan_count = external_channel_count;

  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->worker_plan_count,
      sizeof(*builder->worker_plans), (void**)&builder->worker_plans));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->worker_storage_count,
      sizeof(*builder->worker_storage), (void**)&builder->worker_storage));
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
  // At most one completion route is needed per external binding. The trailing
  // rows share the binding allocation and require no additional alignment.
  if (builder->plan->binding_plan_count != 0) {
    builder->completion_routes =
        (loom_aie2p_array_completion_route_t*)(builder->binding_plans +
                                               builder->plan
                                                   ->binding_plan_count);
  }

  builder->plan->worker_plans = builder->worker_plans;
  builder->plan->worker_storage = builder->worker_storage;
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
  loom_aie2p_array_plan_builder_t builder = {
      .module = module,
      .function_op = function_op,
      .descriptor_set = loom_aie2p_array_descriptor_set(),
      .family = loom_xdna_npu2_array_family(),
      .leaves = leaves,
      .leaf_count = leaf_count,
      .diagnostic_emitter = diagnostic_emitter,
      .valid = true,
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
      &builder.facts, diagnostic_emitter, arena, &plan, builder.channels,
      &builder.valid));
  if (!builder.valid) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_plan(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_workers(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_fold_states(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channels(&builder));
  if (builder.valid) {
    loom_aie2p_array_finalize_worker_port_states(&builder);
    loom_aie2p_array_finalize_physical_counts(&builder);
    *out_plan = plan;
    *out_valid = true;
  }
  return iree_ok_status();
}
