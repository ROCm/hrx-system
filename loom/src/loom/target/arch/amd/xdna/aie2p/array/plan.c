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

typedef struct loom_aie2p_array_entity_t {
  loom_value_id_t value_id;
  loom_aie2p_array_entity_kind_t kind;
  uint32_t index;
} loom_aie2p_array_entity_t;

typedef struct loom_aie2p_array_tile_state_t {
  const loom_xdna_tile_facts_t* facts;
  uint32_t* bank_cursors;
  uint16_t next_buffer_descriptor;
  uint8_t next_memory_to_stream_channel;
  uint8_t next_stream_to_memory_channel;
  uint8_t next_lock;
  uint8_t next_bank;
} loom_aie2p_array_tile_state_t;

typedef struct loom_aie2p_array_pending_endpoint_t {
  // Compute tile selected for the endpoint but not yet charged for its ring.
  loom_xdna_tile_coordinate_t coordinate;
  // Byte length of each pending ring record.
  uint32_t record_byte_length;
  // Number of pending ring records.
  uint32_t record_count;
} loom_aie2p_array_pending_endpoint_t;

typedef struct loom_aie2p_array_plan_builder_t {
  const loom_module_t* module;
  const loom_op_t* function_op;
  const loom_low_descriptor_set_t* descriptor_set;
  const loom_xdna_array_family_t* family;
  const loom_aie2p_array_leaf_t* leaves;
  iree_host_size_t leaf_count;
  // Caller-owned sink for structured failures in authored array topology.
  iree_diagnostic_emitter_t diagnostic_emitter;
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
  loom_aie2p_array_channel_slot_t* channel_slots;
  loom_aie2p_array_lock_plan_t* locks;
  loom_aie2p_array_dma_plan_t* dma_channels;
  loom_aie2p_array_binding_plan_t* binding_plans;
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
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static bool loom_aie2p_array_symbol_ref_equal(loom_symbol_ref_t lhs,
                                              loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static const loom_named_attr_t* loom_aie2p_array_find_attr(
    const loom_module_t* module, const loom_op_t* op, iree_string_view_t name) {
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  const bool has_attrs = loom_low_packet_try_op_attrs(op, &attrs, NULL);
  IREE_ASSERT(has_attrs, "verified descriptor packet must carry attributes");
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(module->strings.entries[attr->name_id], name)) {
      return attr;
    }
  }
  IREE_ASSERT_UNREACHABLE("verified descriptor immediate must be present");
  return NULL;
}

static loom_attribute_t loom_aie2p_array_attr(const loom_module_t* module,
                                              const loom_op_t* op,
                                              iree_string_view_t name) {
  return loom_aie2p_array_find_attr(module, op, name)->value;
}

static int64_t loom_aie2p_array_enum_attr_value(
    const loom_aie2p_array_plan_builder_t* builder,
    const loom_low_descriptor_t* descriptor, iree_string_view_t field_name,
    loom_attribute_t attr) {
  if (attr.kind == LOOM_ATTR_I64) return attr.i64;
  IREE_ASSERT_EQ(attr.kind, LOOM_ATTR_STRING);
  const iree_string_view_t token =
      builder->module->strings.entries[attr.string_id];
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &builder->descriptor_set->immediates[descriptor->immediate_start + i];
    if (!iree_string_view_equal(
            loom_low_descriptor_set_string(builder->descriptor_set,
                                           immediate->field_name_string_offset),
            field_name)) {
      continue;
    }
    IREE_ASSERT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_ENUM);
    const loom_low_enum_domain_t* domain =
        &builder->descriptor_set->enum_domains[immediate->enum_domain_id];
    for (uint16_t j = 0; j < domain->value_count; ++j) {
      const loom_low_enum_value_t* value =
          &builder->descriptor_set->enum_values[domain->value_start + j];
      if (iree_string_view_equal(token, loom_low_descriptor_set_string(
                                            builder->descriptor_set,
                                            value->token_string_offset))) {
        return value->value;
      }
    }
    IREE_ASSERT_UNREACHABLE("verified enum token belongs to its domain");
    return 0;
  }
  IREE_ASSERT_UNREACHABLE("verified descriptor immediate must be present");
  return 0;
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

static iree_status_t loom_aie2p_array_exact_u64(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id,
    const char* purpose, uint64_t* out_value) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  int64_t value = 0;
  if (!loom_value_facts_query_all_equal_element(
          &builder->facts.context,
          loom_value_fact_table_lookup(&builder->facts, value_id),
          &element_facts) ||
      !loom_value_facts_as_exact_i64(element_facts, &value) || value < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array %s must resolve to one exact non-negative i64 fact",
        purpose);
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
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
  if (!loom_type_is_tile(type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P channels must transport tile values");
  }
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
  if (element_bit_width <= 0 ||
      !iree_checked_mul_u64(element_count, (uint64_t)element_bit_width,
                            &bit_length) ||
      (bit_length & 7u) != 0 || bit_length / 8u > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P channel tile must have a representable whole-byte footprint");
  }
  *out_byte_length = (uint32_t)(bit_length / 8u);
  return iree_ok_status();
}

static const loom_type_t* loom_aie2p_array_result_value_type(
    const loom_module_t* module, const loom_op_t* op) {
  IREE_ASSERT_EQ(op->result_count, 1u);
  const loom_type_t register_type =
      loom_module_value_type(module, loom_op_results(op)[0]);
  const loom_type_t* value_type = loom_type_register_value_type(register_type);
  IREE_ASSERT(value_type != NULL,
              "verified array descriptors must produce typed registers");
  return value_type;
}

// Fibonacci hashing mixes the otherwise nearly sequential SSA value IDs for
// the power-of-two compact entity table.
static uint32_t loom_aie2p_array_hash_value_id(loom_value_id_t value_id) {
  return (uint32_t)value_id * 2654435769u;
}

static iree_host_size_t loom_aie2p_array_entity_slot(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id) {
  IREE_ASSERT(iree_host_size_is_power_of_two(builder->entity_capacity));
  iree_host_size_t slot =
      loom_aie2p_array_hash_value_id(value_id) & (builder->entity_capacity - 1);
  while (builder->entities[slot].value_id != LOOM_VALUE_ID_INVALID &&
         builder->entities[slot].value_id != value_id) {
    slot = (slot + 1) & (builder->entity_capacity - 1);
  }
  return slot;
}

static const loom_aie2p_array_entity_t* loom_aie2p_array_find_entity(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || builder->entity_capacity == 0) {
    return NULL;
  }
  const loom_aie2p_array_entity_t* entity =
      &builder->entities[loom_aie2p_array_entity_slot(builder, value_id)];
  return entity->value_id == value_id ? entity : NULL;
}

static void loom_aie2p_array_define_entity(
    loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id,
    loom_aie2p_array_entity_kind_t kind, uint32_t index) {
  IREE_ASSERT_NE(value_id, LOOM_VALUE_ID_INVALID);
  loom_aie2p_array_entity_t* entity =
      &builder->entities[loom_aie2p_array_entity_slot(builder, value_id)];
  IREE_ASSERT_EQ(entity->value_id, LOOM_VALUE_ID_INVALID);
  *entity = (loom_aie2p_array_entity_t){
      .value_id = value_id,
      .kind = kind,
      .index = index,
  };
}

static iree_status_t loom_aie2p_array_lookup_entity(
    const loom_aie2p_array_plan_builder_t* builder, loom_value_id_t value_id,
    loom_aie2p_array_entity_kind_t expected_kind, const char* purpose,
    uint32_t* out_index) {
  const loom_aie2p_array_entity_t* entity =
      loom_aie2p_array_find_entity(builder, value_id);
  if (entity == NULL || entity->kind != expected_kind) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P array %s references an invalid entity",
                            purpose);
  }
  *out_index = entity->index;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_count_topology(
    loom_aie2p_array_plan_builder_t* builder, const loom_block_t* block) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(builder->descriptor_set, op, &packet);
    if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
      if (!loom_low_return_isa(op)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P array function contains a non-topology operation");
      }
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
  return iree_ok_status();
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

static iree_status_t loom_aie2p_array_extract_group(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  loom_aie2p_array_group_t* group = &builder->groups[builder->group_cursor];
  group->value_id = loom_op_results(op)[0];
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_exact_u32(builder, loom_op_operands(op)[0],
                                 "group lane count", &group->lane_count));
  loom_aie2p_array_define_entity(builder, group->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_GROUP,
                                 (uint32_t)builder->group_cursor++);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_binding(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  loom_aie2p_array_binding_t* binding =
      &builder->bindings[builder->binding_cursor];
  binding->value_id = loom_op_results(op)[0];
  binding->ordinal = (uint32_t)loom_attr_as_i64(
      loom_aie2p_array_attr(builder->module, op, IREE_SV("ordinal")));
  const loom_low_descriptor_t* descriptor =
      &builder->descriptor_set
           ->descriptors[AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_BINDING];
  binding->access =
      (loom_aie2p_array_binding_access_t)loom_aie2p_array_enum_attr_value(
          builder, descriptor, IREE_SV("access"),
          loom_aie2p_array_attr(builder->module, op, IREE_SV("access")));
  loom_aie2p_array_define_entity(builder, binding->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_BINDING,
                                 (uint32_t)builder->binding_cursor++);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_worker(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op,
    loom_aie2p_array_worker_rate_t rate) {
  loom_aie2p_array_worker_t* worker = &builder->workers[builder->worker_cursor];
  worker->value_id = loom_op_results(op)[0];
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[0], LOOM_AIE2P_ARRAY_ENTITY_GROUP,
      "worker group", &worker->group_index));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
      builder, loom_op_operands(op)[1], "worker lane", &worker->lane));
  worker->entry = loom_attr_as_symbol(
      loom_aie2p_array_attr(builder->module, op, IREE_SV("entry")));
  worker->fold_record_count = 0;
  worker->fold_output_port = 0;
  worker->fold_output_count = 0;
  worker->fold_kind = LOOM_COMBINING_KIND_ADDI;
  worker->fold_fast_math_flags = 0;
  if (rate == LOOM_AIE2P_ARRAY_WORKER_RATE_FOLDED) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
        builder, loom_op_operands(op)[2], "worker fold record count",
        &worker->fold_record_count));
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
      IREE_RETURN_IF_ERROR(
          iree_diagnostic_emit(builder->diagnostic_emitter, &emission));
      return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    }
    worker->fold_output_port = (uint32_t)loom_attr_as_i64(
        loom_aie2p_array_attr(builder->module, op, IREE_SV("output_port")));
    worker->fold_output_count = (uint32_t)loom_attr_as_i64(
        loom_aie2p_array_attr(builder->module, op, IREE_SV("output_count")));
    const loom_low_descriptor_t* descriptor =
        &builder->descriptor_set
             ->descriptors[AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER_FOLD];
    worker->fold_kind = (loom_combining_kind_t)loom_aie2p_array_enum_attr_value(
        builder, descriptor, IREE_SV("kind"),
        loom_aie2p_array_attr(builder->module, op, IREE_SV("kind")));
    worker->fold_fast_math_flags = (uint8_t)loom_attr_as_i64(
        loom_aie2p_array_attr(builder->module, op, IREE_SV("fast_math")));
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

static iree_status_t loom_aie2p_array_extract_endpoint(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op,
    loom_aie2p_array_endpoint_direction_t direction) {
  loom_aie2p_array_endpoint_t* endpoint =
      &builder->endpoints[builder->endpoint_cursor];
  endpoint->value_id = loom_op_results(op)[0];
  endpoint->direction = direction;
  endpoint->port = (uint32_t)loom_attr_as_i64(
      loom_aie2p_array_attr(builder->module, op, IREE_SV("port")));
  endpoint->message_type =
      *loom_aie2p_array_result_value_type(builder->module, op);
  endpoint->binding_view_source_endpoint_index = UINT32_MAX;
  endpoint->binding_byte_offset = 0;
  endpoint->binding_view_partitioned = false;
  endpoint->partition_lane = 0;
  endpoint->partition_lane_count = 1;

  const loom_value_id_t owner_value = loom_op_operands(op)[0];
  const loom_aie2p_array_entity_t* owner =
      loom_aie2p_array_find_entity(builder, owner_value);
  if (owner != NULL && owner->kind == LOOM_AIE2P_ARRAY_ENTITY_BINDING) {
    endpoint->owner_kind = LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING;
  } else if (owner != NULL && owner->kind == LOOM_AIE2P_ARRAY_ENTITY_WORKER) {
    endpoint->owner_kind = LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array endpoint must be owned by a binding or worker");
  }
  endpoint->owner_index = owner->index;
  loom_aie2p_array_define_entity(builder, endpoint->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
                                 (uint32_t)builder->endpoint_cursor++);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_binding_view(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op,
    bool partitioned) {
  loom_aie2p_array_endpoint_t* endpoint =
      &builder->endpoints[builder->endpoint_cursor];
  endpoint->value_id = loom_op_results(op)[0];
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[0], LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
      "binding view source", &endpoint->binding_view_source_endpoint_index));
  const loom_aie2p_array_endpoint_t* source =
      &builder->endpoints[endpoint->binding_view_source_endpoint_index];
  endpoint->direction = source->direction;
  endpoint->owner_kind = source->owner_kind;
  endpoint->owner_index = source->owner_index;
  endpoint->port = source->port;
  endpoint->message_type =
      *loom_aie2p_array_result_value_type(builder->module, op);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u64(
      builder, loom_op_operands(op)[1], "binding view byte offset",
      &endpoint->binding_byte_offset));
  endpoint->binding_view_partitioned = partitioned;
  endpoint->partition_lane = 0;
  endpoint->partition_lane_count = 1;
  if (partitioned) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
        builder, loom_op_operands(op)[2], "partition lane",
        &endpoint->partition_lane));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
        builder, loom_op_operands(op)[3], "partition lane count",
        &endpoint->partition_lane_count));
  }
  loom_aie2p_array_define_entity(builder, endpoint->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
                                 (uint32_t)builder->endpoint_cursor++);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_channel(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  const uint32_t channel_index = (uint32_t)builder->channel_cursor;
  loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  channel->value_id = loom_op_results(op)[0];
  channel->source_channel_index = channel_index;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[0], LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
      "channel sender", &channel->sender_endpoint_index));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[1], LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
      "channel receiver", &channel->receiver_endpoint_index));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_exact_u32(builder, loom_op_operands(op)[2],
                                 "channel capacity", &channel->capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
      builder, loom_op_operands(op)[3], "channel record count",
      &channel->record_count));
  const loom_type_t message_type =
      *loom_aie2p_array_result_value_type(builder->module, op);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_record_byte_length(
      builder, message_type, &channel->record_byte_length));
  ++builder->channel_cursor;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_location(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  uint32_t worker_index = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[0], LOOM_AIE2P_ARRAY_ENTITY_WORKER,
      "location worker", &worker_index));
  loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
  if (worker->coordinate.column != UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P worker has multiple locations");
  }
  uint32_t column = 0;
  uint32_t row = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
      builder, loom_op_operands(op)[1], "worker column", &column));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
      builder, loom_op_operands(op)[2], "worker row", &row));
  if (column > UINT16_MAX || row > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P worker location is out of range");
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
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(builder->descriptor_set, op, &packet);
    if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) continue;
    switch (packet.descriptor_ordinal) {
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTANT_U32:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTANT_U64:
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_GROUP: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_group(builder, op));
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_BINDING: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_binding(builder, op));
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
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_endpoint(
            builder, op, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND));
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_RECEIVER: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_endpoint(
            builder, op, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE));
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_RECEIVER: {
        IREE_RETURN_IF_ERROR(
            loom_aie2p_array_extract_binding_view(builder, op, false));
        break;
      }
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_RECEIVER: {
        IREE_RETURN_IF_ERROR(
            loom_aie2p_array_extract_binding_view(builder, op, true));
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

static const loom_low_function_requirements_t* loom_aie2p_array_find_leaf(
    const loom_aie2p_array_plan_builder_t* builder, loom_symbol_ref_t entry) {
  const loom_low_function_requirements_t* result = NULL;
  for (iree_host_size_t i = 0; i < builder->leaf_count; ++i) {
    if (loom_aie2p_array_symbol_ref_equal(builder->leaves[i].entry, entry)) {
      IREE_ASSERT(result == NULL, "leaf table entries must be unique");
      result = &builder->leaves[i].requirements;
    }
  }
  return result;
}

static loom_aie2p_array_tile_state_t* loom_aie2p_array_tile_state(
    loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate) {
  const iree_host_size_t index =
      (iree_host_size_t)coordinate.row * builder->family->column_count +
      coordinate.column;
  return &builder->tile_states[index];
}

static bool loom_aie2p_array_coordinate_has_worker(
    const loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate) {
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_xdna_tile_coordinate_t worker_coordinate =
        builder->workers[i].coordinate;
    if (worker_coordinate.column == coordinate.column &&
        worker_coordinate.row == coordinate.row) {
      return true;
    }
  }
  return false;
}

static bool loom_aie2p_array_try_allocate_storage_in_bank(
    loom_aie2p_array_tile_state_t* state, uint8_t bank, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_owner_offset) {
  const uint32_t bank_capacity =
      state->facts->memory.local_capacity / state->facts->memory.bank_count;
  uint64_t cursor = state->bank_cursors[bank];
  if (!iree_checked_align_u64(cursor, alignment, &cursor) ||
      cursor + byte_length > bank_capacity) {
    return false;
  }
  state->bank_cursors[bank] = (uint32_t)(cursor + byte_length);
  *out_owner_offset = bank * bank_capacity + (uint32_t)cursor;
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
    int8_t credit_count) {
  loom_aie2p_array_tile_state_t* state =
      loom_aie2p_array_tile_state(builder, coordinate);
  if ((uint32_t)state->next_lock + 2u > state->facts->lock_count ||
      credit_count < state->facts->lock_value_minimum ||
      credit_count > state->facts->lock_value_maximum) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel lock resources are exhausted");
  }
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
    const loom_aie2p_array_pending_endpoint_t* pending_endpoint) {
  const uint8_t bank_count = state->facts->memory.bank_count;
  uint32_t bank_cursors[UINT8_MAX + 1u];
  memcpy(bank_cursors, state->bank_cursors, bank_count * sizeof(*bank_cursors));
  loom_aie2p_array_tile_state_t probe = *state;
  probe.bank_cursors = bank_cursors;
  if (pending_endpoint != NULL &&
      pending_endpoint->coordinate.column == coordinate.column &&
      pending_endpoint->coordinate.row == coordinate.row) {
    for (uint32_t record = 0; record < pending_endpoint->record_count;
         ++record) {
      uint32_t owner_offset = 0;
      if (!loom_aie2p_array_try_allocate_channel_storage(
              &probe, pending_endpoint->record_byte_length,
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
    const loom_aie2p_array_pending_endpoint_t* pending_endpoint) {
  const bool shares_pending_tile =
      pending_endpoint != NULL &&
      pending_endpoint->coordinate.column == coordinate.column &&
      pending_endpoint->coordinate.row == coordinate.row;
  const uint32_t required_lock_count = shares_pending_tile ? 4u : 2u;
  return loom_aie2p_array_can_allocate_dma(state, direction,
                                           descriptor_count) &&
         (uint32_t)state->next_lock + required_lock_count <=
             state->facts->lock_count &&
         loom_aie2p_array_can_allocate_ring_storage(
             state, coordinate, record_byte_length, descriptor_count,
             pending_endpoint);
}

static iree_status_t loom_aie2p_array_allocate_dma(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_dma_direction_t direction, uint32_t descriptor_count,
    bool shim_side, uint8_t* out_dma_channel) {
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
  builder->dma_channels[builder->dma_channel_cursor++] =
      (loom_aie2p_array_dma_plan_t){
          .channel_index = channel_index,
          .coordinate = coordinate,
          .direction = direction,
          .dma_channel = dma_channel,
          .buffer_descriptor_start = buffer_descriptor_start,
          .buffer_descriptor_count = (uint16_t)descriptor_count,
          .shim_side = shim_side ? 1 : 0,
      };
  *out_dma_channel = dma_channel;
  return iree_ok_status();
}

// Selects a compute DMA endpoint whose memory and locks are directly visible
// to |worker_coordinate|. AIE cores can address the local memory and locks of
// adjacent compute tiles, so a worker with more logical ports than local DMA
// channels can use available engines in that architectural window.
// The worker's own tile remains first to keep compact plans local when it has
// capacity.
static iree_status_t loom_aie2p_array_select_compute_dma(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate,
    loom_aie2p_array_dma_direction_t direction, uint32_t descriptor_count,
    uint32_t record_byte_length,
    const loom_aie2p_array_pending_endpoint_t* pending_endpoint,
    loom_xdna_tile_coordinate_t* out_coordinate, uint8_t* out_dma_channel) {
  loom_aie2p_array_tile_state_t* worker_state =
      loom_aie2p_array_tile_state(builder, worker_coordinate);
  if (loom_aie2p_array_can_allocate_channel_endpoint(
          worker_state, worker_coordinate, direction, descriptor_count,
          record_byte_length, pending_endpoint)) {
    *out_coordinate = worker_coordinate;
    return loom_aie2p_array_allocate_dma(
        builder, channel_index, worker_coordinate, direction, descriptor_count,
        /*shim_side=*/false, out_dma_channel);
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
      if (loom_aie2p_array_coordinate_has_worker(builder, candidate) !=
          select_worker_tiles) {
        continue;
      }
      loom_aie2p_array_tile_state_t* candidate_state =
          loom_aie2p_array_tile_state(builder, candidate);
      if (candidate_state->facts->kind != LOOM_XDNA_TILE_KIND_COMPUTE ||
          !loom_aie2p_array_can_allocate_channel_endpoint(
              candidate_state, candidate, direction, descriptor_count,
              record_byte_length, pending_endpoint)) {
        continue;
      }
      *out_coordinate = candidate;
      return loom_aie2p_array_allocate_dma(
          builder, channel_index, candidate, direction, descriptor_count,
          /*shim_side=*/false, out_dma_channel);
    }
  }
  return iree_make_status(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      "AIE2P worker-visible compute DMA resources are exhausted");
}

static iree_status_t loom_aie2p_array_select_shim_dma(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    uint16_t preferred_column, loom_aie2p_array_dma_direction_t direction,
    uint32_t descriptor_count, loom_xdna_tile_coordinate_t* out_coordinate,
    uint8_t* out_dma_channel) {
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
      IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_dma(
          builder, channel_index, coordinate, direction, descriptor_count,
          /*shim_side=*/true, out_dma_channel));
      *out_coordinate = coordinate;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "AIE2P shim DMA resources are exhausted");
}

static iree_status_t loom_aie2p_array_validate_worker_leaf(
    const loom_aie2p_array_plan_builder_t* builder, uint32_t worker_index,
    const loom_low_function_requirements_t* requirements) {
  if (requirements == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AIE2P worker has no core function requirements");
  }
  const loom_aie2p_array_worker_t* worker = &builder->workers[worker_index];
  IREE_ASSERT_EQ(worker->entry.module_id, 0u);
  IREE_ASSERT_LT(worker->entry.symbol_id, builder->module->symbols.count);
  const loom_symbol_t* worker_symbol =
      &builder->module->symbols.entries[worker->entry.symbol_id];
  if (worker_symbol->defining_op == NULL ||
      !loom_low_func_def_isa(worker_symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P worker entry must name one core Low function definition");
  }
  const loom_func_like_t worker_function =
      loom_func_like_const_cast(builder->module, worker_symbol->defining_op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(worker_function, &argument_count);
  if (argument_count != 0 ||
      loom_low_func_def_results(worker_symbol->defining_op).count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P worker entry must have no register arguments or results");
  }
  if (requirements->return_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P worker entry must return after one channel firing");
  }
  for (iree_host_size_t i = 0; i < builder->plan->endpoint_count; ++i) {
    const loom_aie2p_array_endpoint_t* endpoint = &builder->endpoints[i];
    if (endpoint->binding_view_source_endpoint_index == UINT32_MAX &&
        endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
        endpoint->owner_index == worker_index) {
      iree_host_size_t import_match_count = 0;
      for (iree_host_size_t j = 0; j < requirements->resource_count; ++j) {
        if ((uint64_t)loom_low_resource_index(requirements->resources[j]) ==
            endpoint->port) {
          ++import_match_count;
        }
      }
      if (import_match_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P worker port must match one leaf resource import");
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
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_workers(
    loom_aie2p_array_plan_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &builder->workers[i];
    const loom_low_function_requirements_t* requirements =
        loom_aie2p_array_find_leaf(builder, worker->entry);
    IREE_RETURN_IF_ERROR(loom_aie2p_array_validate_worker_leaf(
        builder, (uint32_t)i, requirements));
    builder->worker_plans[i] = (loom_aie2p_array_worker_plan_t){
        .worker_index = (uint32_t)i,
        .coordinate = worker->coordinate,
        .requirements = requirements,
    };

    loom_aie2p_array_tile_state_t* tile_state =
        loom_aie2p_array_tile_state(builder, worker->coordinate);
    for (uint32_t space_index = 0; space_index < LOOM_STORAGE_SPACE_COUNT_;
         ++space_index) {
      const loom_storage_space_t storage_space =
          (loom_storage_space_t)space_index;
      const loom_low_storage_layout_requirement_t requirement =
          loom_low_storage_layout_requirement(&requirements->storage_layout,
                                              storage_space);
      if (requirement.byte_length == 0) continue;
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

static iree_status_t loom_aie2p_array_append_worker_port(
    loom_aie2p_array_plan_builder_t* builder,
    const loom_aie2p_array_endpoint_t* endpoint, uint32_t channel_index,
    uint32_t first_channel_slot) {
  if (endpoint->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
    return iree_ok_status();
  }
  builder->worker_ports[builder->worker_port_cursor++] =
      (loom_aie2p_array_worker_port_plan_t){
          .worker_index = endpoint->owner_index,
          .port = endpoint->port,
          .direction = endpoint->direction,
          .channel_index = channel_index,
          .first_channel_slot = first_channel_slot,
      };
  return iree_ok_status();
}

// Returns the canonical channel owning a multicast source, or UINT32_MAX when
// the channel itself owns its source storage, DMA, locks, and worker ABI port.
static uint32_t loom_aie2p_array_multicast_source_channel(
    const loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  return channel->source_channel_index == channel_index
             ? UINT32_MAX
             : channel->source_channel_index;
}

static const loom_aie2p_array_dma_plan_t* loom_aie2p_array_multicast_source_dma(
    const loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index) {
  const uint32_t source_channel_index =
      loom_aie2p_array_multicast_source_channel(builder, channel_index);
  if (source_channel_index == UINT32_MAX) return NULL;
  for (iree_host_size_t i = 0; i < builder->dma_channel_cursor; ++i) {
    const loom_aie2p_array_dma_plan_t* dma = &builder->dma_channels[i];
    if (dma->channel_index == source_channel_index && !dma->shim_side &&
        dma->direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM) {
      return dma;
    }
  }
  IREE_ASSERT_UNREACHABLE("planned worker multicast must have one source DMA");
  return NULL;
}

static const loom_aie2p_array_channel_slot_t*
loom_aie2p_array_find_planned_channel_slot(
    const loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    uint32_t slot) {
  for (iree_host_size_t i = 0; i < builder->channel_slot_cursor; ++i) {
    const loom_aie2p_array_channel_slot_t* candidate =
        &builder->channel_slots[i];
    if (candidate->channel_index == channel_index && candidate->slot == slot) {
      return candidate;
    }
  }
  IREE_ASSERT_UNREACHABLE("planned worker multicast source slot must exist");
  return NULL;
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
      const uint32_t source_channel_index =
          loom_aie2p_array_multicast_source_channel(builder, channel_index);
      if (source_channel_index != UINT32_MAX) {
        channel_slot->sender_storage =
            loom_aie2p_array_find_planned_channel_slot(
                builder, source_channel_index, slot)
                ->sender_storage;
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

static iree_status_t loom_aie2p_array_plan_external_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver, uint32_t first_slot) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
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
  loom_xdna_tile_coordinate_t compute_coordinate = {0};
  uint8_t compute_dma_channel = 0;
  const loom_aie2p_array_dma_plan_t* multicast_source_dma =
      ingress ? NULL
              : loom_aie2p_array_multicast_source_dma(builder, channel_index);
  const bool owns_compute_dma = multicast_source_dma == NULL;
  if (multicast_source_dma != NULL) {
    compute_coordinate = multicast_source_dma->coordinate;
    compute_dma_channel = multicast_source_dma->dma_channel;
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_select_compute_dma(
        builder, channel_index, worker->coordinate, compute_direction,
        channel->capacity, channel->record_byte_length,
        /*pending_endpoint=*/NULL, &compute_coordinate, &compute_dma_channel));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channel_slots(
      builder, channel_index, sender, receiver,
      ingress ? NULL : &compute_coordinate,
      ingress ? &compute_coordinate : NULL));

  uint32_t binding_source_channel_index = channel_index;
  if (ingress) {
    for (uint32_t i = 0; i < channel_index; ++i) {
      if (builder->channels[i].transport ==
              LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA &&
          builder->channels[i].sender_endpoint_index ==
              channel->sender_endpoint_index) {
        binding_source_channel_index = i;
        break;
      }
    }
  }
  loom_xdna_tile_coordinate_t shim_coordinate = {0};
  uint8_t shim_dma_channel = 0;
  if (binding_source_channel_index == channel_index) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_select_shim_dma(
        builder, channel_index, compute_coordinate.column, shim_direction,
        /*descriptor_count=*/1, &shim_coordinate, &shim_dma_channel));
  } else {
    const loom_aie2p_array_dma_plan_t* source_dma = NULL;
    for (iree_host_size_t i = 0; i < builder->dma_channel_cursor; ++i) {
      const loom_aie2p_array_dma_plan_t* dma = &builder->dma_channels[i];
      if (dma->channel_index == binding_source_channel_index &&
          dma->shim_side) {
        source_dma = dma;
        break;
      }
    }
    IREE_ASSERT(source_dma != NULL,
                "planned multicast source must own one shim DMA");
    shim_coordinate = source_dma->coordinate;
    shim_dma_channel = source_dma->dma_channel;
  }

  if (channel->capacity > INT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel capacity exceeds lock range");
  }
  if (owns_compute_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
        builder, channel_index, compute_coordinate, worker_endpoint->direction,
        (int8_t)channel->capacity));
  }

  if (ingress) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_route_ingress(
        &builder->route_builder, channel_index, shim_coordinate,
        shim_dma_channel, compute_coordinate, compute_dma_channel));
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_route_egress(
        &builder->route_builder, channel_index, compute_coordinate,
        compute_dma_channel, shim_coordinate, shim_dma_channel));
  }

  if (binding_source_channel_index == channel_index) {
    loom_aie2p_array_binding_plan_t binding_plan = {
        .binding_index = base_binding_endpoint->owner_index,
        .channel_index = channel_index,
        .shim_coordinate = shim_coordinate,
        .direction = shim_direction,
        .dma_channel = shim_dma_channel,
        .partition_lane = binding_endpoint->partition_lane,
        .partition_lane_count = binding_endpoint->partition_lane_count,
    };
    const loom_xdna_tile_facts_t* shim_tile = NULL;
    IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(
        builder->family, shim_coordinate, &shim_tile));
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
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_worker_port(
        builder, worker_endpoint, channel_index, first_slot));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_neighbor_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver, uint32_t first_slot) {
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
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
      builder, channel_index, owner, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
      (int8_t)channel->capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_worker_port(
      builder, sender, channel_index, first_slot));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_worker_port(
      builder, receiver, channel_index, first_slot));
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_routed_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver, uint32_t first_slot) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const loom_aie2p_array_worker_t* sender_worker =
      &builder->workers[sender->owner_index];
  const loom_aie2p_array_worker_t* receiver_worker =
      &builder->workers[receiver->owner_index];
  loom_xdna_tile_coordinate_t sender_dma_coordinate = {0};
  uint8_t sender_dma_channel = 0;
  const loom_aie2p_array_dma_plan_t* multicast_source_dma =
      loom_aie2p_array_multicast_source_dma(builder, channel_index);
  const bool owns_sender_dma = multicast_source_dma == NULL;
  if (multicast_source_dma != NULL) {
    sender_dma_coordinate = multicast_source_dma->coordinate;
    sender_dma_channel = multicast_source_dma->dma_channel;
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_select_compute_dma(
        builder, channel_index, sender_worker->coordinate,
        LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM, channel->capacity,
        channel->record_byte_length, /*pending_endpoint=*/NULL,
        &sender_dma_coordinate, &sender_dma_channel));
  }
  const loom_aie2p_array_pending_endpoint_t pending_sender = {
      .coordinate = sender_dma_coordinate,
      .record_byte_length = channel->record_byte_length,
      .record_count = channel->capacity,
  };
  loom_xdna_tile_coordinate_t receiver_dma_coordinate = {0};
  uint8_t receiver_dma_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_select_compute_dma(
      builder, channel_index, receiver_worker->coordinate,
      LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY, channel->capacity,
      channel->record_byte_length, owns_sender_dma ? &pending_sender : NULL,
      &receiver_dma_coordinate, &receiver_dma_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channel_slots(
      builder, channel_index, sender, receiver, &sender_dma_coordinate,
      &receiver_dma_coordinate));

  if (channel->capacity > INT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel capacity exceeds lock range");
  }
  if (owns_sender_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
        builder, channel_index, sender_dma_coordinate,
        LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND, (int8_t)channel->capacity));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
      builder, channel_index, receiver_dma_coordinate,
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE, (int8_t)channel->capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_workers(
      &builder->route_builder, channel_index, sender_dma_coordinate,
      sender_dma_channel, receiver_dma_coordinate, receiver_dma_channel));
  if (owns_sender_dma) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_worker_port(
        builder, sender, channel_index, first_slot));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_worker_port(
      builder, receiver, channel_index, first_slot));
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_channels(
    loom_aie2p_array_plan_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &builder->channels[i];
    const loom_aie2p_array_endpoint_t* sender =
        &builder->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &builder->endpoints[channel->receiver_endpoint_index];
    const uint32_t first_slot = (uint32_t)builder->channel_slot_cursor;
    switch (channel->transport) {
      case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_external_channel(
            builder, (uint32_t)i, sender, receiver, first_slot));
        break;
      }
      case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_neighbor_channel(
            builder, (uint32_t)i, sender, receiver, first_slot));
        break;
      }
      case LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA: {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_routed_channel(
            builder, (uint32_t)i, sender, receiver, first_slot));
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
      IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(
          builder->family, (loom_xdna_tile_coordinate_t){column, row},
          &state->facts));
      bank_cursor_count += state->facts->memory.bank_count;
    }
  }
  uint32_t* bank_cursors = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, bank_cursor_count, sizeof(*bank_cursors),
      (void**)&bank_cursors));
  memset(bank_cursors, 0, bank_cursor_count * sizeof(*bank_cursors));
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    builder->tile_states[i].bank_cursors = bank_cursors;
    bank_cursors += builder->tile_states[i].facts->memory.bank_count;
  }

  uint8_t* next_channels = NULL;
  const iree_host_size_t next_channel_count = tile_count * 4u;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, next_channel_count, sizeof(*next_channels),
      (void**)&next_channels));
  memset(next_channels, 0, next_channel_count * sizeof(*next_channels));
  builder->route_builder.link_channels.northbound = next_channels;
  builder->route_builder.link_channels.southbound = next_channels + tile_count;
  builder->route_builder.link_channels.westbound =
      next_channels + tile_count * 2u;
  builder->route_builder.link_channels.eastbound =
      next_channels + tile_count * 3u;
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
  if (channel_slot_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel slot count is too large");
  }

  uint64_t worker_storage_count = 0;
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_low_function_requirements_t* requirements =
        loom_aie2p_array_find_leaf(builder, builder->workers[i].entry);
    if (requirements == NULL) continue;
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
      worker_storage_count > IREE_HOST_SIZE_MAX ||
      lock_count > IREE_HOST_SIZE_MAX ||
      dma_channel_count > IREE_HOST_SIZE_MAX ||
      route_count > IREE_HOST_SIZE_MAX) {
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
      sizeof(*builder->worker_ports), (void**)&builder->worker_ports));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->channel_slot_count,
      sizeof(*builder->channel_slots), (void**)&builder->channel_slots));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->lock_count, sizeof(*builder->locks),
      (void**)&builder->locks));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->dma_channel_count,
      sizeof(*builder->dma_channels), (void**)&builder->dma_channels));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->route_count,
      sizeof(*builder->route_builder.routes),
      (void**)&builder->route_builder.routes));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->binding_plan_count,
      sizeof(*builder->binding_plans), (void**)&builder->binding_plans));

  builder->plan->worker_plans = builder->worker_plans;
  builder->plan->worker_storage = builder->worker_storage;
  builder->plan->worker_ports = builder->worker_ports;
  builder->plan->channel_slots = builder->channel_slots;
  builder->plan->locks = builder->locks;
  builder->plan->dma_channels = builder->dma_channels;
  builder->plan->routes = builder->route_builder.routes;
  builder->plan->binding_plans = builder->binding_plans;
  return loom_aie2p_array_initialize_tile_states(builder);
}

static iree_status_t loom_aie2p_array_finalize_physical_counts(
    loom_aie2p_array_plan_builder_t* builder) {
  builder->plan->worker_storage_count = builder->worker_storage_cursor;
  builder->plan->worker_port_count = builder->worker_port_cursor;
  builder->plan->channel_slot_count = builder->channel_slot_cursor;
  builder->plan->lock_count = builder->lock_cursor;
  builder->plan->dma_channel_count = builder->dma_channel_cursor;
  builder->plan->route_count = builder->route_builder.route_count;
  builder->plan->binding_plan_count = builder->binding_plan_cursor;
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_plan_build(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_aie2p_array_leaf_t* leaves, iree_host_size_t leaf_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_aie2p_array_plan_t* out_plan) {
  *out_plan = (loom_aie2p_array_plan_t){0};
  if (!module || !function_op || !arena ||
      !loom_low_func_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array planning requires one target-low function definition");
  }
  const loom_func_like_t function =
      loom_func_like_const_cast(module, function_op);
  loom_region_t* body = loom_func_like_body(function);
  if (!body || body->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array planning requires one structured entry block");
  }
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  if (contract_id >= module->strings.count ||
      !iree_string_view_equal(module->strings.entries[contract_id],
                              IREE_SV("amd.xdna.aie2p.array"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array planning requires amd.xdna.aie2p.array Low IR");
  }

  loom_aie2p_array_plan_builder_t builder = {
      .module = module,
      .function_op = function_op,
      .descriptor_set = loom_aie2p_array_descriptor_set(),
      .family = loom_xdna_npu2_array_family(),
      .leaves = leaves,
      .leaf_count = leaf_count,
      .diagnostic_emitter = diagnostic_emitter,
      .arena = arena,
      .plan = out_plan,
  };
  builder.route_builder.family = builder.family;
  *out_plan = (loom_aie2p_array_plan_t){
      .function_op = function_op,
      .family = builder.family,
  };
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&builder.facts, arena,
                                                        module->values.count));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(&builder.facts, module, function));
  const loom_block_t* block = loom_region_entry_block(body);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_count_topology(&builder, block));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_topology(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_topology(&builder, block));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_validate(
      &builder.facts, diagnostic_emitter, arena, out_plan, builder.channels));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_plan(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_workers(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channels(&builder));
  return loom_aie2p_array_finalize_physical_counts(&builder);
}
