// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
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

typedef struct loom_aie2p_array_plan_builder_t {
  const loom_module_t* module;
  const loom_op_t* function_op;
  const loom_low_descriptor_set_t* descriptor_set;
  const loom_xdna_array_family_t* family;
  const loom_aie2p_array_leaf_t* leaves;
  iree_host_size_t leaf_count;
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
  loom_aie2p_array_route_plan_t* routes;
  loom_aie2p_array_binding_plan_t* binding_plans;

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
  iree_host_size_t route_cursor;
  iree_host_size_t binding_plan_cursor;

  loom_aie2p_array_tile_state_t* tile_states;
  uint8_t* upward_link_next_channel;
  uint8_t* downward_link_next_channel;
  uint8_t* westward_link_next_channel;
  uint8_t* eastward_link_next_channel;
  uint8_t* shim_ingress_next_channel;
  uint8_t* shim_egress_next_channel;
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
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTRAIN_LOCATION:
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_GROUP:
        ++builder->plan->group_count;
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_BINDING:
        ++builder->plan->binding_count;
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER:
        ++builder->plan->worker_count;
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_RECEIVER:
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION:
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
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  loom_aie2p_array_worker_t* worker = &builder->workers[builder->worker_cursor];
  worker->value_id = loom_op_results(op)[0];
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[0], LOOM_AIE2P_ARRAY_ENTITY_GROUP,
      "worker group", &worker->group_index));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
      builder, loom_op_operands(op)[1], "worker lane", &worker->lane));
  worker->entry = loom_attr_as_symbol(
      loom_aie2p_array_attr(builder->module, op, IREE_SV("entry")));
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
  endpoint->partition_source_endpoint_index = UINT32_MAX;
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

static iree_status_t loom_aie2p_array_extract_partition(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  loom_aie2p_array_endpoint_t* endpoint =
      &builder->endpoints[builder->endpoint_cursor];
  endpoint->value_id = loom_op_results(op)[0];
  endpoint->direction = LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[0], LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
      "partition source", &endpoint->partition_source_endpoint_index));
  const loom_aie2p_array_endpoint_t* source =
      &builder->endpoints[endpoint->partition_source_endpoint_index];
  endpoint->owner_kind = source->owner_kind;
  endpoint->owner_index = source->owner_index;
  endpoint->port = source->port;
  endpoint->message_type =
      *loom_aie2p_array_result_value_type(builder->module, op);
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_exact_u32(builder, loom_op_operands(op)[1],
                                 "partition lane", &endpoint->partition_lane));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_exact_u32(
      builder, loom_op_operands(op)[2], "partition lane count",
      &endpoint->partition_lane_count));
  loom_aie2p_array_define_entity(builder, endpoint->value_id,
                                 LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
                                 (uint32_t)builder->endpoint_cursor++);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_extract_channel(
    loom_aie2p_array_plan_builder_t* builder, const loom_op_t* op) {
  loom_aie2p_array_channel_t* channel =
      &builder->channels[builder->channel_cursor];
  channel->value_id = loom_op_results(op)[0];
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[0], LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
      "channel sender", &channel->sender_endpoint_index));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_lookup_entity(
      builder, loom_op_operands(op)[1], LOOM_AIE2P_ARRAY_ENTITY_ENDPOINT,
      "channel receiver", &channel->receiver_endpoint_index));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_exact_u32(builder, loom_op_operands(op)[2],
                                 "channel capacity", &channel->capacity));
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
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_GROUP:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_group(builder, op));
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_BINDING:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_binding(builder, op));
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_worker(builder, op));
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_endpoint(
            builder, op, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND));
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_RECEIVER:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_endpoint(
            builder, op, LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE));
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_partition(builder, op));
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CHANNEL:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_channel(builder, op));
        break;
      case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTRAIN_LOCATION:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_extract_location(builder, op));
        break;
      default:
        IREE_ASSERT_UNREACHABLE("generated array descriptor ordinal");
        break;
    }
  }
  return iree_ok_status();
}

static const loom_aie2p_array_endpoint_t* loom_aie2p_array_base_endpoint(
    const loom_aie2p_array_plan_builder_t* builder,
    const loom_aie2p_array_endpoint_t* endpoint) {
  if (endpoint->partition_source_endpoint_index == UINT32_MAX) return endpoint;
  return &builder->endpoints[endpoint->partition_source_endpoint_index];
}

static iree_status_t loom_aie2p_array_validate_partition(
    const loom_aie2p_array_plan_builder_t* builder,
    const loom_aie2p_array_endpoint_t* endpoint) {
  const loom_aie2p_array_endpoint_t* source =
      loom_aie2p_array_base_endpoint(builder, endpoint);
  if (source == endpoint ||
      source->partition_source_endpoint_index != UINT32_MAX ||
      source->direction != LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND ||
      source->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING ||
      endpoint->partition_lane_count == 0 ||
      endpoint->partition_lane >= endpoint->partition_lane_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P channel partition is malformed");
  }
  if (loom_type_element_type(source->message_type) !=
          loom_type_element_type(endpoint->message_type) ||
      loom_type_rank(source->message_type) !=
          loom_type_rank(endpoint->message_type) + 1u) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P partition must remove one leading tile dimension");
  }
  uint32_t leading_dimension = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_dimension_value(
      builder, source->message_type, 0, &leading_dimension));
  if (leading_dimension != endpoint->partition_lane_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P partition count must match the leading tile dimension");
  }
  for (iree_host_size_t i = 0; i < loom_type_rank(endpoint->message_type);
       ++i) {
    uint32_t source_dimension = 0;
    uint32_t endpoint_dimension = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_dimension_value(
        builder, source->message_type, i + 1, &source_dimension));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_dimension_value(
        builder, endpoint->message_type, i, &endpoint_dimension));
    if (source_dimension != endpoint_dimension) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P partition result shape must match the source suffix");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_validate_topology(
    loom_aie2p_array_plan_builder_t* builder) {
  if (builder->plan->worker_count == 0 || builder->plan->channel_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P resident array requires at least one worker and channel");
  }

  for (iree_host_size_t i = 0; i < builder->plan->group_count; ++i) {
    const loom_aie2p_array_group_t* group = &builder->groups[i];
    if (group->lane_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P worker group cannot be empty");
    }
    for (uint32_t lane = 0; lane < group->lane_count; ++lane) {
      iree_host_size_t match_count = 0;
      for (iree_host_size_t j = 0; j < builder->plan->worker_count; ++j) {
        const loom_aie2p_array_worker_t* worker = &builder->workers[j];
        if (worker->group_index == i && worker->lane == lane) ++match_count;
      }
      if (match_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P worker group must instantiate every lane exactly once");
      }
    }
  }

  for (iree_host_size_t i = 0; i < builder->plan->binding_count; ++i) {
    for (iree_host_size_t j = i + 1; j < builder->plan->binding_count; ++j) {
      if (builder->bindings[i].ordinal == builder->bindings[j].ordinal) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P binding ordinal is duplicated");
      }
    }
  }

  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &builder->workers[i];
    if (worker->group_index >= builder->plan->group_count ||
        worker->lane >= builder->groups[worker->group_index].lane_count ||
        worker->coordinate.column == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P worker is incompletely instantiated");
    }
    const loom_xdna_tile_facts_t* tile_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(
        builder->family, worker->coordinate, &tile_facts));
    if (tile_facts->kind != LOOM_XDNA_TILE_KIND_COMPUTE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P worker must occupy a compute tile");
    }
    for (iree_host_size_t j = i + 1; j < builder->plan->worker_count; ++j) {
      if (worker->coordinate.column == builder->workers[j].coordinate.column &&
          worker->coordinate.row == builder->workers[j].coordinate.row) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P workers cannot share a compute tile");
      }
    }
  }

  uint32_t* channel_use_counts = NULL;
  uint32_t* partition_use_counts = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->endpoint_count,
      sizeof(*channel_use_counts), (void**)&channel_use_counts));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->endpoint_count,
      sizeof(*partition_use_counts), (void**)&partition_use_counts));
  memset(channel_use_counts, 0,
         builder->plan->endpoint_count * sizeof(*channel_use_counts));
  memset(partition_use_counts, 0,
         builder->plan->endpoint_count * sizeof(*partition_use_counts));

  for (iree_host_size_t i = 0; i < builder->plan->endpoint_count; ++i) {
    const loom_aie2p_array_endpoint_t* endpoint = &builder->endpoints[i];
    if (endpoint->partition_source_endpoint_index != UINT32_MAX) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_array_validate_partition(builder, endpoint));
      ++partition_use_counts[endpoint->partition_source_endpoint_index];
    }
    for (iree_host_size_t j = i + 1; j < builder->plan->endpoint_count; ++j) {
      const loom_aie2p_array_endpoint_t* other = &builder->endpoints[j];
      if (endpoint->partition_source_endpoint_index == UINT32_MAX &&
          other->partition_source_endpoint_index == UINT32_MAX &&
          endpoint->owner_kind == other->owner_kind &&
          endpoint->owner_index == other->owner_index &&
          endpoint->port == other->port) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P owner port is defined more than once");
      }
    }
  }

  for (iree_host_size_t i = 0; i < builder->plan->channel_count; ++i) {
    loom_aie2p_array_channel_t* channel = &builder->channels[i];
    if (channel->capacity == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P channel capacity cannot be zero");
    }
    const loom_aie2p_array_endpoint_t* sender =
        &builder->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &builder->endpoints[channel->receiver_endpoint_index];
    if (sender->direction != LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND ||
        receiver->direction != LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE ||
        !loom_type_equal(sender->message_type, receiver->message_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P channel endpoints must have matching typed directions");
    }
    ++channel_use_counts[channel->sender_endpoint_index];
    ++channel_use_counts[channel->receiver_endpoint_index];

    const loom_aie2p_array_endpoint_t* base_sender =
        loom_aie2p_array_base_endpoint(builder, sender);
    if (base_sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING &&
        receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const loom_aie2p_array_binding_t* binding =
          &builder->bindings[base_sender->owner_index];
      if ((binding->access & LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ) == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P input channel uses a non-readable binding");
      }
      channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA;
    } else if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
               receiver->owner_kind ==
                   LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
      const loom_aie2p_array_binding_t* binding =
          &builder->bindings[receiver->owner_index];
      if ((binding->access & LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE) == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P output channel uses a non-writable binding");
      }
      channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA;
    } else if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
               receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const loom_aie2p_array_worker_t* sender_worker =
          &builder->workers[sender->owner_index];
      const loom_aie2p_array_worker_t* receiver_worker =
          &builder->workers[receiver->owner_index];
      const int row_delta = (int)sender_worker->coordinate.row -
                            (int)receiver_worker->coordinate.row;
      if (sender_worker->coordinate.column !=
              receiver_worker->coordinate.column ||
          (row_delta != -1 && row_delta != 1)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P neighbor-memory channels require vertically adjacent "
            "worker tiles");
      }
      channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AIE2P channel ownership");
    }
  }

  for (iree_host_size_t i = 0; i < builder->plan->endpoint_count; ++i) {
    const loom_aie2p_array_endpoint_t* endpoint = &builder->endpoints[i];
    if (endpoint->partition_source_endpoint_index != UINT32_MAX) {
      if (channel_use_counts[i] != 1 || partition_use_counts[i] != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P partition endpoint must feed exactly one channel");
      }
      continue;
    }
    if (partition_use_counts[i] == 0) {
      if (channel_use_counts[i] != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P direct endpoint must feed exactly one channel");
      }
      continue;
    }
    if (channel_use_counts[i] != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P partition source cannot also feed a channel directly");
    }
    uint32_t partition_lane_count = 0;
    for (iree_host_size_t j = 0; j < builder->plan->endpoint_count; ++j) {
      const loom_aie2p_array_endpoint_t* partition = &builder->endpoints[j];
      if (partition->partition_source_endpoint_index == i) {
        partition_lane_count = partition->partition_lane_count;
        break;
      }
    }
    if (partition_use_counts[i] != partition_lane_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P partition source must materialize every lane exactly once");
    }
    for (uint32_t lane = 0; lane < partition_lane_count; ++lane) {
      iree_host_size_t match_count = 0;
      for (iree_host_size_t j = 0; j < builder->plan->endpoint_count; ++j) {
        const loom_aie2p_array_endpoint_t* partition = &builder->endpoints[j];
        if (partition->partition_source_endpoint_index == i &&
            partition->partition_lane == lane) {
          ++match_count;
        }
      }
      if (match_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P partition lanes must be unique and complete");
      }
    }
  }
  return iree_ok_status();
}

static const loom_aie2p_leaf_contribution_t* loom_aie2p_array_find_leaf(
    const loom_aie2p_array_plan_builder_t* builder, loom_symbol_ref_t entry) {
  const loom_aie2p_leaf_contribution_t* result = NULL;
  for (iree_host_size_t i = 0; i < builder->leaf_count; ++i) {
    if (loom_aie2p_array_symbol_ref_equal(builder->leaves[i].entry, entry)) {
      IREE_ASSERT(result == NULL, "leaf table entries must be unique");
      result = builder->leaves[i].contribution;
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

static iree_status_t loom_aie2p_array_allocate_tile_storage(
    loom_aie2p_array_tile_state_t* state, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_owner_offset) {
  const uint32_t bank_capacity =
      state->facts->memory.local_capacity / state->facts->memory.bank_count;
  for (uint8_t attempt = 0; attempt < state->facts->memory.bank_count;
       ++attempt) {
    const uint8_t bank = (uint8_t)((state->next_bank + attempt) %
                                   state->facts->memory.bank_count);
    uint64_t cursor = state->bank_cursors[bank];
    if (!iree_checked_align_u64(cursor, alignment, &cursor) ||
        cursor + byte_length > bank_capacity) {
      continue;
    }
    state->bank_cursors[bank] = (uint32_t)(cursor + byte_length);
    state->next_bank = (uint8_t)((bank + 1u) % state->facts->memory.bank_count);
    *out_owner_offset = bank * bank_capacity + (uint32_t)cursor;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "AIE2P tile local memory banks are exhausted");
}

static iree_status_t loom_aie2p_array_allocate_lock_pair(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate, int8_t credit_count,
    bool shim_peer) {
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
      .consumer_ready = 0,
      .shim_peer = shim_peer ? 1 : 0,
  };
  builder->locks[builder->lock_cursor++] = (loom_aie2p_array_lock_plan_t){
      .channel_index = channel_index,
      .coordinate = coordinate,
      .lock_id = ready_lock,
      .initial_value = 0,
      .consumer_ready = 1,
      .shim_peer = shim_peer ? 1 : 0,
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

static iree_status_t loom_aie2p_array_allocate_link_channel(
    uint8_t* next_channel, uint8_t capacity, uint8_t* out_channel) {
  if (*next_channel < capacity) {
    *out_channel = (*next_channel)++;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "AIE2P stream link channels are exhausted");
}

static iree_status_t loom_aie2p_array_port_capacity(
    const loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate,
    loom_xdna_stream_direction_t direction, loom_xdna_stream_port_t port,
    uint8_t* out_capacity) {
  const loom_xdna_tile_facts_t* tile_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_array_tile_facts(builder->family, coordinate, &tile_facts));
  const loom_xdna_stream_port_range_t* range = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_array_stream_port_range(
      builder->family, tile_facts->kind, direction, port, &range));
  *out_capacity = range->count;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_allocate_physical_link(
    loom_aie2p_array_plan_builder_t* builder,
    loom_xdna_tile_coordinate_t source_coordinate,
    loom_xdna_stream_port_t source_port,
    loom_xdna_tile_coordinate_t destination_coordinate,
    loom_xdna_stream_port_t destination_port, uint8_t* next_channel,
    uint8_t* out_channel) {
  uint8_t source_capacity = 0;
  uint8_t destination_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
      builder, source_coordinate, LOOM_XDNA_STREAM_DIRECTION_MASTER,
      source_port, &source_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
      builder, destination_coordinate, LOOM_XDNA_STREAM_DIRECTION_SLAVE,
      destination_port, &destination_capacity));
  const uint8_t capacity = source_capacity < destination_capacity
                               ? source_capacity
                               : destination_capacity;
  return loom_aie2p_array_allocate_link_channel(next_channel, capacity,
                                                out_channel);
}

static iree_status_t loom_aie2p_array_append_route(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_switch_kind_t switch_kind,
    loom_xdna_stream_port_t source_port, uint8_t source_channel,
    loom_xdna_stream_port_t destination_port, uint8_t destination_channel) {
  if (switch_kind == LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH) {
    uint8_t source_capacity = 0;
    uint8_t destination_capacity = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
        builder, coordinate, LOOM_XDNA_STREAM_DIRECTION_SLAVE, source_port,
        &source_capacity));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
        builder, coordinate, LOOM_XDNA_STREAM_DIRECTION_MASTER,
        destination_port, &destination_capacity));
    if (source_channel >= source_capacity ||
        destination_channel >= destination_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P route exceeds a stream port range");
    }
  }
  builder->routes[builder->route_cursor++] = (loom_aie2p_array_route_plan_t){
      .channel_index = channel_index,
      .coordinate = coordinate,
      .switch_kind = switch_kind,
      .source_port = source_port,
      .source_channel = source_channel,
      .destination_port = destination_port,
      .destination_channel = destination_channel,
  };
  return iree_ok_status();
}

static iree_host_size_t loom_aie2p_array_vertical_link_index(
    const loom_aie2p_array_plan_builder_t* builder, uint16_t column,
    uint16_t lower_row) {
  return (iree_host_size_t)lower_row * builder->family->column_count + column;
}

static iree_host_size_t loom_aie2p_array_horizontal_link_index(
    const loom_aie2p_array_plan_builder_t* builder, uint16_t lower_column,
    uint16_t row) {
  return (iree_host_size_t)row * builder->family->column_count + lower_column;
}

static iree_status_t loom_aie2p_array_plan_ingress_route(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel) {
  uint8_t current_channel = 0;
  uint8_t shim_south_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
      builder, shim_coordinate, LOOM_XDNA_STREAM_DIRECTION_SLAVE,
      LOOM_XDNA_STREAM_PORT_SOUTH, &shim_south_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_link_channel(
      &builder->shim_ingress_next_channel[shim_coordinate.column],
      shim_south_capacity, &current_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
      builder, channel_index, shim_coordinate,
      LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX, LOOM_XDNA_STREAM_PORT_DMA,
      shim_dma_channel, LOOM_XDNA_STREAM_PORT_NORTH, current_channel));

  uint16_t column = shim_coordinate.column;
  loom_xdna_stream_port_t incoming_port = LOOM_XDNA_STREAM_PORT_SOUTH;
  while (column != worker_coordinate.column) {
    const bool move_west = column > worker_coordinate.column;
    const uint16_t next_column =
        move_west ? (uint16_t)(column - 1u) : (uint16_t)(column + 1u);
    const uint16_t lower_column = column < next_column ? column : next_column;
    const iree_host_size_t link_index =
        loom_aie2p_array_horizontal_link_index(builder, lower_column, 0);
    uint8_t* next_channel =
        move_west ? &builder->westward_link_next_channel[link_index]
                  : &builder->eastward_link_next_channel[link_index];
    const loom_xdna_stream_port_t outgoing_port =
        move_west ? LOOM_XDNA_STREAM_PORT_WEST : LOOM_XDNA_STREAM_PORT_EAST;
    const loom_xdna_stream_port_t next_incoming_port =
        move_west ? LOOM_XDNA_STREAM_PORT_EAST : LOOM_XDNA_STREAM_PORT_WEST;
    uint8_t link_channel = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_link(
        builder, (loom_xdna_tile_coordinate_t){column, 0}, outgoing_port,
        (loom_xdna_tile_coordinate_t){next_column, 0}, next_incoming_port,
        next_channel, &link_channel));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
        builder, channel_index, (loom_xdna_tile_coordinate_t){column, 0},
        LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, incoming_port,
        current_channel, outgoing_port, link_channel));
    column = next_column;
    incoming_port = next_incoming_port;
    current_channel = link_channel;
  }

  for (uint16_t row = 0; row <= worker_coordinate.row; ++row) {
    const loom_xdna_tile_coordinate_t coordinate = {column, row};
    if (row == worker_coordinate.row) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
          builder, channel_index, coordinate,
          LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH,
          LOOM_XDNA_STREAM_PORT_SOUTH, current_channel,
          LOOM_XDNA_STREAM_PORT_DMA, worker_dma_channel));
      break;
    }
    const loom_xdna_tile_coordinate_t next_coordinate = {
        column,
        (uint16_t)(row + 1u),
    };
    const iree_host_size_t link_index =
        loom_aie2p_array_vertical_link_index(builder, column, row);
    uint8_t link_channel = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_link(
        builder, coordinate, LOOM_XDNA_STREAM_PORT_NORTH, next_coordinate,
        LOOM_XDNA_STREAM_PORT_SOUTH,
        &builder->upward_link_next_channel[link_index], &link_channel));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
        builder, channel_index, coordinate,
        LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, incoming_port,
        current_channel, LOOM_XDNA_STREAM_PORT_NORTH, link_channel));
    incoming_port = LOOM_XDNA_STREAM_PORT_SOUTH;
    current_channel = link_channel;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_egress_route(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel) {
  uint16_t row = worker_coordinate.row;
  uint16_t column = worker_coordinate.column;
  uint8_t current_channel = 0;
  loom_xdna_stream_port_t incoming_port = LOOM_XDNA_STREAM_PORT_DMA;
  while (row > 0) {
    const loom_xdna_tile_coordinate_t coordinate = {column, row};
    const loom_xdna_tile_coordinate_t next_coordinate = {
        column,
        (uint16_t)(row - 1u),
    };
    const iree_host_size_t link_index = loom_aie2p_array_vertical_link_index(
        builder, column, (uint16_t)(row - 1u));
    uint8_t link_channel = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_link(
        builder, coordinate, LOOM_XDNA_STREAM_PORT_SOUTH, next_coordinate,
        LOOM_XDNA_STREAM_PORT_NORTH,
        &builder->downward_link_next_channel[link_index], &link_channel));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
        builder, channel_index, coordinate,
        LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, incoming_port,
        row == worker_coordinate.row ? worker_dma_channel : current_channel,
        LOOM_XDNA_STREAM_PORT_SOUTH, link_channel));
    --row;
    incoming_port = LOOM_XDNA_STREAM_PORT_NORTH;
    current_channel = link_channel;
  }

  while (column != shim_coordinate.column) {
    const bool move_west = column > shim_coordinate.column;
    const uint16_t next_column =
        move_west ? (uint16_t)(column - 1u) : (uint16_t)(column + 1u);
    const uint16_t lower_column = column < next_column ? column : next_column;
    const iree_host_size_t link_index =
        loom_aie2p_array_horizontal_link_index(builder, lower_column, 0);
    uint8_t* next_channel =
        move_west ? &builder->westward_link_next_channel[link_index]
                  : &builder->eastward_link_next_channel[link_index];
    const loom_xdna_stream_port_t outgoing_port =
        move_west ? LOOM_XDNA_STREAM_PORT_WEST : LOOM_XDNA_STREAM_PORT_EAST;
    const loom_xdna_stream_port_t next_incoming_port =
        move_west ? LOOM_XDNA_STREAM_PORT_EAST : LOOM_XDNA_STREAM_PORT_WEST;
    uint8_t link_channel = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_link(
        builder, (loom_xdna_tile_coordinate_t){column, 0}, outgoing_port,
        (loom_xdna_tile_coordinate_t){next_column, 0}, next_incoming_port,
        next_channel, &link_channel));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
        builder, channel_index, (loom_xdna_tile_coordinate_t){column, 0},
        LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, incoming_port,
        current_channel, outgoing_port, link_channel));
    column = next_column;
    incoming_port = next_incoming_port;
    current_channel = link_channel;
  }

  uint8_t shim_link_channel = 0;
  uint8_t shim_south_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
      builder, shim_coordinate, LOOM_XDNA_STREAM_DIRECTION_MASTER,
      LOOM_XDNA_STREAM_PORT_SOUTH, &shim_south_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_link_channel(
      &builder->shim_egress_next_channel[shim_coordinate.column],
      shim_south_capacity, &shim_link_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
      builder, channel_index, shim_coordinate,
      LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, incoming_port,
      current_channel, LOOM_XDNA_STREAM_PORT_SOUTH, shim_link_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
      builder, channel_index, shim_coordinate,
      LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX, LOOM_XDNA_STREAM_PORT_NORTH,
      shim_link_channel, LOOM_XDNA_STREAM_PORT_DMA, shim_dma_channel));
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_validate_worker_leaf(
    const loom_aie2p_array_plan_builder_t* builder, uint32_t worker_index,
    const loom_aie2p_leaf_contribution_t* contribution) {
  if (contribution == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AIE2P worker has no compiled leaf contribution");
  }
  const loom_aie2p_leaf_realization_t* realization = &contribution->realization;
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
  const loom_region_t* worker_body =
      loom_low_func_def_body(worker_symbol->defining_op);
  iree_host_size_t return_count = 0;
  const loom_block_t* worker_block = NULL;
  loom_region_for_each_block(worker_body, worker_block) {
    const loom_op_t* worker_op = NULL;
    loom_block_for_each_op(worker_block, worker_op) {
      if (loom_low_return_isa(worker_op)) ++return_count;
    }
  }
  if (return_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P worker entry must return after one channel firing");
  }
  const loom_aie2p_array_tile_state_t* tile_state = loom_aie2p_array_tile_state(
      (loom_aie2p_array_plan_builder_t*)builder, worker->coordinate);
  if (realization->code.byte_length >
      tile_state->facts->memory.program_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P worker code exceeds tile program memory");
  }

  for (iree_host_size_t i = 0; i < builder->plan->endpoint_count; ++i) {
    const loom_aie2p_array_endpoint_t* endpoint = &builder->endpoints[i];
    if (endpoint->partition_source_endpoint_index == UINT32_MAX &&
        endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
        endpoint->owner_index == worker_index) {
      iree_host_size_t import_match_count = 0;
      for (iree_host_size_t j = 0; j < realization->resource_import_count;
           ++j) {
        if (realization->resource_imports[j].index == endpoint->port) {
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
  for (iree_host_size_t i = 0; i < realization->resource_import_count; ++i) {
    const loom_aie2p_leaf_resource_import_t* resource =
        &realization->resource_imports[i];
    iree_host_size_t endpoint_match_count = 0;
    for (iree_host_size_t j = 0; j < builder->plan->endpoint_count; ++j) {
      const loom_aie2p_array_endpoint_t* endpoint = &builder->endpoints[j];
      if (endpoint->partition_source_endpoint_index == UINT32_MAX &&
          endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
          endpoint->owner_index == worker_index &&
          endpoint->port == resource->index) {
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
    const loom_aie2p_leaf_contribution_t* contribution =
        loom_aie2p_array_find_leaf(builder, worker->entry);
    IREE_RETURN_IF_ERROR(loom_aie2p_array_validate_worker_leaf(
        builder, (uint32_t)i, contribution));
    builder->worker_plans[i] = (loom_aie2p_array_worker_plan_t){
        .worker_index = (uint32_t)i,
        .coordinate = worker->coordinate,
        .contribution = contribution,
    };

    loom_aie2p_array_tile_state_t* tile_state =
        loom_aie2p_array_tile_state(builder, worker->coordinate);
    for (iree_host_size_t j = 0;
         j < contribution->realization.storage_domain_count; ++j) {
      const loom_aie2p_leaf_storage_domain_t* domain =
          &contribution->realization.storage_domains[j];
      const loom_aie2p_leaf_storage_requirement_t* requirement =
          loom_aie2p_leaf_storage_requirement(&contribution->realization,
                                              domain->storage_space);
      if (requirement->byte_length > UINT32_MAX ||
          requirement->minimum_alignment > UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "AIE2P worker storage requirement is not representable");
      }
      uint32_t owner_offset = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_tile_storage(
          tile_state, (uint32_t)requirement->byte_length,
          (uint32_t)requirement->minimum_alignment, &owner_offset));
      uint32_t load_address = 0;
      IREE_RETURN_IF_ERROR(loom_xdna_array_form_load_address(
          builder->family, worker->coordinate, LOOM_XDNA_MEMORY_SPACE_DATA,
          worker->coordinate, owner_offset, (uint32_t)requirement->byte_length,
          &load_address));
      builder->worker_storage[builder->worker_storage_cursor++] =
          (loom_aie2p_array_worker_storage_plan_t){
              .worker_index = (uint32_t)i,
              .storage_space = domain->storage_space,
              .owner_offset = owner_offset,
              .load_address = load_address,
              .byte_length = (uint32_t)requirement->byte_length,
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

static iree_status_t loom_aie2p_array_plan_channel_slots(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t owner,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  loom_aie2p_array_tile_state_t* owner_state =
      loom_aie2p_array_tile_state(builder, owner);
  for (uint32_t slot = 0; slot < channel->capacity; ++slot) {
    uint32_t owner_offset = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_tile_storage(
        owner_state, channel->record_byte_length,
        LOOM_AIE2P_ARRAY_CHANNEL_ALIGNMENT, &owner_offset));
    uint32_t sender_address = UINT32_MAX;
    uint32_t receiver_address = UINT32_MAX;
    if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      IREE_RETURN_IF_ERROR(loom_xdna_array_form_load_address(
          builder->family, builder->workers[sender->owner_index].coordinate,
          LOOM_XDNA_MEMORY_SPACE_DATA, owner, owner_offset,
          channel->record_byte_length, &sender_address));
    }
    if (receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      IREE_RETURN_IF_ERROR(loom_xdna_array_form_load_address(
          builder->family, builder->workers[receiver->owner_index].coordinate,
          LOOM_XDNA_MEMORY_SPACE_DATA, owner, owner_offset,
          channel->record_byte_length, &receiver_address));
    }
    builder->channel_slots[builder->channel_slot_cursor++] =
        (loom_aie2p_array_channel_slot_t){
            .channel_index = channel_index,
            .slot = slot,
            .owner = owner,
            .owner_offset = owner_offset,
            .byte_length = channel->record_byte_length,
            .sender_load_address = sender_address,
            .receiver_load_address = receiver_address,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_external_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver, uint32_t first_slot) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const bool ingress =
      loom_aie2p_array_base_endpoint(builder, sender)->owner_kind ==
      LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING;
  const loom_aie2p_array_endpoint_t* binding_endpoint =
      ingress ? sender : receiver;
  const loom_aie2p_array_endpoint_t* base_binding_endpoint =
      loom_aie2p_array_base_endpoint(builder, binding_endpoint);
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
  uint8_t compute_dma_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_dma(
      builder, channel_index, worker->coordinate, compute_direction,
      channel->capacity, /*shim_side=*/false, &compute_dma_channel));
  loom_xdna_tile_coordinate_t shim_coordinate = {0};
  uint8_t shim_dma_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_select_shim_dma(
      builder, channel_index, worker->coordinate.column, shim_direction,
      channel->capacity, &shim_coordinate, &shim_dma_channel));

  if (channel->capacity > INT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel capacity exceeds lock range");
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
      builder, channel_index, worker->coordinate, (int8_t)channel->capacity,
      /*shim_peer=*/false));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
      builder, channel_index, shim_coordinate, 0, /*shim_peer=*/true));

  if (ingress) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_ingress_route(
        builder, channel_index, shim_coordinate, shim_dma_channel,
        worker->coordinate, compute_dma_channel));
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_egress_route(
        builder, channel_index, worker->coordinate, compute_dma_channel,
        shim_coordinate, shim_dma_channel));
  }

  builder->binding_plans[builder->binding_plan_cursor++] =
      (loom_aie2p_array_binding_plan_t){
          .binding_index = base_binding_endpoint->owner_index,
          .channel_index = channel_index,
          .shim_coordinate = shim_coordinate,
          .direction = shim_direction,
          .dma_channel = shim_dma_channel,
          .partition_lane = binding_endpoint->partition_lane,
          .partition_lane_count = binding_endpoint->partition_lane_count,
      };
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_worker_port(
      builder, worker_endpoint, channel_index, first_slot));
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_neighbor_channel(
    loom_aie2p_array_plan_builder_t* builder, uint32_t channel_index,
    const loom_aie2p_array_endpoint_t* sender,
    const loom_aie2p_array_endpoint_t* receiver, uint32_t first_slot) {
  const loom_aie2p_array_channel_t* channel = &builder->channels[channel_index];
  const loom_xdna_tile_coordinate_t owner =
      builder->workers[sender->owner_index].coordinate;
  if (channel->capacity > INT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel capacity exceeds lock range");
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_lock_pair(
      builder, channel_index, owner, (int8_t)channel->capacity,
      /*shim_peer=*/false));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_worker_port(
      builder, sender, channel_index, first_slot));
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
    loom_xdna_tile_coordinate_t owner = {0};
    if (channel->transport == LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA) {
      const loom_aie2p_array_endpoint_t* worker_endpoint =
          sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER
              ? sender
              : receiver;
      owner = builder->workers[worker_endpoint->owner_index].coordinate;
    } else {
      owner = builder->workers[sender->owner_index].coordinate;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channel_slots(
        builder, (uint32_t)i, owner, sender, receiver));
    if (channel->transport == LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_external_channel(
          builder, (uint32_t)i, sender, receiver, first_slot));
    } else {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_neighbor_channel(
          builder, (uint32_t)i, sender, receiver, first_slot));
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
  const iree_host_size_t next_channel_count = tile_count * 6u;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, next_channel_count, sizeof(*next_channels),
      (void**)&next_channels));
  memset(next_channels, 0, next_channel_count * sizeof(*next_channels));
  builder->upward_link_next_channel = next_channels;
  builder->downward_link_next_channel = next_channels + tile_count;
  builder->westward_link_next_channel = next_channels + tile_count * 2u;
  builder->eastward_link_next_channel = next_channels + tile_count * 3u;
  builder->shim_ingress_next_channel = next_channels + tile_count * 4u;
  builder->shim_egress_next_channel = next_channels + tile_count * 5u;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_allocate_physical_plan(
    loom_aie2p_array_plan_builder_t* builder) {
  uint64_t channel_slot_count = 0;
  iree_host_size_t external_channel_count = 0;
  iree_host_size_t neighbor_channel_count = 0;
  for (iree_host_size_t i = 0; i < builder->plan->channel_count; ++i) {
    if (!iree_checked_add_u64(channel_slot_count, builder->channels[i].capacity,
                              &channel_slot_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P channel slot count overflowed");
    }
    if (builder->channels[i].transport ==
        LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA) {
      ++external_channel_count;
    } else {
      ++neighbor_channel_count;
    }
  }
  if (channel_slot_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P channel slot count is too large");
  }

  uint64_t worker_storage_count = 0;
  for (iree_host_size_t i = 0; i < builder->plan->worker_count; ++i) {
    const loom_aie2p_leaf_contribution_t* contribution =
        loom_aie2p_array_find_leaf(builder, builder->workers[i].entry);
    if (contribution != NULL &&
        !iree_checked_add_u64(worker_storage_count,
                              contribution->realization.storage_domain_count,
                              &worker_storage_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P worker storage count overflowed");
    }
  }
  uint64_t lock_count = 0;
  uint64_t neighbor_lock_count = 0;
  uint64_t dma_channel_count = 0;
  uint64_t route_count = 0;
  if (!iree_checked_mul_u64(external_channel_count, 4u, &lock_count) ||
      !iree_checked_mul_u64(neighbor_channel_count, 2u, &neighbor_lock_count) ||
      !iree_checked_add_u64(lock_count, neighbor_lock_count, &lock_count) ||
      !iree_checked_mul_u64(external_channel_count, 2u, &dma_channel_count) ||
      !iree_checked_mul_u64(
          external_channel_count,
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
      builder->arena, builder->plan->route_count, sizeof(*builder->routes),
      (void**)&builder->routes));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_array(
      builder->arena, builder->plan->binding_plan_count,
      sizeof(*builder->binding_plans), (void**)&builder->binding_plans));

  builder->plan->worker_plans = builder->worker_plans;
  builder->plan->worker_storage = builder->worker_storage;
  builder->plan->worker_ports = builder->worker_ports;
  builder->plan->channel_slots = builder->channel_slots;
  builder->plan->locks = builder->locks;
  builder->plan->dma_channels = builder->dma_channels;
  builder->plan->routes = builder->routes;
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
  builder->plan->route_count = builder->route_cursor;
  builder->plan->binding_plan_count = builder->binding_plan_cursor;
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_plan_build(const loom_module_t* module,
                                          const loom_op_t* function_op,
                                          const loom_aie2p_array_leaf_t* leaves,
                                          iree_host_size_t leaf_count,
                                          iree_arena_allocator_t* arena,
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
      .arena = arena,
      .plan = out_plan,
  };
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
  IREE_RETURN_IF_ERROR(loom_aie2p_array_validate_topology(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_plan(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_workers(&builder));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_channels(&builder));
  return loom_aie2p_array_finalize_physical_counts(&builder);
}
