// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/resident.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

typedef enum loom_aie2p_array_lock_role_e {
  LOOM_AIE2P_ARRAY_LOCK_ROLE_CREDIT = 0,
  LOOM_AIE2P_ARRAY_LOCK_ROLE_READY = 1,
} loom_aie2p_array_lock_role_t;

typedef struct loom_aie2p_array_resident_builder_t {
  // Module receiving all resident worker functions.
  loom_module_t* module;
  // Trusted physical plan consumed by materialization.
  const loom_aie2p_array_plan_t* plan;
  // Scratch arena for names, remaps, and ring state.
  iree_arena_allocator_t* arena;
  // AIE2P core descriptor set used by generated packets.
  const loom_low_descriptor_set_t* descriptor_set;
  // Interned `i` immediate field name.
  loom_string_id_t integer_immediate_name;
  // Interned `id` lock-selector field name.
  loom_string_id_t lock_selector_name;
  // Scalar register type carrying signed lock deltas.
  loom_type_t lock_delta_type;
} loom_aie2p_array_resident_builder_t;

typedef struct loom_aie2p_array_resident_port_state_t {
  // Physical channel port represented by this loop-carried state.
  const loom_aie2p_array_worker_port_plan_t* port;
  // Low register type carrying endpoint-visible local addresses.
  loom_type_t address_type;
  // Current local address argument on the firing header.
  loom_value_id_t current_address;
  // Current ring slot argument, or invalid for a single-slot channel.
  loom_value_id_t current_slot;
  // Address selected for the next firing.
  loom_value_id_t next_address;
  // Ring slot selected for the next firing, or invalid when unused.
  loom_value_id_t next_slot;
} loom_aie2p_array_resident_port_state_t;

typedef struct loom_aie2p_array_resident_address_mask_t {
  // XOR mask mapping either address in a two-slot ring to the other.
  uint32_t mask;
  // Materialized scalar value shared by ports using the same mask.
  loom_value_id_t value;
} loom_aie2p_array_resident_address_mask_t;

static iree_host_size_t loom_aie2p_array_resident_worker_port_count(
    const loom_aie2p_array_plan_t* plan, uint32_t worker_index) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < plan->worker_port_count; ++i) {
    if (plan->worker_ports[i].worker_index == worker_index) ++count;
  }
  return count;
}

static iree_host_size_t loom_aie2p_array_resident_state_argument_count(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    const loom_aie2p_array_channel_t* channel =
        &plan->channels[port_states[i].port->channel_index];
    count += channel->capacity > 2 ? 2 : 1;
  }
  return count;
}

static const loom_aie2p_array_worker_port_plan_t*
loom_aie2p_array_resident_find_port(const loom_aie2p_array_plan_t* plan,
                                    uint32_t worker_index,
                                    uint64_t resource_index) {
  const loom_aie2p_array_worker_port_plan_t* result = NULL;
  for (iree_host_size_t i = 0; i < plan->worker_port_count; ++i) {
    const loom_aie2p_array_worker_port_plan_t* port = &plan->worker_ports[i];
    if (port->worker_index == worker_index && port->port == resource_index) {
      IREE_ASSERT(result == NULL &&
                  "worker resource must have exactly one planned port");
      result = port;
    }
  }
  IREE_ASSERT(result != NULL &&
              "worker resource must have exactly one planned port");
  return result;
}

static uint32_t loom_aie2p_array_resident_port_slot_address(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_worker_port_plan_t* port, uint32_t slot) {
  IREE_ASSERT_LT(port->channel_index, plan->channel_count);
  const loom_aie2p_array_channel_t* channel =
      &plan->channels[port->channel_index];
  IREE_ASSERT_LT(slot, channel->capacity);
  const iree_host_size_t slot_index =
      (iree_host_size_t)port->first_channel_slot + slot;
  IREE_ASSERT_LT(slot_index, plan->channel_slot_count);
  const loom_aie2p_array_channel_slot_t* channel_slot =
      &plan->channel_slots[slot_index];
  const uint32_t address =
      port->direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND
          ? channel_slot->sender_storage.load_address
          : channel_slot->receiver_storage.load_address;
  IREE_ASSERT_NE(address, UINT32_MAX);
  return address;
}

static const loom_aie2p_array_lock_plan_t* loom_aie2p_array_resident_find_lock(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_worker_port_plan_t* port,
    loom_aie2p_array_lock_role_t role) {
  const loom_aie2p_array_channel_t* channel =
      &plan->channels[port->channel_index];
  const loom_aie2p_array_endpoint_direction_t ring_endpoint_direction =
      channel->transport == LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY
          ? LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND
          : port->direction;
  const loom_aie2p_array_lock_plan_t* result = NULL;
  for (iree_host_size_t i = 0; i < plan->lock_count; ++i) {
    const loom_aie2p_array_lock_plan_t* lock = &plan->locks[i];
    if (lock->channel_index == port->channel_index &&
        lock->ring_endpoint_direction == ring_endpoint_direction &&
        lock->consumer_ready == (role == LOOM_AIE2P_ARRAY_LOCK_ROLE_READY)) {
      IREE_ASSERT(result == NULL &&
                  "channel role must have exactly one canonical ring lock");
      result = lock;
    }
  }
  IREE_ASSERT(result != NULL &&
              "channel role must have exactly one canonical ring lock");
  return result;
}

static iree_status_t loom_aie2p_array_resident_add_symbol(
    loom_aie2p_array_resident_builder_t* builder, uint32_t worker_index,
    loom_symbol_ref_t* out_ref) {
  const loom_func_like_t array_function =
      loom_func_like_const_cast(builder->module, builder->plan->function_op);
  const loom_symbol_ref_t array_ref = loom_func_like_callee(array_function);
  IREE_ASSERT_EQ(array_ref.module_id, 0u);
  IREE_ASSERT_LT(array_ref.symbol_id, builder->module->symbols.count);
  const loom_string_id_t array_name_id =
      builder->module->symbols.entries[array_ref.symbol_id].name_id;
  IREE_ASSERT_LT(array_name_id, builder->module->strings.count);
  const iree_string_view_t array_name =
      builder->module->strings.entries[array_name_id];
  const iree_string_view_t infix = IREE_SV("$worker$");
  iree_host_size_t name_capacity = 0;
  if (!iree_host_size_checked_add(array_name.size, infix.size + 11,
                                  &name_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P resident worker symbol name overflow");
  }
  char* name_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(builder->arena, name_capacity,
                                           (void**)&name_storage));
  memcpy(name_storage, array_name.data, array_name.size);
  memcpy(name_storage + array_name.size, infix.data, infix.size);
  const int suffix_length =
      snprintf(name_storage + array_name.size + infix.size, 11, "%" PRIu32,
               worker_index);
  if (suffix_length < 0 || suffix_length >= 11) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P resident worker symbol suffix overflow");
  }
  const iree_host_size_t name_length =
      array_name.size + infix.size + (iree_host_size_t)suffix_length;
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      builder->module, iree_make_string_view(name_storage, name_length),
      &name_id));
  if (loom_module_find_symbol(builder->module, name_id) !=
      LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "AIE2P resident worker symbol already exists");
  }
  out_ref->module_id = 0;
  return loom_module_add_symbol(builder->module, name_id, &out_ref->symbol_id);
}

static iree_status_t loom_aie2p_array_resident_build_constant(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, int64_t value, loom_location_id_t location,
    const char* value_name, loom_value_id_t* out_value) {
  const loom_named_attr_t attribute = {
      .name_id = builder->integer_immediate_name,
      .value = loom_attr_i64(value),
  };
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      ir_builder, builder->descriptor_set,
      &builder->descriptor_set->descriptors[descriptor_ordinal],
      loom_make_named_attr_slice(&attribute, 1), builder->lock_delta_type,
      location, &constant_op));
  *out_value = loom_low_const_result(constant_op);
  if (value_name == NULL) return iree_ok_status();
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      builder->module, iree_make_cstring_view(value_name), &value_name_id));
  return loom_module_set_value_name(builder->module, *out_value, value_name_id);
}

static iree_status_t loom_aie2p_array_resident_build_address(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_type_t result_type, uint32_t address, loom_location_id_t location,
    loom_op_t** out_op) {
  const loom_named_attr_t attribute = {
      .name_id = builder->integer_immediate_name,
      .value = loom_attr_i64(address),
  };
  return loom_low_build_resolved_descriptor_op(
      ir_builder, builder->descriptor_set,
      &builder->descriptor_set->descriptors
           [AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32],
      /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(&attribute, 1), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, out_op);
}

static iree_status_t loom_aie2p_array_resident_build_binary(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_location_id_t location, loom_value_id_t* out_value) {
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      ir_builder, builder->descriptor_set,
      &builder->descriptor_set->descriptors[descriptor_ordinal], operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      &builder->lock_delta_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_unary(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, loom_value_id_t operand,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      ir_builder, builder->descriptor_set,
      &builder->descriptor_set->descriptors[descriptor_ordinal], &operand,
      /*operand_count=*/1, loom_named_attr_slice_empty(), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_lock_op(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, loom_value_id_t delta, uint16_t selector,
    loom_location_id_t location) {
  const loom_named_attr_t attribute = {
      .name_id = builder->lock_selector_name,
      .value = loom_attr_i64(selector),
  };
  loom_op_t* lock_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      ir_builder, builder->descriptor_set,
      &builder->descriptor_set->descriptors[descriptor_ordinal], &delta, 1,
      loom_make_named_attr_slice(&attribute, 1), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &lock_op);
}

static iree_status_t loom_aie2p_array_resident_build_port_lock(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, const loom_aie2p_array_worker_port_plan_t* port,
    loom_aie2p_array_lock_role_t role, uint32_t descriptor_ordinal,
    loom_value_id_t delta, loom_location_id_t location) {
  const loom_aie2p_array_lock_plan_t* lock =
      loom_aie2p_array_resident_find_lock(builder->plan, port, role);
  uint16_t selector = 0;
  IREE_RETURN_IF_ERROR(loom_xdna_array_form_lock_selector(
      builder->plan->family,
      builder->plan->worker_plans[worker_index].coordinate, lock->coordinate,
      lock->lock_id, &selector));
  return loom_aie2p_array_resident_build_lock_op(
      builder, ir_builder, descriptor_ordinal, delta, selector, location);
}

static iree_status_t loom_aie2p_array_resident_build_acquires(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, loom_value_id_t delta, loom_location_id_t location) {
  const loom_aie2p_array_endpoint_direction_t direction_order[] = {
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
  };
  for (iree_host_size_t direction_index = 0;
       direction_index < IREE_ARRAYSIZE(direction_order); ++direction_index) {
    const loom_aie2p_array_endpoint_direction_t direction =
        direction_order[direction_index];
    for (iree_host_size_t i = 0; i < builder->plan->worker_port_count; ++i) {
      const loom_aie2p_array_worker_port_plan_t* port =
          &builder->plan->worker_ports[i];
      if (port->worker_index != worker_index || port->direction != direction) {
        continue;
      }
      const loom_aie2p_array_lock_role_t role =
          direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE
              ? LOOM_AIE2P_ARRAY_LOCK_ROLE_READY
              : LOOM_AIE2P_ARRAY_LOCK_ROLE_CREDIT;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_port_lock(
          builder, ir_builder, worker_index, port, role,
          AIE2P_CORE_DESCRIPTOR_REF_LOCK_ACQUIRE_IMMEDIATE, delta, location));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_releases(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, loom_value_id_t delta, loom_location_id_t location) {
  const loom_aie2p_array_endpoint_direction_t direction_order[] = {
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
  };
  for (iree_host_size_t direction_index = 0;
       direction_index < IREE_ARRAYSIZE(direction_order); ++direction_index) {
    const loom_aie2p_array_endpoint_direction_t direction =
        direction_order[direction_index];
    for (iree_host_size_t i = 0; i < builder->plan->worker_port_count; ++i) {
      const loom_aie2p_array_worker_port_plan_t* port =
          &builder->plan->worker_ports[i];
      if (port->worker_index != worker_index || port->direction != direction) {
        continue;
      }
      const loom_aie2p_array_lock_role_t role =
          direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND
              ? LOOM_AIE2P_ARRAY_LOCK_ROLE_READY
              : LOOM_AIE2P_ARRAY_LOCK_ROLE_CREDIT;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_port_lock(
          builder, ir_builder, worker_index, port, role,
          AIE2P_CORE_DESCRIPTOR_REF_LOCK_RELEASE_IMMEDIATE, delta, location));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_bind_resources(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, loom_block_t* firing_header,
    loom_block_t* source_entry,
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count) {
  iree_host_size_t port_state_index = 0;
  loom_op_t* op = source_entry->first_op;
  while (op != NULL) {
    loom_op_t* next_op = op->next_op;
    if (loom_low_resource_isa(op)) {
      IREE_ASSERT_LT(port_state_index, port_state_count);
      const int64_t resource_index = loom_low_resource_index(op);
      IREE_ASSERT_GE(resource_index, 0);
      const loom_aie2p_array_worker_port_plan_t* port =
          loom_aie2p_array_resident_find_port(builder->plan, worker_index,
                                              (uint64_t)resource_index);
      IREE_ASSERT_LT(port->channel_index, builder->plan->channel_count);
      const loom_aie2p_array_channel_t* channel =
          &builder->plan->channels[port->channel_index];
      IREE_ASSERT_NE(channel->capacity, 0u);
      const loom_value_id_t resource_value = loom_low_resource_result(op);
      loom_aie2p_array_resident_port_state_t* port_state =
          &port_states[port_state_index++];
      *port_state = (loom_aie2p_array_resident_port_state_t){
          .port = port,
          .address_type =
              loom_module_value_type(builder->module, resource_value),
          .current_address = LOOM_VALUE_ID_INVALID,
          .current_slot = LOOM_VALUE_ID_INVALID,
          .next_address = LOOM_VALUE_ID_INVALID,
          .next_slot = LOOM_VALUE_ID_INVALID,
      };
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          ir_builder, firing_header, port_state->address_type,
          &port_state->current_address));
      if (channel->capacity > 2) {
        IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
            ir_builder, firing_header, builder->lock_delta_type,
            &port_state->current_slot));
      }
      IREE_RETURN_IF_ERROR(loom_module_move_value_name(
          builder->module, resource_value, port_state->current_address));
      IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_with(
          builder->module, resource_value, port_state->current_address));
      IREE_RETURN_IF_ERROR(loom_op_erase(builder->module, op));
    }
    op = next_op;
  }
  IREE_ASSERT_EQ(port_state_index, port_state_count);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_initial_state(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_block_t* preheader, loom_block_t* firing_header,
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count, loom_location_id_t location) {
  const iree_host_size_t argument_count =
      loom_aie2p_array_resident_state_argument_count(builder->plan, port_states,
                                                     port_state_count);
  if (argument_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P resident worker ring state exceeds UINT16_MAX values");
  }
  loom_value_id_t* arguments = NULL;
  if (argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, argument_count,
                                  sizeof(*arguments), (void**)&arguments));
  }

  loom_builder_set_block(ir_builder, preheader);
  loom_value_id_t initial_slot = LOOM_VALUE_ID_INVALID;
  iree_host_size_t argument_index = 0;
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    loom_aie2p_array_resident_port_state_t* port_state = &port_states[i];
    loom_op_t* address_op = NULL;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_address(
        builder, ir_builder, port_state->address_type,
        loom_aie2p_array_resident_port_slot_address(builder->plan,
                                                    port_state->port, 0),
        location, &address_op));
    arguments[argument_index++] =
        loom_value_slice_get(loom_low_op_results(address_op), 0);
    const loom_aie2p_array_channel_t* channel =
        &builder->plan->channels[port_state->port->channel_index];
    if (channel->capacity > 2) {
      if (initial_slot == LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
            builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT,
            0, location, "ring_slot_zero", &initial_slot));
      }
      arguments[argument_index++] = initial_slot;
    }
  }
  IREE_ASSERT_EQ(argument_index, argument_count);
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(ir_builder, firing_header, arguments, argument_count,
                           location, &branch_op);
}

static iree_status_t loom_aie2p_array_resident_build_port_advance(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_aie2p_array_resident_port_state_t* port_state,
    loom_aie2p_array_resident_address_mask_t* address_masks,
    iree_host_size_t* address_mask_count, loom_location_id_t location) {
  const loom_aie2p_array_channel_t* channel =
      &builder->plan->channels[port_state->port->channel_index];
  if (channel->capacity == 1) {
    port_state->next_address = port_state->current_address;
    return iree_ok_status();
  }
  if (channel->capacity == 2) {
    loom_value_id_t address_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
        builder, ir_builder,
        AIE2P_CORE_DESCRIPTOR_REF_MOVE_LOCAL_ADDRESS_TO_SCALAR,
        port_state->current_address, builder->lock_delta_type, location,
        &address_bits));
    const uint32_t address_mask = loom_aie2p_array_resident_port_slot_address(
                                      builder->plan, port_state->port, 0) ^
                                  loom_aie2p_array_resident_port_slot_address(
                                      builder->plan, port_state->port, 1);
    loom_value_id_t address_mask_value = LOOM_VALUE_ID_INVALID;
    for (iree_host_size_t i = 0; i < *address_mask_count; ++i) {
      if (address_masks[i].mask == address_mask) {
        address_mask_value = address_masks[i].value;
        break;
      }
    }
    if (address_mask_value == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
          builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32,
          address_mask, location, /*value_name=*/NULL, &address_mask_value));
      address_masks[(*address_mask_count)++] =
          (loom_aie2p_array_resident_address_mask_t){
              .mask = address_mask,
              .value = address_mask_value,
          };
    }
    // For the two planned addresses A and B, `A ^ (A ^ B) == B` and
    // `B ^ (A ^ B) == A`. This advances an arbitrary ping-pong allocation
    // without carrying a slot counter or cloning the worker body.
    loom_value_id_t next_address_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_XOR_I32, address_bits,
        address_mask_value, location, &next_address_bits));
    return loom_aie2p_array_resident_build_unary(
        builder, ir_builder,
        AIE2P_CORE_DESCRIPTOR_REF_MOVE_SCALAR_TO_LOCAL_ADDRESS,
        next_address_bits, port_state->address_type, location,
        &port_state->next_address);
  }

  const uint32_t comparison_count = channel->capacity - 1;
  loom_block_t** comparison_blocks = NULL;
  loom_block_t** address_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, comparison_count, sizeof(*comparison_blocks),
      (void**)&comparison_blocks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, channel->capacity, sizeof(*address_blocks),
      (void**)&address_blocks));
  loom_region_t* region = ir_builder->ip.block->parent_region;
  comparison_blocks[0] = ir_builder->ip.block;
  for (uint32_t i = 1; i < comparison_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_region_append_block(builder->module, region,
                                                  &comparison_blocks[i]));
  }
  for (uint32_t i = 0; i < channel->capacity; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(builder->module, region, &address_blocks[i]));
  }
  loom_block_t* merge_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->module, region, &merge_block));
  loom_value_id_t next_address_arg = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      ir_builder, merge_block, port_state->address_type, &next_address_arg));
  loom_value_id_t next_slot_arg = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      ir_builder, merge_block, builder->lock_delta_type, &next_slot_arg));

  for (uint32_t slot = 0; slot < comparison_count; ++slot) {
    loom_builder_set_block(ir_builder, comparison_blocks[slot]);
    loom_value_id_t expected_slot = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, slot,
        location, /*value_name=*/NULL, &expected_slot));
    loom_value_id_t matches_slot = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32,
        port_state->current_slot, expected_slot, location, &matches_slot));
    loom_block_t* false_destination = slot + 1 < comparison_count
                                          ? comparison_blocks[slot + 1]
                                          : address_blocks[0];
    loom_op_t* branch_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
        ir_builder, matches_slot, address_blocks[slot + 1], false_destination,
        location, &branch_op));
  }

  for (uint32_t slot = 0; slot < channel->capacity; ++slot) {
    loom_builder_set_block(ir_builder, address_blocks[slot]);
    loom_op_t* address_op = NULL;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_address(
        builder, ir_builder, port_state->address_type,
        loom_aie2p_array_resident_port_slot_address(builder->plan,
                                                    port_state->port, slot),
        location, &address_op));
    loom_value_id_t next_slot = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, slot,
        location, /*value_name=*/NULL, &next_slot));
    const loom_value_id_t arguments[] = {
        loom_value_slice_get(loom_low_op_results(address_op), 0),
        next_slot,
    };
    loom_op_t* branch_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, merge_block, arguments,
                                           IREE_ARRAYSIZE(arguments), location,
                                           &branch_op));
  }

  loom_builder_set_block(ir_builder, merge_block);
  port_state->next_address = next_address_arg;
  port_state->next_slot = next_slot_arg;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_ring_advance(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_block_t* firing_header,
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count, loom_location_id_t location) {
  loom_aie2p_array_resident_address_mask_t* address_masks = NULL;
  if (port_state_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, port_state_count, sizeof(*address_masks),
        (void**)&address_masks));
  }
  iree_host_size_t address_mask_count = 0;
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_port_advance(
        builder, ir_builder, &port_states[i], address_masks,
        &address_mask_count, location));
  }

  const iree_host_size_t argument_count =
      loom_aie2p_array_resident_state_argument_count(builder->plan, port_states,
                                                     port_state_count);
  loom_value_id_t* arguments = NULL;
  if (argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, argument_count,
                                  sizeof(*arguments), (void**)&arguments));
  }
  iree_host_size_t argument_index = 0;
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    const loom_aie2p_array_resident_port_state_t* port_state = &port_states[i];
    arguments[argument_index++] = port_state->next_address;
    const loom_aie2p_array_channel_t* channel =
        &builder->plan->channels[port_state->port->channel_index];
    if (channel->capacity > 2) {
      arguments[argument_index++] = port_state->next_slot;
    }
  }
  IREE_ASSERT_EQ(argument_index, argument_count);
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(ir_builder, firing_header, arguments, argument_count,
                           location, &branch_op);
}

static iree_status_t loom_aie2p_array_resident_rewrite_returns(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_region_t* resident_body, uint16_t source_block_start,
    uint16_t source_block_count, loom_block_t* latch) {
  for (uint16_t block_ordinal = 0; block_ordinal < source_block_count;
       ++block_ordinal) {
    loom_block_t* block = loom_region_block(
        resident_body, (uint16_t)(source_block_start + block_ordinal));
    loom_op_t* op = block->first_op;
    while (op != NULL) {
      loom_op_t* next_op = op->next_op;
      if (loom_low_return_isa(op)) {
        loom_builder_set_before(ir_builder, op);
        loom_op_t* branch_op = NULL;
        IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, latch, /*args=*/NULL,
                                               /*args_count=*/0, op->location,
                                               &branch_op));
        IREE_RETURN_IF_ERROR(loom_op_erase(builder->module, op));
      }
      op = next_op;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_materialize_worker(
    loom_aie2p_array_resident_builder_t* builder, uint32_t worker_index,
    loom_aie2p_array_resident_worker_t* out_worker) {
  const loom_aie2p_array_worker_t* worker =
      &builder->plan->workers[worker_index];
  IREE_ASSERT_EQ(worker->entry.module_id, 0u);
  IREE_ASSERT_LT(worker->entry.symbol_id, builder->module->symbols.count);
  loom_op_t* source_function =
      builder->module->symbols.entries[worker->entry.symbol_id].defining_op;
  IREE_ASSERT(source_function != NULL &&
              loom_low_func_def_isa(source_function));
  const loom_region_t* source_body = loom_low_func_def_body(source_function);
  IREE_ASSERT(source_body != NULL && source_body->block_count != 0);

  loom_symbol_ref_t resident_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_add_symbol(
      builder, worker_index, &resident_ref));
  loom_builder_t ir_builder;
  loom_builder_initialize(builder->module, &builder->module->arena,
                          loom_module_block(builder->module), &ir_builder);
  loom_low_func_def_build_flags_t build_flags =
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN;
  const loom_symbol_ref_t target = loom_low_func_def_target(source_function);
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET;
  }
  loom_op_t* resident_function = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &ir_builder, build_flags,
      /*visibility=*/0, LOOM_LOW_RETAIN_RETAIN,
      /*cc=*/0, /*purity=*/0, /*inline_policy=*/0, /*allocation=*/0,
      /*schedule=*/0, loom_low_func_def_descriptor_set(source_function), target,
      /*abi=*/0, loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), resident_ref,
      /*arg_types=*/NULL, /*arg_types_count=*/0,
      /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, source_function->location,
      &resident_function));
  loom_region_t* resident_body = loom_low_func_def_body(resident_function);
  resident_body->flags = source_body->flags;
  resident_body->source_flags = source_body->source_flags;
  loom_block_t* preheader = loom_region_entry_block(resident_body);
  loom_builder_enter_region(&ir_builder, resident_function, resident_body);

  loom_value_id_t acquire_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, &ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, -1,
      source_function->location, "acquire_delta", &acquire_delta));
  loom_value_id_t release_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, &ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, 1,
      source_function->location, "release_delta", &release_delta));

  loom_block_t* firing_header = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->module, resident_body, &firing_header));
  const uint16_t source_block_start = resident_body->block_count;
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(builder->module,
                                                builder->module, builder->arena,
                                                /*options=*/NULL, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_clone_region_blocks(
      &ir_builder, source_body, resident_body, source_block_start, &remap));
  loom_block_t* source_entry =
      loom_region_block(resident_body, source_block_start);

  const iree_host_size_t port_state_count =
      loom_aie2p_array_resident_worker_port_count(builder->plan, worker_index);
  loom_aie2p_array_resident_port_state_t* port_states = NULL;
  if (port_state_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, port_state_count,
                                  sizeof(*port_states), (void**)&port_states));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_bind_resources(
      builder, &ir_builder, worker_index, firing_header, source_entry,
      port_states, port_state_count));

  loom_builder_set_block(&ir_builder, firing_header);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_acquires(
      builder, &ir_builder, worker_index, acquire_delta,
      source_function->location));
  loom_op_t* firing_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(
      &ir_builder, source_entry, /*args=*/NULL, /*args_count=*/0,
      source_function->location, &firing_branch));

  loom_block_t* latch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->module, resident_body, &latch));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_rewrite_returns(
      builder, &ir_builder, resident_body, source_block_start,
      source_body->block_count, latch));
  loom_builder_set_block(&ir_builder, latch);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_releases(
      builder, &ir_builder, worker_index, release_delta,
      source_function->location));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_ring_advance(
      builder, &ir_builder, firing_header, port_states, port_state_count,
      source_function->location));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_initial_state(
      builder, &ir_builder, preheader, firing_header, port_states,
      port_state_count, source_function->location));
  *out_worker = (loom_aie2p_array_resident_worker_t){
      .worker_index = worker_index,
      .entry = resident_ref,
      .function_op = resident_function,
  };
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_materialize_resident_program(
    loom_module_t* module, const loom_aie2p_array_plan_t* plan,
    iree_arena_allocator_t* arena,
    loom_aie2p_array_resident_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_aie2p_array_resident_program_t){0};

  loom_aie2p_array_resident_worker_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->worker_count, sizeof(*workers), (void**)&workers));
  loom_aie2p_array_resident_builder_t builder = {
      .module = module,
      .plan = plan,
      .arena = arena,
      .descriptor_set = loom_aie2p_core_descriptor_set(),
  };
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("i"), &builder.integer_immediate_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("id"),
                                                 &builder.lock_selector_name));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      builder.descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1,
      &builder.lock_delta_type));
  for (iree_host_size_t i = 0; i < plan->worker_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_materialize_worker(
        &builder, (uint32_t)i, &workers[i]));
  }
  *out_program = (loom_aie2p_array_resident_program_t){
      .workers = workers,
      .worker_count = plan->worker_count,
  };
  return iree_ok_status();
}
