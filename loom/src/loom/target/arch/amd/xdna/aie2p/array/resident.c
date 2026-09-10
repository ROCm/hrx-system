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

typedef uint8_t loom_aie2p_array_port_direction_flags_t;
enum loom_aie2p_array_port_direction_flag_bits_e {
  LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_RECEIVE = 1u << 0,
  LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_SEND = 1u << 1,
  LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_ALL =
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_RECEIVE |
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_SEND,
};

typedef enum loom_aie2p_array_state_value_kind_e {
  LOOM_AIE2P_ARRAY_STATE_VALUE_CURRENT = 0,
  LOOM_AIE2P_ARRAY_STATE_VALUE_NEXT = 1,
} loom_aie2p_array_state_value_kind_t;

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
  // Interned `imm` indexed-memory immediate field name.
  loom_string_id_t memory_immediate_name;
  // Interned `idx` vector-lane immediate field name.
  loom_string_id_t vector_index_name;
  // Interned `id` lock-selector field name.
  loom_string_id_t lock_selector_name;
  // Scalar register type carrying signed lock deltas.
  loom_type_t lock_delta_type;
  // 512-bit vector register range used to move scalar F32 values.
  loom_type_t vector512_type;
  // One 512-bit accumulator register.
  loom_type_t accumulator512_type;
  // Four-register accumulator range used by configured F32 addition.
  loom_type_t accumulator2048_type;
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

static bool loom_aie2p_array_resident_direction_selected(
    loom_aie2p_array_endpoint_direction_t direction,
    loom_aie2p_array_port_direction_flags_t flags) {
  const loom_aie2p_array_port_direction_flags_t direction_flag =
      direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE
          ? LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_RECEIVE
          : LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_SEND;
  return iree_any_bit_set(flags, direction_flag);
}

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

static iree_status_t loom_aie2p_array_resident_build_op(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      ir_builder, builder->descriptor_set,
      &builder->descriptor_set->descriptors[descriptor_ordinal], operands,
      operand_count, attrs, result_type, result_type != NULL ? 1 : 0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &op));
  if (out_value != NULL) {
    IREE_ASSERT(result_type != NULL);
    *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_address(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_type_t result_type, uint32_t address, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_named_attr_t attribute = {
      .name_id = builder->integer_immediate_name,
      .value = loom_attr_i64(address),
  };
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32,
      /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(&attribute, 1), &result_type, location,
      out_value);
}

static iree_status_t loom_aie2p_array_resident_build_binary(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_location_id_t location, loom_value_id_t* out_value) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder, descriptor_ordinal, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      &builder->lock_delta_type, location, out_value);
}

static iree_status_t loom_aie2p_array_resident_build_unary(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, loom_value_id_t operand,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder, descriptor_ordinal, &operand,
      /*operand_count=*/1, loom_named_attr_slice_empty(), &result_type,
      location, out_value);
}

static iree_status_t loom_aie2p_array_resident_build_load_i32(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t address, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_named_attr_t offset = {
      .name_id = builder->memory_immediate_name,
      .value = loom_attr_i64(0),
  };
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_I32_INDEXED_IMMEDIATE, &address, 1,
      loom_make_named_attr_slice(&offset, 1), &builder->lock_delta_type,
      location, out_value);
}

static iree_status_t loom_aie2p_array_resident_build_store_i32(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t value, loom_value_id_t address,
    loom_location_id_t location) {
  const loom_named_attr_t offset = {
      .name_id = builder->memory_immediate_name,
      .value = loom_attr_i64(0),
  };
  const loom_value_id_t operands[] = {value, address};
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_STORE_SCALAR_I32_INDEXED_IMMEDIATE, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&offset, 1),
      /*result_type=*/NULL, location, /*out_value=*/NULL);
}

static iree_status_t loom_aie2p_array_resident_build_f32_accumulator(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t scalar_value, loom_value_id_t zero_accumulator_lane,
    loom_location_id_t location, loom_value_id_t* out_accumulator) {
  loom_value_id_t vector_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I32X16,
      &scalar_value, 1, loom_named_attr_slice_empty(), &builder->vector512_type,
      location, &vector_value));
  loom_value_id_t accumulator_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_MOVE_VECTOR512_TO_ACCUMULATOR512, &vector_value,
      1, loom_named_attr_slice_empty(), &builder->accumulator512_type, location,
      &accumulator_lane));
  const loom_value_id_t lanes[] = {
      accumulator_lane,
      zero_accumulator_lane,
      zero_accumulator_lane,
      zero_accumulator_lane,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      ir_builder, lanes, IREE_ARRAYSIZE(lanes), builder->accumulator2048_type,
      location, &concat_op));
  *out_accumulator = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_add_f32_accumulators(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t mode,
    loom_location_id_t location, loom_value_id_t* out_value) {
  const loom_value_id_t operands[] = {lhs, rhs, mode};
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_F32X64_CONFIGURED,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      &builder->accumulator2048_type, location, out_value);
}

static iree_status_t loom_aie2p_array_resident_build_extract_f32(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t accumulator, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_slice_build(ir_builder, accumulator, /*offset=*/0,
                           builder->accumulator512_type, location, &slice_op));
  loom_value_id_t sum_vector = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t sum_lane = loom_low_slice_result(slice_op);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_MOVE_ACCUMULATOR512_TO_VECTOR512, &sum_lane, 1,
      loom_named_attr_slice_empty(), &builder->vector512_type, location,
      &sum_vector));
  const loom_named_attr_t lane = {
      .name_id = builder->vector_index_name,
      .value = loom_attr_i64(0),
  };
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_EXTRACT_I32_IMMEDIATE,
      &sum_vector, 1, loom_make_named_attr_slice(&lane, 1),
      &builder->lock_delta_type, location, out_value);
}

static iree_status_t loom_aie2p_array_resident_build_lock_op(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t descriptor_ordinal, loom_value_id_t delta, uint16_t selector,
    loom_location_id_t location) {
  const loom_named_attr_t attribute = {
      .name_id = builder->lock_selector_name,
      .value = loom_attr_i64(selector),
  };
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder, descriptor_ordinal, &delta, 1,
      loom_make_named_attr_slice(&attribute, 1), /*result_type=*/NULL, location,
      /*out_value=*/NULL);
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
    uint32_t worker_index, loom_aie2p_array_port_direction_flags_t flags,
    loom_value_id_t delta, loom_location_id_t location) {
  const loom_aie2p_array_endpoint_direction_t direction_order[] = {
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
  };
  for (iree_host_size_t direction_index = 0;
       direction_index < IREE_ARRAYSIZE(direction_order); ++direction_index) {
    const loom_aie2p_array_endpoint_direction_t direction =
        direction_order[direction_index];
    if (!loom_aie2p_array_resident_direction_selected(direction, flags)) {
      continue;
    }
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
    uint32_t worker_index, loom_aie2p_array_port_direction_flags_t flags,
    loom_value_id_t delta, loom_location_id_t location) {
  const loom_aie2p_array_endpoint_direction_t direction_order[] = {
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND,
      LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE,
  };
  for (iree_host_size_t direction_index = 0;
       direction_index < IREE_ARRAYSIZE(direction_order); ++direction_index) {
    const loom_aie2p_array_endpoint_direction_t direction =
        direction_order[direction_index];
    if (!loom_aie2p_array_resident_direction_selected(direction, flags)) {
      continue;
    }
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

static iree_status_t loom_aie2p_array_resident_define_state_arguments(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_block_t* block,
    const loom_aie2p_array_resident_port_state_t* source_states,
    iree_host_size_t port_state_count,
    loom_aie2p_array_resident_port_state_t* out_states) {
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    const loom_aie2p_array_channel_t* channel =
        &builder->plan->channels[source_states[i].port->channel_index];
    out_states[i] = source_states[i];
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        ir_builder, block, source_states[i].address_type,
        &out_states[i].current_address));
    out_states[i].current_slot = LOOM_VALUE_ID_INVALID;
    if (channel->capacity > 2) {
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          ir_builder, block, builder->lock_delta_type,
          &out_states[i].current_slot));
    }
    out_states[i].next_address = LOOM_VALUE_ID_INVALID;
    out_states[i].next_slot = LOOM_VALUE_ID_INVALID;
  }
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
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_address(
        builder, ir_builder, port_state->address_type,
        loom_aie2p_array_resident_port_slot_address(builder->plan,
                                                    port_state->port, 0),
        location, &address));
    arguments[argument_index++] = address;
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
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_address(
        builder, ir_builder, port_state->address_type,
        loom_aie2p_array_resident_port_slot_address(builder->plan,
                                                    port_state->port, slot),
        location, &address));
    loom_value_id_t next_slot = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, slot,
        location, /*value_name=*/NULL, &next_slot));
    const loom_value_id_t arguments[] = {address, next_slot};
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

static void loom_aie2p_array_resident_initialize_next_state(
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count) {
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    port_states[i].next_address = port_states[i].current_address;
    port_states[i].next_slot = port_states[i].current_slot;
  }
}

static iree_status_t loom_aie2p_array_resident_build_ring_advances(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count,
    loom_aie2p_array_port_direction_flags_t flags,
    loom_location_id_t location) {
  loom_aie2p_array_resident_address_mask_t* address_masks = NULL;
  if (port_state_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, port_state_count, sizeof(*address_masks),
        (void**)&address_masks));
  }
  iree_host_size_t address_mask_count = 0;
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    if (!loom_aie2p_array_resident_direction_selected(
            port_states[i].port->direction, flags)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_port_advance(
        builder, ir_builder, &port_states[i], address_masks,
        &address_mask_count, location));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_state_branch(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_block_t* destination,
    const loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count,
    loom_aie2p_array_state_value_kind_t value_kind,
    const loom_value_id_t* trailing_arguments,
    iree_host_size_t trailing_argument_count, loom_location_id_t location) {
  iree_host_size_t argument_count =
      loom_aie2p_array_resident_state_argument_count(builder->plan, port_states,
                                                     port_state_count);
  if (!iree_host_size_checked_add(argument_count, trailing_argument_count,
                                  &argument_count) ||
      argument_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P resident worker loop state exceeds UINT16_MAX values");
  }
  loom_value_id_t* arguments = NULL;
  if (argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, argument_count,
                                  sizeof(*arguments), (void**)&arguments));
  }
  iree_host_size_t argument_index = 0;
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    const loom_aie2p_array_resident_port_state_t* port_state = &port_states[i];
    arguments[argument_index++] =
        value_kind == LOOM_AIE2P_ARRAY_STATE_VALUE_NEXT
            ? port_state->next_address
            : port_state->current_address;
    const loom_aie2p_array_channel_t* channel =
        &builder->plan->channels[port_state->port->channel_index];
    if (channel->capacity > 2) {
      arguments[argument_index++] =
          value_kind == LOOM_AIE2P_ARRAY_STATE_VALUE_NEXT
              ? port_state->next_slot
              : port_state->current_slot;
    }
  }
  for (iree_host_size_t i = 0; i < trailing_argument_count; ++i) {
    arguments[argument_index++] = trailing_arguments[i];
  }
  IREE_ASSERT_EQ(argument_index, argument_count);
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(ir_builder, destination, arguments, argument_count,
                           location, &branch_op);
}

static iree_status_t loom_aie2p_array_resident_build_ring_advance(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_block_t* firing_header,
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count, loom_location_id_t location) {
  loom_aie2p_array_resident_initialize_next_state(port_states,
                                                  port_state_count);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_ring_advances(
      builder, ir_builder, port_states, port_state_count,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_ALL, location));
  return loom_aie2p_array_resident_build_state_branch(
      builder, ir_builder, firing_header, port_states, port_state_count,
      LOOM_AIE2P_ARRAY_STATE_VALUE_NEXT, /*trailing_arguments=*/NULL,
      /*trailing_argument_count=*/0, location);
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

static loom_aie2p_array_resident_port_state_t*
loom_aie2p_array_resident_find_fold_output(
    const loom_aie2p_array_worker_t* worker,
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count) {
  loom_aie2p_array_resident_port_state_t* result = NULL;
  for (iree_host_size_t i = 0; i < port_state_count; ++i) {
    const loom_aie2p_array_worker_port_plan_t* port = port_states[i].port;
    if (port->direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND &&
        port->port == worker->fold_output_port) {
      IREE_ASSERT(result == NULL &&
                  "validated folded worker must have one output port");
      result = &port_states[i];
    }
  }
  IREE_ASSERT(result != NULL &&
              "validated folded worker must have one output port");
  return result;
}

static iree_status_t loom_aie2p_array_resident_build_zero_accumulator(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_location_id_t location, loom_value_id_t* out_accumulator,
    loom_value_id_t* out_lane) {
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_ACCUMULATOR_CLEAR_F32X64,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &builder->accumulator2048_type, location, out_accumulator));
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_slice_build(ir_builder, *out_accumulator, /*offset=*/0,
                           builder->accumulator512_type, location, &slice_op));
  *out_lane = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_materialize_recordwise_body(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, loom_region_t* resident_body,
    const loom_region_t* source_body, uint16_t source_block_start,
    loom_block_t* firing_header, loom_block_t* source_entry,
    loom_block_t* preheader,
    loom_aie2p_array_resident_port_state_t* port_states,
    iree_host_size_t port_state_count, loom_value_id_t acquire_delta,
    loom_value_id_t release_delta, loom_location_id_t location) {
  loom_builder_set_block(ir_builder, firing_header);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_acquires(
      builder, ir_builder, worker_index,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_ALL, acquire_delta, location));
  loom_op_t* firing_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, source_entry,
                                         /*args=*/NULL, /*args_count=*/0,
                                         location, &firing_branch));

  loom_block_t* latch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->module, resident_body, &latch));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_rewrite_returns(
      builder, ir_builder, resident_body, source_block_start,
      source_body->block_count, latch));
  loom_builder_set_block(ir_builder, latch);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_releases(
      builder, ir_builder, worker_index,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_ALL, release_delta, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_ring_advance(
      builder, ir_builder, firing_header, port_states, port_state_count,
      location));
  return loom_aie2p_array_resident_build_initial_state(
      builder, ir_builder, preheader, firing_header, port_states,
      port_state_count, location);
}

static iree_status_t loom_aie2p_array_resident_materialize_folded_body(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, const loom_aie2p_array_worker_t* worker,
    loom_region_t* resident_body, const loom_region_t* source_body,
    uint16_t source_block_start, loom_block_t* activation_header,
    loom_block_t* record_header, loom_block_t* source_entry,
    loom_block_t* preheader,
    loom_aie2p_array_resident_port_state_t* record_states,
    iree_host_size_t port_state_count, loom_value_id_t acquire_delta,
    loom_value_id_t release_delta, loom_location_id_t location) {
  loom_aie2p_array_resident_port_state_t* activation_states = NULL;
  if (port_state_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, port_state_count, sizeof(*activation_states),
        (void**)&activation_states));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_define_state_arguments(
      builder, ir_builder, activation_header, record_states, port_state_count,
      activation_states));

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  loom_value_id_t remaining_records = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      ir_builder, record_header, builder->accumulator2048_type, &accumulator));
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      ir_builder, record_header, builder->lock_delta_type, &remaining_records));
  loom_value_id_t is_first_record = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      ir_builder, record_header, builder->lock_delta_type, &is_first_record));

  loom_builder_set_block(ir_builder, preheader);
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, 0,
      location, "fold_zero", &zero));
  loom_value_id_t record_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32,
      worker->fold_record_count, location, "fold_record_count", &record_count));
  loom_value_id_t mode = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_MOVA, 60,
      location, "fold_mode", &mode));
  loom_value_id_t zero_accumulator = LOOM_VALUE_ID_INVALID;
  loom_value_id_t zero_accumulator_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_zero_accumulator(
      builder, ir_builder, location, &zero_accumulator,
      &zero_accumulator_lane));

  loom_builder_set_block(ir_builder, activation_header);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_acquires(
      builder, ir_builder, worker_index,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_SEND, acquire_delta, location));
  const loom_value_id_t initial_record_state[] = {
      zero_accumulator,
      record_count,
      release_delta,
  };
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_state_branch(
      builder, ir_builder, record_header, activation_states, port_state_count,
      LOOM_AIE2P_ARRAY_STATE_VALUE_CURRENT, initial_record_state,
      IREE_ARRAYSIZE(initial_record_state), location));

  loom_builder_set_block(ir_builder, record_header);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_acquires(
      builder, ir_builder, worker_index,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_RECEIVE, acquire_delta, location));
  loom_op_t* record_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, source_entry,
                                         /*args=*/NULL, /*args_count=*/0,
                                         location, &record_branch));

  loom_block_t* record_latch = NULL;
  loom_block_t* initialize_block = NULL;
  loom_block_t* add_block = NULL;
  loom_block_t* accumulation_merge_block = NULL;
  loom_block_t* continue_block = NULL;
  loom_block_t* completion_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->module, resident_body, &record_latch));
  IREE_RETURN_IF_ERROR(loom_region_append_block(builder->module, resident_body,
                                                &initialize_block));
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(builder->module, resident_body, &add_block));
  IREE_RETURN_IF_ERROR(loom_region_append_block(builder->module, resident_body,
                                                &accumulation_merge_block));
  IREE_RETURN_IF_ERROR(loom_region_append_block(builder->module, resident_body,
                                                &continue_block));
  IREE_RETURN_IF_ERROR(loom_region_append_block(builder->module, resident_body,
                                                &completion_block));
  loom_value_id_t sum = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_define_block_arg(ir_builder, accumulation_merge_block,
                                    builder->accumulator2048_type, &sum));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_rewrite_returns(
      builder, ir_builder, resident_body, source_block_start,
      source_body->block_count, record_latch));

  loom_builder_set_block(ir_builder, record_latch);
  loom_aie2p_array_resident_port_state_t* output_state =
      loom_aie2p_array_resident_find_fold_output(worker, record_states,
                                                 port_state_count);
  loom_value_id_t contribution = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_load_i32(
      builder, ir_builder, output_state->current_address, location,
      &contribution));
  loom_value_id_t contribution_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_f32_accumulator(
      builder, ir_builder, contribution, zero_accumulator_lane, location,
      &contribution_accumulator));
  loom_value_id_t initialize_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_NEZ_I32,
      is_first_record, builder->lock_delta_type, location,
      &initialize_accumulator));
  loom_op_t* initialize_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
      ir_builder, initialize_accumulator, initialize_block, add_block, location,
      &initialize_branch));

  loom_builder_set_block(ir_builder, initialize_block);
  loom_op_t* initialized_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(
      ir_builder, accumulation_merge_block, &contribution_accumulator,
      /*args_count=*/1, location, &initialized_branch));

  loom_builder_set_block(ir_builder, add_block);
  loom_value_id_t accumulated_sum = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_add_f32_accumulators(
      builder, ir_builder, accumulator, contribution_accumulator, mode,
      location, &accumulated_sum));
  loom_op_t* accumulated_branch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(ir_builder, accumulation_merge_block, &accumulated_sum,
                        /*args_count=*/1, location, &accumulated_branch));

  loom_builder_set_block(ir_builder, accumulation_merge_block);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_releases(
      builder, ir_builder, worker_index,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_RECEIVE, release_delta, location));
  loom_aie2p_array_resident_initialize_next_state(record_states,
                                                  port_state_count);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_ring_advances(
      builder, ir_builder, record_states, port_state_count,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_RECEIVE, location));
  loom_value_id_t next_remaining_records = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_SUB_I32, remaining_records,
      release_delta, location, &next_remaining_records));
  loom_value_id_t has_more_records = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_NEZ_I32,
      next_remaining_records, builder->lock_delta_type, location,
      &has_more_records));
  loom_op_t* completion_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(ir_builder, has_more_records,
                                              continue_block, completion_block,
                                              location, &completion_branch));

  loom_builder_set_block(ir_builder, continue_block);
  const loom_value_id_t next_record_state[] = {sum, next_remaining_records,
                                               zero};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_state_branch(
      builder, ir_builder, record_header, record_states, port_state_count,
      LOOM_AIE2P_ARRAY_STATE_VALUE_NEXT, next_record_state,
      IREE_ARRAYSIZE(next_record_state), location));

  loom_builder_set_block(ir_builder, completion_block);
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_extract_f32(
      builder, ir_builder, sum, location, &result));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_store_i32(
      builder, ir_builder, result, output_state->current_address, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_releases(
      builder, ir_builder, worker_index,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_SEND, release_delta, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_ring_advances(
      builder, ir_builder, record_states, port_state_count,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_SEND, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_state_branch(
      builder, ir_builder, activation_header, record_states, port_state_count,
      LOOM_AIE2P_ARRAY_STATE_VALUE_NEXT, /*trailing_arguments=*/NULL,
      /*trailing_argument_count=*/0, location));

  return loom_aie2p_array_resident_build_initial_state(
      builder, ir_builder, preheader, activation_header, activation_states,
      port_state_count, location);
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

  loom_block_t* activation_header = NULL;
  IREE_RETURN_IF_ERROR(loom_region_append_block(builder->module, resident_body,
                                                &activation_header));
  loom_block_t* record_header = activation_header;
  if (worker->fold_record_count != 0) {
    IREE_RETURN_IF_ERROR(loom_region_append_block(
        builder->module, resident_body, &record_header));
  }
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
      builder, &ir_builder, worker_index, record_header, source_entry,
      port_states, port_state_count));
  if (worker->fold_record_count == 0) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_materialize_recordwise_body(
        builder, &ir_builder, worker_index, resident_body, source_body,
        source_block_start, record_header, source_entry, preheader, port_states,
        port_state_count, acquire_delta, release_delta,
        source_function->location));
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_materialize_folded_body(
        builder, &ir_builder, worker_index, worker, resident_body, source_body,
        source_block_start, activation_header, record_header, source_entry,
        preheader, port_states, port_state_count, acquire_delta, release_delta,
        source_function->location));
  }
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
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("imm"), &builder.memory_immediate_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("idx"),
                                                 &builder.vector_index_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("id"),
                                                 &builder.lock_selector_name));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      builder.descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1,
      &builder.lock_delta_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      builder.descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 2,
      &builder.vector512_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      builder.descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 1,
      &builder.accumulator512_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      builder.descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 4,
      &builder.accumulator2048_type));
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
