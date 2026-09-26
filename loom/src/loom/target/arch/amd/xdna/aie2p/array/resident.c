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
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/transforms/cfg/block_fusion.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"

enum { LOOM_AIE2P_ARRAY_SCALAR_FOLD_PACK_WIDTH = 16 };

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
  // Immutable source module owning the selected array and leaf functions.
  const loom_module_t* source_module;
  // Module receiving all resident worker functions.
  loom_module_t* module;
  // Trusted physical plan consumed by materialization.
  const loom_aie2p_array_plan_t* plan;
  // Scratch arena for names, remaps, and ring state.
  iree_arena_allocator_t* arena;
  // AIE2P core descriptor set used by generated packets.
  const loom_low_descriptor_set_t* descriptor_set;
  // Interned AIE2P core representation contract.
  loom_string_id_t representation_contract;
  // Interned `i` immediate field name.
  loom_string_id_t integer_immediate_name;
  // Interned `imm` indexed-memory immediate field name.
  loom_string_id_t memory_immediate_name;
  // Interned `idx` vector-lane immediate field name.
  loom_string_id_t vector_index_name;
  // Interned `id` lock-selector field name.
  loom_string_id_t lock_selector_name;
  // AIE2P endpoint register type carrying resident ring addresses.
  loom_type_t endpoint_address_type;
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

static uint32_t loom_aie2p_array_resident_port_slot_address(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_worker_port_plan_t* port, uint32_t slot) {
  IREE_ASSERT_LT(port->channel_index, plan->channel_count);
  const loom_aie2p_array_channel_t* channel =
      &plan->channels[port->channel_index];
  IREE_ASSERT_LT(slot, channel->capacity);
  const iree_host_size_t slot_index =
      (iree_host_size_t)channel->first_channel_slot + slot;
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

static iree_status_t loom_aie2p_array_resident_add_symbol(
    loom_aie2p_array_resident_builder_t* builder, uint32_t worker_index,
    loom_symbol_ref_t* out_ref) {
  const loom_func_like_t array_function = loom_func_like_const_cast(
      builder->source_module, builder->plan->function_op);
  const loom_symbol_ref_t array_ref = loom_func_like_callee(array_function);
  IREE_ASSERT_EQ(array_ref.module_id, 0u);
  IREE_ASSERT_LT(array_ref.symbol_id, builder->source_module->symbols.count);
  const loom_string_id_t array_name_id =
      builder->source_module->symbols.entries[array_ref.symbol_id].name_id;
  IREE_ASSERT_LT(array_name_id, builder->source_module->strings.count);
  const iree_string_view_t array_name =
      loom_string_table_get(&builder->source_module->strings, array_name_id);
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
  if (value_name == NULL) {
    return iree_ok_status();
  }
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
      &builder->descriptor_set->descriptors[descriptor_ordinal],
      /*access_flags=*/0, operands, operand_count, attrs, result_type,
      result_type != NULL ? 1 : 0,
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

static iree_status_t loom_aie2p_array_resident_build_load_accumulator_f32x16(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t address, uint32_t byte_offset, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_named_attr_t offset = {
      .name_id = builder->memory_immediate_name,
      .value = loom_attr_i64(byte_offset),
  };
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_LOAD_ACCUMULATOR_INDEXED_IMMEDIATE, &address, 1,
      loom_make_named_attr_slice(&offset, 1), &builder->accumulator512_type,
      location, out_value);
}

static iree_status_t loom_aie2p_array_resident_build_store_accumulator_f32x16(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t value, loom_value_id_t address, uint32_t byte_offset,
    loom_location_id_t location) {
  const loom_named_attr_t offset = {
      .name_id = builder->memory_immediate_name,
      .value = loom_attr_i64(byte_offset),
  };
  const loom_value_id_t operands[] = {value, address};
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_STORE_ACCUMULATOR_INDEXED_IMMEDIATE, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&offset, 1),
      /*result_type=*/NULL, location,
      /*out_value=*/NULL);
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

static iree_status_t loom_aie2p_array_resident_build_insert_i32(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t vector, uint32_t lane, loom_value_id_t scalar,
    loom_location_id_t location, loom_value_id_t* out_vector) {
  loom_value_id_t lane_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, lane,
      location, /*value_name=*/NULL, &lane_value));
  // The insertion index is read from r29, not an encoded scalar register.
  // Preserve that domain on the value just as source-to-Low lowering does.
  loom_type_t index_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      builder->descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_MR29_INSERT, 1,
      &index_type));
  loom_op_t* index_copy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(ir_builder, lane_value,
                                           /*detached=*/false, index_type,
                                           location, &index_copy));
  const loom_value_id_t operands[] = {vector, loom_low_copy_result(index_copy),
                                      scalar};
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_INSERT_I32_REGISTER,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      &builder->vector512_type, location, out_vector);
}

static iree_status_t loom_aie2p_array_resident_build_scalar_fold_contribution(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_aie2p_array_resident_port_state_t* const* output_states,
    uint32_t output_start, uint32_t output_count, loom_location_id_t location,
    loom_value_id_t* out_accumulator) {
  const loom_aie2p_array_resident_port_state_t* first_state =
      output_states[output_start];
  loom_value_id_t first_scalar = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_load_i32(
      builder, ir_builder, first_state->current_address, location,
      &first_scalar));
  loom_value_id_t vector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I32X16,
      &first_scalar, 1, loom_named_attr_slice_empty(), &builder->vector512_type,
      location, &vector));
  for (uint32_t i = 1; i < output_count; ++i) {
    loom_value_id_t scalar = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_load_i32(
        builder, ir_builder, output_states[output_start + i]->current_address,
        location, &scalar));
    loom_value_id_t next_vector = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_insert_i32(
        builder, ir_builder, vector, i, scalar, location, &next_vector));
    vector = next_vector;
  }
  loom_value_id_t accumulator_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_op(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_MOVE_VECTOR512_TO_ACCUMULATOR512, &vector, 1,
      loom_named_attr_slice_empty(), &builder->accumulator512_type, location,
      &accumulator_lane));
  const loom_value_id_t lanes[] = {
      accumulator_lane,
      accumulator_lane,
      accumulator_lane,
      accumulator_lane,
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
    loom_value_id_t accumulator, uint32_t lane, loom_location_id_t location,
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
  const loom_named_attr_t lane_attr = {
      .name_id = builder->vector_index_name,
      .value = loom_attr_i64(lane),
  };
  return loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_EXTRACT_I32_IMMEDIATE,
      &sum_vector, 1, loom_make_named_attr_slice(&lane_attr, 1),
      &builder->lock_delta_type, location, out_value);
}

static iree_status_t loom_aie2p_array_resident_build_fold_contribution(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t address, uint32_t byte_length,
    loom_value_id_t zero_accumulator_lane, loom_location_id_t location,
    loom_value_id_t* out_accumulator) {
  if (byte_length == sizeof(float)) {
    loom_value_id_t scalar_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_load_i32(
        builder, ir_builder, address, location, &scalar_value));
    return loom_aie2p_array_resident_build_f32_accumulator(
        builder, ir_builder, scalar_value, zero_accumulator_lane, location,
        out_accumulator);
  }

  const uint32_t accumulator_lane_byte_length = 16 * sizeof(float);
  const uint32_t lane_count = byte_length / accumulator_lane_byte_length;
  loom_value_id_t lanes[4] = {
      zero_accumulator_lane,
      zero_accumulator_lane,
      zero_accumulator_lane,
      zero_accumulator_lane,
  };
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_array_resident_build_load_accumulator_f32x16(
            builder, ir_builder, address, i * accumulator_lane_byte_length,
            location, &lanes[i]));
  }
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      ir_builder, lanes, IREE_ARRAYSIZE(lanes), builder->accumulator2048_type,
      location, &concat_op));
  *out_accumulator = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_build_fold_store(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_value_id_t accumulator, loom_value_id_t address, uint32_t byte_length,
    loom_location_id_t location) {
  if (byte_length == sizeof(float)) {
    loom_value_id_t result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_extract_f32(
        builder, ir_builder, accumulator, /*lane=*/0, location, &result));
    return loom_aie2p_array_resident_build_store_i32(builder, ir_builder,
                                                     result, address, location);
  }

  const uint32_t accumulator_lane_byte_length = 16 * sizeof(float);
  const uint32_t lane_count = byte_length / accumulator_lane_byte_length;
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(ir_builder, accumulator, i,
                                              builder->accumulator512_type,
                                              location, &slice_op));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_array_resident_build_store_accumulator_f32x16(
            builder, ir_builder, loom_low_slice_result(slice_op), address,
            i * accumulator_lane_byte_length, location));
  }
  return iree_ok_status();
}

typedef enum loom_aie2p_array_fold_transfer_e {
  LOOM_AIE2P_ARRAY_FOLD_TRANSFER_INITIALIZE = 0,
  LOOM_AIE2P_ARRAY_FOLD_TRANSFER_ADD = 1,
  LOOM_AIE2P_ARRAY_FOLD_TRANSFER_COMPLETE = 2,
} loom_aie2p_array_fold_transfer_t;

static iree_status_t loom_aie2p_array_resident_build_fragment_address(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    const loom_aie2p_array_resident_port_state_t* output, uint32_t byte_offset,
    loom_location_id_t location, loom_value_id_t* out_address) {
  *out_address = output->current_address;
  if (byte_offset == 0) {
    return iree_ok_status();
  }
  loom_value_id_t address_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_MOVE_LOCAL_ADDRESS_TO_SCALAR,
      output->current_address, builder->lock_delta_type, location,
      &address_bits));
  loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32, byte_offset,
      location, /*value_name=*/NULL, &offset));
  loom_value_id_t fragment_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, address_bits,
      offset, location, &fragment_bits));
  return loom_aie2p_array_resident_build_unary(
      builder, ir_builder,
      AIE2P_CORE_DESCRIPTOR_REF_MOVE_SCALAR_TO_LOCAL_ADDRESS, fragment_bits,
      builder->endpoint_address_type, location, out_address);
}

// Repeated fragments share one loop body. Only the byte offset crosses its
// backedge; every accumulator reaches memory before the next fragment.
static iree_status_t loom_aie2p_array_resident_build_private_fold_span(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    const loom_aie2p_array_fold_state_plan_t* state,
    const loom_aie2p_array_fold_span_t* span,
    const loom_aie2p_array_resident_port_state_t* output,
    loom_aie2p_array_fold_transfer_t transfer, loom_value_id_t mode,
    loom_value_id_t zero_accumulator_lane, loom_location_id_t location) {
  loom_value_id_t output_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_fragment_address(
      builder, ir_builder, output, span->output_byte_offset, location,
      &output_address));
  loom_value_id_t state_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_address(
      builder, ir_builder, builder->endpoint_address_type,
      state->load_address + span->state_byte_offset, location, &state_address));

  loom_region_t* region = ir_builder->ip.block->parent_region;
  loom_block_t* loop_body = NULL;
  loom_block_t* loop_advance = NULL;
  loom_block_t* loop_exit = NULL;
  loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t end = LOOM_VALUE_ID_INVALID;
  if (span->repeat_count > 1) {
    loom_value_id_t output_base = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
        builder, ir_builder,
        AIE2P_CORE_DESCRIPTOR_REF_MOVE_LOCAL_ADDRESS_TO_SCALAR, output_address,
        builder->lock_delta_type, location, &output_base));
    loom_value_id_t state_base = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
        builder, ir_builder,
        AIE2P_CORE_DESCRIPTOR_REF_MOVE_LOCAL_ADDRESS_TO_SCALAR, state_address,
        builder->lock_delta_type, location, &state_base));
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, 0,
        location, /*value_name=*/NULL, &zero));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32,
        span->byte_length, location, /*value_name=*/NULL, &step));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32,
        span->byte_length * span->repeat_count, location, /*value_name=*/NULL,
        &end));
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(builder->module, region, &loop_body));
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(builder->module, region, &loop_advance));
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(builder->module, region, &loop_exit));
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        ir_builder, loop_body, builder->lock_delta_type, &offset));
    loom_op_t* entry_branch = NULL;
    IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, loop_body, &zero, 1,
                                           location, &entry_branch));
    loom_builder_set_block(ir_builder, loop_body);

    loom_value_id_t output_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, output_base,
        offset, location, &output_bits));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
        builder, ir_builder,
        AIE2P_CORE_DESCRIPTOR_REF_MOVE_SCALAR_TO_LOCAL_ADDRESS, output_bits,
        builder->endpoint_address_type, location, &output_address));
    loom_value_id_t state_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, state_base,
        offset, location, &state_bits));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_unary(
        builder, ir_builder,
        AIE2P_CORE_DESCRIPTOR_REF_MOVE_SCALAR_TO_LOCAL_ADDRESS, state_bits,
        builder->endpoint_address_type, location, &state_address));
  }

  const loom_value_id_t source_address =
      transfer == LOOM_AIE2P_ARRAY_FOLD_TRANSFER_COMPLETE ? state_address
                                                          : output_address;
  const loom_value_id_t target_address =
      transfer == LOOM_AIE2P_ARRAY_FOLD_TRANSFER_COMPLETE ? output_address
                                                          : state_address;
  loom_value_id_t contribution = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_fold_contribution(
      builder, ir_builder, source_address, span->byte_length,
      zero_accumulator_lane, location, &contribution));
  loom_value_id_t result = contribution;
  if (transfer == LOOM_AIE2P_ARRAY_FOLD_TRANSFER_ADD) {
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_fold_contribution(
        builder, ir_builder, state_address, span->byte_length,
        zero_accumulator_lane, location, &accumulator));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_add_f32_accumulators(
        builder, ir_builder, accumulator, contribution, mode, location,
        &result));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_fold_store(
      builder, ir_builder, result, target_address, span->byte_length,
      location));

  if (span->repeat_count > 1) {
    loom_value_id_t next_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, offset, step,
        location, &next_offset));
    loom_value_id_t has_more = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_binary(
        builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_NE_I32, next_offset,
        end, location, &has_more));
    loom_op_t* branch = NULL;
    IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
        ir_builder, has_more, loop_advance, loop_exit, location, &branch));
    loom_builder_set_block(ir_builder, loop_advance);
    IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, loop_body, &next_offset,
                                           1, location, &branch));
    loom_builder_set_block(ir_builder, loop_exit);
  }
  return iree_ok_status();
}

// Source output addresses and stores are untouched. Only final post-return
// contributions enter private state, with no accumulator live between spans
// or through another source firing.
static iree_status_t loom_aie2p_array_resident_build_private_fold_transfer(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    const loom_aie2p_array_fold_state_plan_t* state,
    loom_aie2p_array_resident_port_state_t* const* outputs,
    loom_aie2p_array_fold_transfer_t transfer, loom_value_id_t mode,
    loom_value_id_t zero_accumulator_lane, loom_location_id_t location) {
  loom_region_t* region = ir_builder->ip.block->parent_region;
  for (uint32_t i = 0; i < state->span_count; ++i) {
    const loom_aie2p_array_fold_span_t* span = &state->spans[i];
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_private_fold_span(
        builder, ir_builder, state, span, outputs[span->output_index], transfer,
        mode, zero_accumulator_lane, location));
    if (i + 1 < state->span_count) {
      loom_block_t* next_span = NULL;
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(builder->module, region, &next_span));
      loom_op_t* branch = NULL;
      IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, next_span,
                                             /*args=*/NULL, /*args_count=*/0,
                                             location, &branch));
      loom_builder_set_block(ir_builder, next_span);
    }
  }
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
      &builder->plan->locks[port->credit_lock_index + role];
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
  const loom_aie2p_array_worker_plan_t* worker_plan =
      &builder->plan->worker_plans[worker_index];
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
    for (uint32_t i = 0; i < worker_plan->port_count; ++i) {
      const loom_aie2p_array_worker_port_plan_t* port =
          &builder->plan->worker_ports[worker_plan->first_port + i];
      if (port->direction != direction) {
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
  const loom_aie2p_array_worker_plan_t* worker_plan =
      &builder->plan->worker_plans[worker_index];
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
    for (uint32_t i = 0; i < worker_plan->port_count; ++i) {
      const loom_aie2p_array_worker_port_plan_t* port =
          &builder->plan->worker_ports[worker_plan->first_port + i];
      if (port->direction != direction) {
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

static iree_status_t loom_aie2p_array_resident_initialize_port_state(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_block_t* firing_header,
    const loom_aie2p_array_worker_port_plan_t* port,
    loom_aie2p_array_resident_port_state_t* port_state) {
  const loom_aie2p_array_channel_t* channel =
      &builder->plan->channels[port->channel_index];
  *port_state = (loom_aie2p_array_resident_port_state_t){
      .port = port,
      .current_address = LOOM_VALUE_ID_INVALID,
      .current_slot = LOOM_VALUE_ID_INVALID,
      .next_address = LOOM_VALUE_ID_INVALID,
      .next_slot = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      ir_builder, firing_header, builder->endpoint_address_type,
      &port_state->current_address));
  if (channel->capacity > 2) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        ir_builder, firing_header, builder->lock_delta_type,
        &port_state->current_slot));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_bind_resources(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, loom_block_t* firing_header,
    loom_block_t* source_entry,
    loom_aie2p_array_resident_port_state_t* port_states) {
  const loom_aie2p_array_worker_plan_t* worker_plan =
      &builder->plan->worker_plans[worker_index];
  const loom_aie2p_array_worker_t* worker =
      &builder->plan->workers[worker_index];
  iree_host_size_t resource_ordinal = 0;
  loom_op_t* op = source_entry->first_op;
  while (op != NULL) {
    loom_op_t* next_op = op->next_op;
    if (loom_low_resource_isa(op)) {
      const uint32_t port_index =
          builder->plan->worker_resource_ports[worker_plan->first_port +
                                               resource_ordinal++];
      const loom_aie2p_array_worker_port_plan_t* port =
          &builder->plan->worker_ports[port_index];
      const loom_value_id_t resource_value = loom_low_resource_result(op);
      loom_aie2p_array_resident_port_state_t* port_state =
          &port_states[port->resident_state_ordinal];
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_initialize_port_state(
          builder, ir_builder, firing_header, port, port_state));
      // The worker may consume its imported pointer through native address
      // updates. Keep that value identity separate from the ring address used
      // by the latch; allocation can coalesce copies that remain read-only.
      loom_builder_set_before(ir_builder, op);
      loom_op_t* copy_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_copy_build(
          ir_builder, port_state->current_address, /*detached=*/false,
          builder->endpoint_address_type, op->location, &copy_op));
      const loom_value_id_t local_address = loom_low_copy_result(copy_op);
      IREE_RETURN_IF_ERROR(loom_module_move_value_name(
          builder->module, resource_value, local_address));
      IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_with(
          builder->module, resource_value, local_address));
      IREE_RETURN_IF_ERROR(loom_op_erase(builder->module, op));
    }
    op = next_op;
  }
  IREE_ASSERT_EQ(resource_ordinal, worker->leaf->requirements.resource_count);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_resident_initialize_generated_states(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    uint32_t worker_index, loom_block_t* firing_header,
    loom_aie2p_array_resident_port_state_t* port_states) {
  const loom_aie2p_array_worker_plan_t* worker_plan =
      &builder->plan->worker_plans[worker_index];
  const uint32_t imported_port_count =
      (uint32_t)builder->plan->workers[worker_index]
          .leaf->requirements.resource_count;
  for (uint32_t i = 0; i < worker_plan->port_count; ++i) {
    const loom_aie2p_array_worker_port_plan_t* port =
        &builder->plan->worker_ports[worker_plan->first_port + i];
    if (port->resident_state_ordinal == UINT32_MAX ||
        port->resident_state_ordinal < imported_port_count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_initialize_port_state(
        builder, ir_builder, firing_header, port,
        &port_states[port->resident_state_ordinal]));
  }
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
        ir_builder, block, builder->endpoint_address_type,
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
        builder, ir_builder, builder->endpoint_address_type,
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
        next_address_bits, builder->endpoint_address_type, location,
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
      ir_builder, merge_block, builder->endpoint_address_type,
      &next_address_arg));
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
        builder, ir_builder, builder->endpoint_address_type,
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

static iree_status_t loom_aie2p_array_resident_build_zero_accumulator(
    loom_aie2p_array_resident_builder_t* builder, loom_builder_t* ir_builder,
    loom_location_id_t location, loom_value_id_t* out_accumulator,
    loom_value_id_t* out_lane) {
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_op(
      builder, ir_builder, AIE2P_CORE_DESCRIPTOR_REF_ACCUMULATOR_CLEAR_F32X64,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &builder->accumulator2048_type, location, out_accumulator));
  if (out_lane == NULL) {
    return iree_ok_status();
  }
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
  const uint32_t output_count = worker->fold_output_count;
  const loom_aie2p_array_worker_plan_t* worker_plan =
      &builder->plan->worker_plans[worker_index];
  const loom_aie2p_array_fold_state_plan_t* private_state =
      &worker_plan->fold_state;
  const bool use_private_state = private_state->byte_length != 0;
  loom_aie2p_array_resident_port_state_t* activation_states = NULL;
  if (port_state_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, port_state_count, sizeof(*activation_states),
        (void**)&activation_states));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_define_state_arguments(
      builder, ir_builder, activation_header, record_states, port_state_count,
      activation_states));

  loom_aie2p_array_resident_port_state_t** output_states = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, output_count,
                                                 sizeof(*output_states),
                                                 (void**)&output_states));
  for (uint32_t i = 0; i < output_count; ++i) {
    const uint32_t state_ordinal =
        builder->plan->worker_fold_output_states[worker_plan->first_port + i];
    output_states[i] = &record_states[state_ordinal];
  }

  bool pack_scalar_outputs = true;
  for (uint32_t i = 0; i < output_count; ++i) {
    const loom_aie2p_array_channel_t* output_channel =
        &builder->plan->channels[output_states[i]->port->channel_index];
    if (output_channel->record_byte_length != sizeof(float)) {
      pack_scalar_outputs = false;
      break;
    }
  }
  const uint32_t accumulator_count =
      use_private_state ? 0
                        : (pack_scalar_outputs
                               ? 1 + (output_count - 1) /
                                         LOOM_AIE2P_ARRAY_SCALAR_FOLD_PACK_WIDTH
                               : output_count);

  iree_host_size_t fold_value_count = 0;
  if (!iree_host_size_checked_mul(accumulator_count, 5, &fold_value_count) ||
      !iree_host_size_checked_add(fold_value_count, 2, &fold_value_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P folded worker state is too large");
  }
  loom_value_id_t* fold_values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->arena, fold_value_count,
                                sizeof(*fold_values), (void**)&fold_values));
  loom_value_id_t* accumulators = fold_values;
  loom_value_id_t* contributions = accumulators + accumulator_count;
  loom_value_id_t* merged_accumulators = contributions + accumulator_count;
  loom_value_id_t* added_accumulators = merged_accumulators + accumulator_count;
  loom_value_id_t* record_state = added_accumulators + accumulator_count;

  for (uint32_t i = 0; i < accumulator_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        ir_builder, record_header, builder->accumulator2048_type,
        &accumulators[i]));
  }
  loom_value_id_t remaining_records = LOOM_VALUE_ID_INVALID;
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
      pack_scalar_outputs ? NULL : &zero_accumulator_lane));

  loom_builder_set_block(ir_builder, activation_header);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_acquires(
      builder, ir_builder, worker_index,
      LOOM_AIE2P_ARRAY_PORT_DIRECTION_FLAG_SEND, acquire_delta, location));
  for (uint32_t i = 0; i < accumulator_count; ++i) {
    record_state[i] = zero_accumulator;
  }
  record_state[accumulator_count] = record_count;
  record_state[accumulator_count + 1] = release_delta;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_state_branch(
      builder, ir_builder, record_header, activation_states, port_state_count,
      LOOM_AIE2P_ARRAY_STATE_VALUE_CURRENT, record_state, accumulator_count + 2,
      location));

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
  for (uint32_t i = 0; i < accumulator_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        ir_builder, accumulation_merge_block, builder->accumulator2048_type,
        &merged_accumulators[i]));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_rewrite_returns(
      builder, ir_builder, resident_body, source_block_start,
      source_body->block_count, record_latch));

  loom_builder_set_block(ir_builder, record_latch);
  if (!use_private_state && pack_scalar_outputs) {
    for (uint32_t i = 0; i < accumulator_count; ++i) {
      const uint32_t output_start = i * LOOM_AIE2P_ARRAY_SCALAR_FOLD_PACK_WIDTH;
      const uint32_t packed_output_count =
          iree_min(output_count - output_start,
                   (uint32_t)LOOM_AIE2P_ARRAY_SCALAR_FOLD_PACK_WIDTH);
      IREE_RETURN_IF_ERROR(
          loom_aie2p_array_resident_build_scalar_fold_contribution(
              builder, ir_builder, output_states, output_start,
              packed_output_count, location, &contributions[i]));
    }
  } else if (!use_private_state) {
    for (uint32_t i = 0; i < output_count; ++i) {
      const loom_aie2p_array_resident_port_state_t* output_state =
          output_states[i];
      const loom_aie2p_array_channel_t* output_channel =
          &builder->plan->channels[output_state->port->channel_index];
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_fold_contribution(
          builder, ir_builder, output_state->current_address,
          output_channel->record_byte_length, zero_accumulator_lane, location,
          &contributions[i]));
    }
  }
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
  if (use_private_state) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_private_fold_transfer(
        builder, ir_builder, private_state, output_states,
        LOOM_AIE2P_ARRAY_FOLD_TRANSFER_INITIALIZE, mode, zero_accumulator_lane,
        location));
  }
  loom_op_t* initialized_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, accumulation_merge_block,
                                         contributions, accumulator_count,
                                         location, &initialized_branch));

  loom_builder_set_block(ir_builder, add_block);
  if (use_private_state) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_private_fold_transfer(
        builder, ir_builder, private_state, output_states,
        LOOM_AIE2P_ARRAY_FOLD_TRANSFER_ADD, mode, zero_accumulator_lane,
        location));
  }
  for (uint32_t i = 0; i < accumulator_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_add_f32_accumulators(
        builder, ir_builder, accumulators[i], contributions[i], mode, location,
        &added_accumulators[i]));
  }
  loom_op_t* accumulated_branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(ir_builder, accumulation_merge_block,
                                         added_accumulators, accumulator_count,
                                         location, &accumulated_branch));

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
  for (uint32_t i = 0; i < accumulator_count; ++i) {
    record_state[i] = merged_accumulators[i];
  }
  record_state[accumulator_count] = next_remaining_records;
  record_state[accumulator_count + 1] = zero;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_state_branch(
      builder, ir_builder, record_header, record_states, port_state_count,
      LOOM_AIE2P_ARRAY_STATE_VALUE_NEXT, record_state, accumulator_count + 2,
      location));

  loom_builder_set_block(ir_builder, completion_block);
  if (use_private_state) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_private_fold_transfer(
        builder, ir_builder, private_state, output_states,
        LOOM_AIE2P_ARRAY_FOLD_TRANSFER_COMPLETE, mode, zero_accumulator_lane,
        location));
  } else if (pack_scalar_outputs) {
    for (uint32_t i = 0; i < output_count; ++i) {
      loom_value_id_t result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_extract_f32(
          builder, ir_builder,
          merged_accumulators[i / LOOM_AIE2P_ARRAY_SCALAR_FOLD_PACK_WIDTH],
          i % LOOM_AIE2P_ARRAY_SCALAR_FOLD_PACK_WIDTH, location, &result));
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_store_i32(
          builder, ir_builder, result, output_states[i]->current_address,
          location));
    }
  } else {
    for (uint32_t i = 0; i < output_count; ++i) {
      const loom_aie2p_array_resident_port_state_t* output_state =
          output_states[i];
      const loom_aie2p_array_channel_t* output_channel =
          &builder->plan->channels[output_state->port->channel_index];
      IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_fold_store(
          builder, ir_builder, merged_accumulators[i],
          output_state->current_address, output_channel->record_byte_length,
          location));
    }
  }
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

// Materialization introduces unconditional edges around the cloned firing.
// Contract them before scheduling so protocol boundaries do not force unrelated
// register work or outstanding memory operations to drain an entire block.
static iree_status_t loom_aie2p_array_resident_fuse_blocks(
    loom_module_t* module, loom_region_t* body,
    iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, &scratch_arena);
  loom_cfg_graph_t graph;
  iree_status_t status =
      loom_cfg_graph_build(module, body, &scratch_arena, &graph);
  loom_dominance_info_t dominance = {
      .module = module,
      .arena = &scratch_arena,
  };
  if (iree_status_is_ok(status)) {
    status = loom_dominance_info_add_cfg_graph(&dominance, &graph, NULL);
  }
  if (iree_status_is_ok(status)) {
    uint16_t fused_count = 0;
    status = loom_cfg_fuse_single_predecessor_blocks(
        &rewriter, &graph, &dominance, &scratch_arena, &fused_count);
  }
  // Fusion invalidates the snapshot. Leaf compilation builds its own facts
  // from the finished resident CFG after this temporary storage is released.
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static iree_status_t loom_aie2p_array_resident_materialize_worker(
    loom_aie2p_array_resident_builder_t* builder, uint32_t worker_index,
    loom_aie2p_array_resident_worker_t* out_worker) {
  const loom_aie2p_array_worker_t* worker =
      &builder->plan->workers[worker_index];
  const loom_aie2p_array_worker_plan_t* worker_plan =
      &builder->plan->worker_plans[worker_index];
  const loom_aie2p_array_leaf_t* leaf = worker->leaf;
  const loom_op_t* source_function = leaf->function_op;
  const loom_region_t* source_body = loom_low_func_def_body(source_function);

  loom_symbol_ref_t resident_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_add_symbol(
      builder, worker_index, &resident_ref));
  loom_low_memory_access_map_t* memory_accesses = NULL;
  loom_ir_remap_options_t remap_options = {0};
  if (leaf->memory_accesses != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_memory_access_map_create(
        &builder->module->arena, &memory_accesses));
    loom_low_memory_access_clone_t* clone = NULL;
    IREE_RETURN_IF_ERROR(loom_low_memory_access_clone_create(
        leaf->memory_accesses, memory_accesses, builder->arena, &clone));
    remap_options.clone_observer = (loom_ir_clone_observer_t){
        .fn = loom_low_memory_access_clone_op,
        .user_data = clone,
    };
  }
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(builder->source_module,
                                                builder->module, builder->arena,
                                                &remap_options, &remap));
  loom_location_id_t resident_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &remap, source_function->location, &resident_location));

  loom_builder_t ir_builder;
  loom_builder_initialize(builder->module, &builder->module->arena,
                          loom_module_block(builder->module), &ir_builder);
  loom_low_func_def_build_flags_t build_flags =
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN;
  loom_op_t* resident_function = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &ir_builder, build_flags,
      /*visibility=*/0, LOOM_LOW_RETAIN_RETAIN,
      /*cc=*/0, /*purity=*/0, /*inline_policy=*/0, /*allocation=*/0,
      /*schedule=*/0, builder->representation_contract, loom_symbol_ref_null(),
      /*abi=*/0, loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), resident_ref,
      /*arg_types=*/NULL, /*arg_types_count=*/0,
      /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, resident_location,
      &resident_function));
  loom_region_t* resident_body = loom_low_func_def_body(resident_function);
  resident_body->flags = source_body->flags;
  resident_body->source_flags = source_body->source_flags;
  loom_block_t* preheader = loom_region_entry_block(resident_body);
  loom_builder_enter_region(&ir_builder, resident_function, resident_body);

  loom_value_id_t acquire_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, &ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, -1,
      resident_location, "acquire_delta", &acquire_delta));
  loom_value_id_t release_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_build_constant(
      builder, &ir_builder, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, 1,
      resident_location, "release_delta", &release_delta));

  loom_block_t* activation_header = NULL;
  IREE_RETURN_IF_ERROR(loom_region_append_block(builder->module, resident_body,
                                                &activation_header));
  loom_block_t* record_header = activation_header;
  if (worker->fold_record_count != 0) {
    IREE_RETURN_IF_ERROR(loom_region_append_block(
        builder->module, resident_body, &record_header));
  }
  const uint16_t source_block_start = resident_body->block_count;
  IREE_RETURN_IF_ERROR(loom_ir_clone_region_blocks(
      &ir_builder, source_body, resident_body, source_block_start, &remap));
  loom_block_t* source_entry =
      loom_region_block(resident_body, source_block_start);

  const iree_host_size_t port_state_count = worker_plan->resident_state_count;
  loom_aie2p_array_resident_port_state_t* port_states = NULL;
  if (port_state_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, port_state_count,
                                  sizeof(*port_states), (void**)&port_states));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_bind_resources(
      builder, &ir_builder, worker_index, record_header, source_entry,
      port_states));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_initialize_generated_states(
      builder, &ir_builder, worker_index, record_header, port_states));
  if (worker->fold_record_count == 0) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_materialize_recordwise_body(
        builder, &ir_builder, worker_index, resident_body, source_body,
        source_block_start, record_header, source_entry, preheader, port_states,
        port_state_count, acquire_delta, release_delta, resident_location));
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_materialize_folded_body(
        builder, &ir_builder, worker_index, worker, resident_body, source_body,
        source_block_start, activation_header, record_header, source_entry,
        preheader, port_states, port_state_count, acquire_delta, release_delta,
        resident_location));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_array_resident_fuse_blocks(
      builder->module, resident_body, builder->arena->block_pool));
  *out_worker = (loom_aie2p_array_resident_worker_t){
      .worker_index = worker_index,
      .entry = resident_ref,
      .function_op = resident_function,
      .function_target_facts = leaf->function_target_facts,
      .memory_accesses = memory_accesses,
  };
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_materialize_resident_program(
    const loom_module_t* source_module, loom_module_t* resident_module,
    const loom_aie2p_array_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_array_resident_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(resident_module);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_program);
  IREE_ASSERT(source_module != resident_module);
  IREE_ASSERT(source_module->context == resident_module->context);
  *out_program = (loom_aie2p_array_resident_program_t){0};

  loom_aie2p_array_resident_worker_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->worker_count, sizeof(*workers), (void**)&workers));
  loom_aie2p_array_resident_builder_t builder = {
      .source_module = source_module,
      .module = resident_module,
      .plan = plan,
      .arena = arena,
      .descriptor_set = loom_aie2p_core_descriptor_set(),
  };
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      resident_module,
      loom_low_descriptor_set_string(builder.descriptor_set,
                                     builder.descriptor_set->key_string_ref),
      &builder.representation_contract));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      resident_module, IREE_SV("i"), &builder.integer_immediate_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      resident_module, IREE_SV("imm"), &builder.memory_immediate_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      resident_module, IREE_SV("idx"), &builder.vector_index_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(resident_module, IREE_SV("id"),
                                                 &builder.lock_selector_name));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      builder.descriptor_set, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1,
      &builder.endpoint_address_type));
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
