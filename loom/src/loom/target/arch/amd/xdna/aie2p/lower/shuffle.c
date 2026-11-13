// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/shuffle.h"

#include <limits.h>
#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/vector_packet.h"
#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"
#include "loom/target/registers.h"

enum {
  LOOM_AIE2P_SHUFFLE_MAX_PACKET_COUNT = 4,
  LOOM_AIE2P_SHUFFLE_PACKET_GENERIC = 4,
  LOOM_AIE2P_SHUFFLE_PACKET_ENCODING_MASK = 0x7,
};

typedef struct loom_aie2p_shuffle_plan_t {
  // Physical carrier shared by the source and result.
  loom_aie2p_vector_carrier_kind_t carrier_kind;
  // Raw payload width used by vector broadcast and select instructions.
  uint8_t element_bit_count;
  // Number of physical units occupied by the complete carrier.
  uint8_t carrier_unit_count;
  // Number of 512-bit or 64-predicate-bit packets with logical payload.
  uint8_t logical_packet_count;
  // Number of logical lanes in the rank-one source and result vectors.
  uint16_t element_count;
  // Three-bit source packet ordinals, or PACKET_GENERIC, by result packet.
  uint16_t packet_source_aliases;
} loom_aie2p_shuffle_plan_t;
static_assert(sizeof(loom_aie2p_shuffle_plan_t) == 8,
              "AIE2P shuffle plans must stay cache dense");

typedef struct loom_aie2p_shuffle_emit_state_t {
  loom_low_lower_context_t* context;
  const loom_op_t* source_op;
  const loom_aie2p_shuffle_plan_t* plan;
  loom_attribute_t source_lanes;
  loom_value_id_t low_source;
  loom_aie2p_vector_packet_emitter_t packet_emitter;
  loom_type_t native_packet_type;
  loom_type_t result_type;
  loom_string_id_t lane_immediate_name;
  loom_value_id_t source_native_packets[LOOM_AIE2P_SHUFFLE_MAX_PACKET_COUNT];
  loom_value_id_t source_vector_packets[LOOM_AIE2P_SHUFFLE_MAX_PACKET_COUNT];
} loom_aie2p_shuffle_emit_state_t;

static uint8_t loom_aie2p_shuffle_packet_lane_count(uint8_t element_bit_count) {
  return element_bit_count == 1 ? 64 : (uint8_t)(512 / element_bit_count);
}

static uint8_t loom_aie2p_shuffle_packet_source_alias(
    const loom_aie2p_shuffle_plan_t* plan, uint8_t result_packet) {
  return (uint8_t)((plan->packet_source_aliases >> (result_packet * 3)) &
                   LOOM_AIE2P_SHUFFLE_PACKET_ENCODING_MASK);
}

static bool loom_aie2p_shuffle_plan_uses_broadcast_select(
    const loom_aie2p_shuffle_plan_t* plan) {
  for (uint8_t packet = 0; packet < plan->logical_packet_count; ++packet) {
    if (loom_aie2p_shuffle_packet_source_alias(plan, packet) ==
        LOOM_AIE2P_SHUFFLE_PACKET_GENERIC) {
      return true;
    }
  }
  return false;
}

static bool loom_aie2p_shuffle_packet_is_source_alias(
    const loom_aie2p_shuffle_plan_t* plan, loom_attribute_t source_lanes,
    uint8_t result_packet, uint8_t* out_source_packet) {
  const uint8_t packet_lane_count =
      loom_aie2p_shuffle_packet_lane_count(plan->element_bit_count);
  const uint16_t result_lane_base = result_packet * packet_lane_count;
  const uint16_t live_lane_count = (uint16_t)iree_min(
      packet_lane_count, plan->element_count - result_lane_base);
  const uint16_t first_source_lane =
      (uint16_t)source_lanes.i64_array[result_lane_base];
  const uint8_t source_packet = first_source_lane / packet_lane_count;
  const uint16_t source_lane_base = source_packet * packet_lane_count;
  for (uint16_t lane = 0; lane < live_lane_count; ++lane) {
    if (source_lanes.i64_array[result_lane_base + lane] !=
        source_lane_base + lane) {
      return false;
    }
  }
  *out_source_packet = source_packet;
  return true;
}

static bool loom_aie2p_shuffle_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_aie2p_shuffle_plan_t* out_plan) {
  *out_plan = (loom_aie2p_shuffle_plan_t){0};
  if (!loom_vector_shuffle_isa(source_op)) {
    return false;
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_shuffle_source(source_op));
  const loom_aie2p_vector_carrier_t carrier =
      loom_aie2p_vector_carrier_for_type(source_type);
  if (carrier.kind == LOOM_AIE2P_VECTOR_CARRIER_NONE) {
    return false;
  }

  const uint16_t element_bit_count = loom_aie2p_scalar_type_physical_bit_count(
      loom_type_element_type(source_type));
  const int64_t element_count = loom_type_dim_static_size_at(source_type, 0);

  const uint8_t packet_lane_count =
      loom_aie2p_shuffle_packet_lane_count((uint8_t)element_bit_count);
  *out_plan = (loom_aie2p_shuffle_plan_t){
      .carrier_kind = carrier.kind,
      .element_bit_count = (uint8_t)element_bit_count,
      .carrier_unit_count = (uint8_t)carrier.unit_count,
      .logical_packet_count =
          (uint8_t)((element_count + packet_lane_count - 1) /
                    packet_lane_count),
      .element_count = (uint16_t)element_count,
  };
  const loom_attribute_t source_lanes =
      loom_vector_shuffle_source_lanes(source_op);
  for (uint8_t packet = 0; packet < out_plan->logical_packet_count; ++packet) {
    uint8_t source_packet = 0;
    if (!loom_aie2p_shuffle_packet_is_source_alias(out_plan, source_lanes,
                                                   packet, &source_packet)) {
      source_packet = LOOM_AIE2P_SHUFFLE_PACKET_GENERIC;
    }
    out_plan->packet_source_aliases |= (uint16_t)source_packet << (packet * 3);
  }
  return true;
}

static bool loom_aie2p_shuffle_is_identity(
    const loom_aie2p_shuffle_plan_t* plan) {
  for (uint8_t packet = 0; packet < plan->logical_packet_count; ++packet) {
    if (loom_aie2p_shuffle_packet_source_alias(plan, packet) != packet) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_aie2p_select_shuffle_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  loom_aie2p_shuffle_plan_t matched_plan = {0};
  if (!loom_aie2p_shuffle_plan_from_op(loom_low_lower_context_module(context),
                                       source_op, &matched_plan)) {
    return iree_ok_status();
  }

  loom_aie2p_shuffle_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = matched_plan;
  if (loom_aie2p_shuffle_is_identity(retained_plan)) {
    loom_low_lower_inherit_value_type(context,
                                      loom_vector_shuffle_source(source_op),
                                      loom_vector_shuffle_result(source_op));
  } else {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
        context, retained_plan->carrier_kind, retained_plan->carrier_unit_count,
        &result_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
        context, loom_vector_shuffle_result(source_op), result_type));
  }
  *out_plan =
      loom_low_lower_plan_make(LOOM_AIE2P_PLAN_SHUFFLE_FIXED, retained_plan);
  return iree_ok_status();
}

void loom_aie2p_mark_shuffle_plan_demands(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan) {
  (void)plan;
  loom_low_lower_require_source_value_storage(
      context, loom_vector_shuffle_source(source_op));
}

void loom_aie2p_describe_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  (void)context;
  (void)source_op;
  const loom_aie2p_shuffle_plan_t* shuffle_plan =
      (const loom_aie2p_shuffle_plan_t*)plan.target_data;
  iree_string_view_t plan_key = IREE_SV("fixed-permutation.packet-routing");
  if (loom_aie2p_shuffle_plan_uses_broadcast_select(shuffle_plan)) {
    switch (shuffle_plan->element_bit_count) {
      case 1:
        plan_key = IREE_SV("fixed-permutation.predicate.broadcast-select");
        break;
      case 8:
        plan_key = IREE_SV("fixed-permutation.i8.broadcast-select");
        break;
      case 16:
        plan_key = IREE_SV("fixed-permutation.i16.broadcast-select");
        break;
      case 32:
        plan_key = IREE_SV("fixed-permutation.i32.broadcast-select");
        break;
      case 64:
        plan_key = IREE_SV("fixed-permutation.i64.broadcast-select");
        break;
      default:
        IREE_ASSERT_UNREACHABLE("selected AIE2P shuffle payload width");
        break;
    }
  }
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = plan_key,
  };
}

static iree_status_t loom_aie2p_shuffle_emit_descriptor_op(
    loom_aie2p_shuffle_emit_state_t* state, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_value_id_t* out_result) {
  return loom_aie2p_vector_packet_emit_descriptor_op(
      &state->packet_emitter, descriptor_ordinal, operands, operand_count,
      attrs, result_type, tied_results, tied_result_count, out_result);
}

static int64_t loom_aie2p_shuffle_signed_i32_bits(uint32_t value) {
  return value <= INT32_MAX ? (int64_t)value
                            : (int64_t)value - (INT64_C(1) << 32);
}

static loom_named_attr_t loom_aie2p_shuffle_immediate_attr(
    loom_string_id_t name, int64_t value) {
  return (loom_named_attr_t){
      .name_id = name,
      .value = loom_attr_i64(value),
  };
}

static iree_status_t loom_aie2p_shuffle_emit_constant(
    loom_aie2p_shuffle_emit_state_t* state, uint32_t descriptor_ordinal,
    int64_t value, loom_type_t result_type, loom_value_id_t* out_result) {
  return loom_aie2p_vector_packet_emit_constant(&state->packet_emitter,
                                                descriptor_ordinal, value,
                                                result_type, out_result);
}

static iree_status_t loom_aie2p_shuffle_source_native_packet(
    loom_aie2p_shuffle_emit_state_t* state, uint8_t packet,
    loom_value_id_t* out_packet) {
  loom_value_id_t* cached = &state->source_native_packets[packet];
  if (*cached != LOOM_VALUE_ID_INVALID) {
    *out_packet = *cached;
    return iree_ok_status();
  }

  const loom_aie2p_vector_carrier_t carrier = {
      .kind = state->plan->carrier_kind,
      .unit_count = state->plan->carrier_unit_count,
  };
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_native(
      &state->packet_emitter, state->low_source, carrier, packet, cached));
  *out_packet = *cached;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_shuffle_source_vector_packet(
    loom_aie2p_shuffle_emit_state_t* state, uint8_t packet,
    loom_value_id_t* out_packet) {
  loom_value_id_t* cached = &state->source_vector_packets[packet];
  if (*cached != LOOM_VALUE_ID_INVALID) {
    *out_packet = *cached;
    return iree_ok_status();
  }

  loom_value_id_t native_packet = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_shuffle_source_native_packet(state, packet, &native_packet));
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_native_to_vector(
      &state->packet_emitter, state->plan->carrier_kind, native_packet,
      cached));
  *out_packet = *cached;
  return iree_ok_status();
}

static uint32_t loom_aie2p_shuffle_broadcast_descriptor(
    uint8_t element_bit_count) {
  switch (element_bit_count) {
    case 1:
    case 8:
      return AIE2P_CORE_DESCRIPTOR_REF_BROADCAST_I8X64_FROM_VECTOR;
    case 16:
      return AIE2P_CORE_DESCRIPTOR_REF_BROADCAST_I16X32_FROM_VECTOR;
    case 32:
      return AIE2P_CORE_DESCRIPTOR_REF_BROADCAST_I32X16_FROM_VECTOR;
    case 64:
      return AIE2P_CORE_DESCRIPTOR_REF_BROADCAST_I64X8_FROM_VECTOR;
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P shuffle broadcast width");
      return 0;
  }
}

static uint32_t loom_aie2p_shuffle_select_descriptor(
    uint8_t element_bit_count) {
  switch (element_bit_count) {
    case 1:
    case 8:
      return AIE2P_CORE_DESCRIPTOR_REF_SELECT_I8X64;
    case 16:
      return AIE2P_CORE_DESCRIPTOR_REF_SELECT_I16X32_MASK64;
    case 32:
    case 64:
      return AIE2P_CORE_DESCRIPTOR_REF_SELECT_I32X16_MASK64;
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P shuffle select width");
      return 0;
  }
}

static iree_status_t loom_aie2p_shuffle_emit_broadcast(
    loom_aie2p_shuffle_emit_state_t* state, uint16_t source_lane,
    loom_value_id_t* out_broadcast) {
  const uint8_t packet_lane_count =
      loom_aie2p_shuffle_packet_lane_count(state->plan->element_bit_count);
  const uint8_t source_packet = source_lane / packet_lane_count;
  const uint8_t packet_lane = source_lane % packet_lane_count;
  loom_value_id_t source_vector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_shuffle_source_vector_packet(
      state, source_packet, &source_vector));
  const loom_named_attr_t immediate = loom_aie2p_shuffle_immediate_attr(
      state->lane_immediate_name, packet_lane);
  return loom_aie2p_shuffle_emit_descriptor_op(
      state,
      loom_aie2p_shuffle_broadcast_descriptor(state->plan->element_bit_count),
      &source_vector, 1, loom_make_named_attr_slice(&immediate, 1),
      state->packet_emitter.vector_type, /*tied_results=*/NULL,
      /*tied_result_count=*/0, out_broadcast);
}

static iree_status_t loom_aie2p_shuffle_emit_selector(
    loom_aie2p_shuffle_emit_state_t* state, uint64_t mask,
    loom_value_id_t* out_selector) {
  if (state->plan->element_bit_count == 1 ||
      state->plan->element_bit_count == 8) {
    return loom_aie2p_vector_packet_emit_byte_selector(&state->packet_emitter,
                                                       mask, out_selector);
  }

  const uint32_t low_word = (uint32_t)mask;
  return loom_aie2p_shuffle_emit_constant(
      state, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_PREDICATE_LOW32,
      loom_aie2p_shuffle_signed_i32_bits(low_word),
      state->packet_emitter.predicate_type, out_selector);
}

static uint64_t loom_aie2p_shuffle_selector_lane_mask(uint8_t element_bit_count,
                                                      uint8_t lane) {
  if (element_bit_count == 64) {
    return UINT64_C(3) << (lane * 2);
  }
  return UINT64_C(1) << lane;
}

static iree_status_t loom_aie2p_shuffle_emit_vector_packet(
    loom_aie2p_shuffle_emit_state_t* state, uint8_t result_packet,
    loom_value_id_t* out_packet) {
  const uint8_t packet_lane_count =
      loom_aie2p_shuffle_packet_lane_count(state->plan->element_bit_count);
  const uint16_t result_lane_base = result_packet * packet_lane_count;
  const uint8_t live_lane_count = (uint8_t)iree_min(
      packet_lane_count, state->plan->element_count - result_lane_base);
  loom_value_id_t composed = LOOM_VALUE_ID_INVALID;
  for (uint8_t lane = 0; lane < live_lane_count; ++lane) {
    const uint16_t source_lane =
        (uint16_t)state->source_lanes.i64_array[result_lane_base + lane];
    bool is_first_occurrence = true;
    for (uint8_t prior_lane = 0; prior_lane < lane; ++prior_lane) {
      if (state->source_lanes.i64_array[result_lane_base + prior_lane] ==
          source_lane) {
        is_first_occurrence = false;
        break;
      }
    }
    if (!is_first_occurrence) {
      continue;
    }

    uint64_t source_mask = 0;
    for (uint8_t result_lane = lane; result_lane < live_lane_count;
         ++result_lane) {
      if (state->source_lanes.i64_array[result_lane_base + result_lane] ==
          source_lane) {
        source_mask |= loom_aie2p_shuffle_selector_lane_mask(
            state->plan->element_bit_count, result_lane);
      }
    }
    loom_value_id_t broadcast = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_shuffle_emit_broadcast(state, source_lane, &broadcast));
    if (composed == LOOM_VALUE_ID_INVALID) {
      composed = broadcast;
      continue;
    }
    loom_value_id_t selector = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_shuffle_emit_selector(state, source_mask, &selector));
    const loom_value_id_t select_operands[] = {composed, broadcast, selector};
    IREE_RETURN_IF_ERROR(loom_aie2p_shuffle_emit_descriptor_op(
        state,
        loom_aie2p_shuffle_select_descriptor(state->plan->element_bit_count),
        select_operands, IREE_ARRAYSIZE(select_operands),
        loom_named_attr_slice_empty(), state->packet_emitter.vector_type,
        /*tied_results=*/NULL, /*tied_result_count=*/0, &composed));
  }
  IREE_ASSERT_NE(composed, LOOM_VALUE_ID_INVALID);
  *out_packet = composed;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_shuffle_vector_to_native_packet(
    loom_aie2p_shuffle_emit_state_t* state, loom_value_id_t vector_packet,
    loom_value_id_t* out_packet) {
  return loom_aie2p_vector_packet_write_native(&state->packet_emitter,
                                               state->plan->carrier_kind,
                                               vector_packet, out_packet);
}

static iree_status_t loom_aie2p_shuffle_emit_state_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_aie2p_shuffle_plan_t* plan,
    loom_aie2p_shuffle_emit_state_t* out_state) {
  *out_state = (loom_aie2p_shuffle_emit_state_t){
      .context = context,
      .source_op = source_op,
      .plan = plan,
      .source_lanes = loom_vector_shuffle_source_lanes(source_op),
      .low_source = LOOM_VALUE_ID_INVALID,
      .native_packet_type = loom_type_none(),
      .result_type = loom_type_none(),
      .lane_immediate_name = LOOM_STRING_ID_INVALID,
      .source_native_packets = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                                LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID},
      .source_vector_packets = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                                LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID},
  };
  out_state->low_source = loom_low_lower_lookup_value(
      context, loom_vector_shuffle_source(source_op));
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emitter_initialize(
      context, source_op, &out_state->packet_emitter));

  out_state->result_type = loom_low_lower_value_binding_type(
      context, loom_vector_shuffle_result(source_op));
  const uint8_t units_per_packet =
      loom_aie2p_vector_packet_carrier_unit_count(plan->carrier_kind);
  out_state->native_packet_type =
      loom_low_register_carrier_type_with_unit_count(out_state->result_type,
                                                     units_per_packet);
  if (!loom_aie2p_shuffle_plan_uses_broadcast_select(plan)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  return loom_builder_intern_string(builder, IREE_SV("idx"),
                                    &out_state->lane_immediate_name);
}

iree_status_t loom_aie2p_emit_shuffle_plan(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t plan) {
  const loom_aie2p_shuffle_plan_t* shuffle_plan =
      (const loom_aie2p_shuffle_plan_t*)plan.target_data;
  if (loom_aie2p_shuffle_is_identity(shuffle_plan)) {
    return loom_low_lower_bind_value_alias(
        context, loom_vector_shuffle_source(source_op),
        loom_vector_shuffle_result(source_op));
  }

  loom_aie2p_shuffle_emit_state_t state;
  IREE_RETURN_IF_ERROR(loom_aie2p_shuffle_emit_state_initialize(
      context, source_op, shuffle_plan, &state));
  loom_value_id_t result_packets[LOOM_AIE2P_SHUFFLE_MAX_PACKET_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint8_t packet = 0; packet < shuffle_plan->logical_packet_count;
       ++packet) {
    const uint8_t source_packet =
        loom_aie2p_shuffle_packet_source_alias(shuffle_plan, packet);
    if (source_packet != LOOM_AIE2P_SHUFFLE_PACKET_GENERIC) {
      IREE_RETURN_IF_ERROR(loom_aie2p_shuffle_source_native_packet(
          &state, source_packet, &result_packets[packet]));
      continue;
    }

    loom_value_id_t vector_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_shuffle_emit_vector_packet(&state, packet, &vector_packet));
    IREE_RETURN_IF_ERROR(loom_aie2p_shuffle_vector_to_native_packet(
        &state, vector_packet, &result_packets[packet]));
  }

  const uint8_t units_per_packet =
      loom_aie2p_vector_packet_carrier_unit_count(shuffle_plan->carrier_kind);
  const uint8_t physical_packet_count =
      shuffle_plan->carrier_unit_count / units_per_packet;
  for (uint8_t packet = shuffle_plan->logical_packet_count;
       packet < physical_packet_count; ++packet) {
    result_packets[packet] =
        result_packets[shuffle_plan->logical_packet_count - 1];
  }

  loom_value_id_t result = result_packets[0];
  if (physical_packet_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        loom_low_lower_context_builder(context), result_packets,
        physical_packet_count, state.result_type, source_op->location,
        &concat_op));
    result = loom_low_concat_result(concat_op);
  }
  return loom_low_lower_bind_value(
      context, loom_vector_shuffle_result(source_op), result);
}
