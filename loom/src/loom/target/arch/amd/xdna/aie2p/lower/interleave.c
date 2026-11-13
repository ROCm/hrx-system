// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/interleave.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/interleave.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/vector_packet.h"
#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"

enum {
  LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT = 4,
  LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT = 64,
};

typedef enum loom_aie2p_interleave_mechanism_e {
  LOOM_AIE2P_INTERLEAVE_MECHANISM_VSHUFFLE = 0,
  LOOM_AIE2P_INTERLEAVE_MECHANISM_BLOCK_ROUTE = 1,
} loom_aie2p_interleave_mechanism_t;

typedef struct loom_aie2p_interleave_plan_t {
  // Shared semantic byte route and packet layout.
  loom_vector_interleave_packet_plan_t route;
  // Packet realization selected for the semantic block size.
  uint8_t mechanism;
  // Low VSHUFFLE control; the high result uses the next control.
  uint8_t low_control;
  // Physical carrier family of each source value.
  uint8_t source_carrier_kind;
  // Physical allocation units in each complete source carrier.
  uint8_t source_carrier_unit_count;
  // Physical carrier family of each result value.
  uint8_t result_carrier_kind;
  // Physical allocation units in each complete result carrier.
  uint8_t result_carrier_unit_count;
} loom_aie2p_interleave_plan_t;
static_assert(sizeof(loom_aie2p_interleave_plan_t) == 18,
              "AIE2P interleave plans must stay cache dense");

static bool loom_aie2p_interleave_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_aie2p_interleave_plan_t* out_plan) {
  *out_plan = (loom_aie2p_interleave_plan_t){0};

  loom_type_t source_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  loom_vector_interleave_kind_t kind = LOOM_VECTOR_INTERLEAVE_KIND_ZIP;
  int64_t axis = 0;
  if (loom_vector_interleave_isa(source_op)) {
    source_type =
        loom_module_value_type(module, loom_vector_interleave_even(source_op));
    const loom_type_t odd_type =
        loom_module_value_type(module, loom_vector_interleave_odd(source_op));
    result_type = loom_module_value_type(
        module, loom_vector_interleave_result(source_op));
    if (!loom_type_equal(source_type, odd_type)) {
      return false;
    }
    axis = loom_vector_interleave_axis(source_op);
  } else if (loom_vector_deinterleave_isa(source_op)) {
    source_type = loom_module_value_type(
        module, loom_vector_deinterleave_source(source_op));
    result_type = loom_module_value_type(
        module, loom_vector_deinterleave_even(source_op));
    const loom_type_t odd_type =
        loom_module_value_type(module, loom_vector_deinterleave_odd(source_op));
    if (!loom_type_equal(result_type, odd_type)) {
      return false;
    }
    kind = LOOM_VECTOR_INTERLEAVE_KIND_UNZIP;
    axis = loom_vector_deinterleave_axis(source_op);
  } else {
    return false;
  }

  const loom_type_t half_type =
      kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP ? source_type : result_type;
  const loom_type_t combined_type =
      kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP ? result_type : source_type;
  const loom_scalar_type_t element_type = loom_type_element_type(half_type);
  const uint16_t physical_bit_count =
      element_type == LOOM_SCALAR_TYPE_I1
          ? 8
          : loom_aie2p_scalar_type_physical_bit_count(element_type);
  loom_vector_interleave_packet_plan_t route;
  if (!loom_vector_interleave_packet_plan_initialize(
          kind, half_type, combined_type, axis, physical_bit_count,
          LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT,
          LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT, &route)) {
    return false;
  }

  const loom_aie2p_vector_carrier_t source_carrier =
      loom_aie2p_vector_carrier_for_type(source_type);
  const loom_aie2p_vector_carrier_t result_carrier =
      loom_aie2p_vector_carrier_for_type(result_type);
  if (source_carrier.kind == LOOM_AIE2P_VECTOR_CARRIER_NONE ||
      result_carrier.kind == LOOM_AIE2P_VECTOR_CARRIER_NONE ||
      source_carrier.unit_count > UINT8_MAX ||
      result_carrier.unit_count > UINT8_MAX) {
    return false;
  }

  const uint32_t chunk_bit_count = route.chunk_byte_count * 8u;
  const bool has_vshuffle = chunk_bit_count <= 256u &&
                            (chunk_bit_count & (chunk_bit_count - 1u)) == 0;
  uint8_t chunk_byte_log2 = 0;
  for (uint16_t remaining_bytes = route.chunk_byte_count; remaining_bytes > 1;
       remaining_bytes >>= 1) {
    ++chunk_byte_log2;
  }
  const uint8_t low_control = !has_vshuffle ? 0
                              : kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP
                                  ? (uint8_t)(20u - 2u * chunk_byte_log2)
                                  : (uint8_t)(2u * chunk_byte_log2);

  *out_plan = (loom_aie2p_interleave_plan_t){
      .route = route,
      .mechanism = has_vshuffle ? LOOM_AIE2P_INTERLEAVE_MECHANISM_VSHUFFLE
                                : LOOM_AIE2P_INTERLEAVE_MECHANISM_BLOCK_ROUTE,
      .low_control = low_control,
      .source_carrier_kind = source_carrier.kind,
      .source_carrier_unit_count = (uint8_t)source_carrier.unit_count,
      .result_carrier_kind = result_carrier.kind,
      .result_carrier_unit_count = (uint8_t)result_carrier.unit_count,
  };
  return true;
}

iree_status_t loom_aie2p_query_interleave_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();
  if (environment->vector_lane_projection.source_lane_count != 0) {
    return iree_ok_status();
  }
  loom_aie2p_interleave_plan_t plan = {0};
  if (loom_aie2p_interleave_plan_from_op(environment->module, source_op,
                                         &plan)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_select_interleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  loom_aie2p_interleave_plan_t matched_plan = {0};
  if (!loom_aie2p_interleave_plan_from_op(
          loom_low_lower_context_module(context), source_op, &matched_plan)) {
    return iree_ok_status();
  }

  loom_aie2p_interleave_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = matched_plan;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
      context, retained_plan->result_carrier_kind,
      retained_plan->result_carrier_unit_count, &result_type));
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
        context, loom_op_const_results(source_op)[i], result_type));
  }
  *out_plan = loom_low_lower_plan_make(LOOM_AIE2P_PLAN_INTERLEAVE_NATIVE,
                                       retained_plan);
  return iree_ok_status();
}

void loom_aie2p_mark_interleave_plan_demands(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan) {
  (void)plan;
  if (loom_vector_interleave_isa(source_op)) {
    loom_low_lower_require_source_value_storage(
        context, loom_vector_interleave_even(source_op));
    loom_low_lower_require_source_value_storage(
        context, loom_vector_interleave_odd(source_op));
  } else {
    loom_low_lower_require_source_value_storage(
        context, loom_vector_deinterleave_source(source_op));
  }
}

void loom_aie2p_describe_interleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  (void)context;
  (void)source_op;
  const loom_aie2p_interleave_plan_t* interleave_plan =
      (const loom_aie2p_interleave_plan_t*)plan.target_data;
  const bool is_predicate = interleave_plan->source_carrier_kind ==
                                LOOM_AIE2P_VECTOR_CARRIER_PREDICATE ||
                            interleave_plan->result_carrier_kind ==
                                LOOM_AIE2P_VECTOR_CARRIER_PREDICATE;
  const bool uses_block_route =
      interleave_plan->mechanism == LOOM_AIE2P_INTERLEAVE_MECHANISM_BLOCK_ROUTE;
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key =
          interleave_plan->route.kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP
              ? (uses_block_route
                     ? (is_predicate
                            ? IREE_SV("interleave.predicate-block-route")
                            : IREE_SV("interleave.block-route"))
                     : (is_predicate ? IREE_SV("interleave.predicate-vshuffle")
                                     : IREE_SV("interleave.vshuffle")))
              : (uses_block_route
                     ? (is_predicate
                            ? IREE_SV("deinterleave.predicate-block-route")
                            : IREE_SV("deinterleave.block-route"))
                     : (is_predicate
                            ? IREE_SV("deinterleave.predicate-vshuffle")
                            : IREE_SV("deinterleave.vshuffle"))),
  };
}

static iree_status_t loom_aie2p_interleave_emit_control(
    loom_aie2p_vector_packet_emitter_t* emitter, uint8_t value,
    loom_value_id_t* out_control) {
  return loom_aie2p_vector_packet_emit_constant(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_MOVA, value,
      emitter->scalar_type, out_control);
}

static iree_status_t loom_aie2p_interleave_emit_shuffle(
    loom_aie2p_vector_packet_emitter_t* emitter, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t control, loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs, control};
  return loom_aie2p_vector_packet_emit_descriptor_op(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_SHUFFLE_X_CONFIGURED, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      emitter->vector_type, /*tied_results=*/NULL,
      /*tied_result_count=*/0, out_result);
}

typedef struct loom_aie2p_interleave_emitted_packet_t {
  // Emitted native or X-carrier packet.
  loom_value_id_t value;
  // Whether |value| already has the result carrier's native kind.
  bool is_native;
} loom_aie2p_interleave_emitted_packet_t;

typedef struct loom_aie2p_interleave_route_state_t {
  // Shared packet emission helpers and lazily materialized constants.
  loom_aie2p_vector_packet_emitter_t* emitter;
  // Selected static interleave layout.
  const loom_aie2p_interleave_plan_t* plan;
  // Complete Low carrier values supplying this operation.
  loom_value_id_t low_sources[2];
  // Physical carrier layout shared by the source values.
  loom_aie2p_vector_carrier_t source_carrier;
  // Lazily extracted native packets indexed by source and packet.
  loom_value_id_t source_native_packets[2]
                                       [LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT];
  // Lazily converted X packets indexed by source and packet.
  loom_value_id_t source_vector_packets[2]
                                       [LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT];
  // Lazily materialized VSHIFT controls indexed by byte rotation.
  loom_value_id_t shift_controls[LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT];
} loom_aie2p_interleave_route_state_t;

typedef struct loom_aie2p_interleave_route_packet_t {
  // Deinterleave result ordinal, or zero for interleave.
  uint8_t result_index;
  // Logical packet ordinal within the result value.
  uint8_t result_packet;
  // Next result byte awaiting a routed source segment.
  uint8_t result_packet_byte;
  // Number of meaningful bytes in this result packet.
  uint8_t live_byte_count;
  // Packet value accumulated from routed source segments.
  loom_aie2p_interleave_emitted_packet_t emitted;
} loom_aie2p_interleave_route_packet_t;

static void loom_aie2p_interleave_route_state_initialize(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_aie2p_interleave_plan_t* plan, loom_value_id_t low_source_0,
    loom_value_id_t low_source_1,
    loom_aie2p_interleave_route_state_t* out_state) {
  *out_state = (loom_aie2p_interleave_route_state_t){
      .emitter = emitter,
      .plan = plan,
      .low_sources = {low_source_0, low_source_1},
      .source_carrier =
          {
              .kind = plan->source_carrier_kind,
              .unit_count = plan->source_carrier_unit_count,
          },
  };
  for (uint8_t source_index = 0; source_index < 2; ++source_index) {
    for (uint8_t packet = 0; packet < LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT;
         ++packet) {
      out_state->source_native_packets[source_index][packet] =
          LOOM_VALUE_ID_INVALID;
      out_state->source_vector_packets[source_index][packet] =
          LOOM_VALUE_ID_INVALID;
    }
  }
  for (uint8_t shift = 0; shift < LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT;
       ++shift) {
    out_state->shift_controls[shift] = LOOM_VALUE_ID_INVALID;
  }
}

static iree_status_t loom_aie2p_interleave_route_source_native_packet(
    loom_aie2p_interleave_route_state_t* state, uint8_t source_index,
    uint8_t packet, loom_value_id_t* out_packet) {
  loom_value_id_t* cached = &state->source_native_packets[source_index][packet];
  if (*cached == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_native(
        state->emitter, state->low_sources[source_index], state->source_carrier,
        packet, cached));
  }
  *out_packet = *cached;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_interleave_route_source_vector_packet(
    loom_aie2p_interleave_route_state_t* state, uint8_t source_index,
    uint8_t packet, loom_value_id_t* out_packet) {
  loom_value_id_t* cached = &state->source_vector_packets[source_index][packet];
  if (*cached == LOOM_VALUE_ID_INVALID) {
    loom_value_id_t native_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_interleave_route_source_native_packet(
        state, source_index, packet, &native_packet));
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_native_to_vector(
        state->emitter, state->source_carrier.kind, native_packet, cached));
  }
  *out_packet = *cached;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_interleave_route_shift_control(
    loom_aie2p_interleave_route_state_t* state, uint8_t shift,
    loom_value_id_t* out_control) {
  loom_value_id_t* cached = &state->shift_controls[shift];
  if (*cached == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_interleave_emit_control(state->emitter, shift, cached));
  }
  *out_control = *cached;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_interleave_route_aligned_source(
    loom_aie2p_interleave_route_state_t* state, uint8_t source_index,
    uint16_t source_byte, uint8_t result_packet_byte,
    loom_value_id_t* out_aligned_source) {
  const uint8_t source_packet =
      source_byte / LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT;
  const uint8_t source_packet_byte =
      source_byte % LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT;
  loom_value_id_t source_vector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_interleave_route_source_vector_packet(
      state, source_index, source_packet, &source_vector));
  const uint8_t shift =
      (source_packet_byte + LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT -
       result_packet_byte) %
      LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT;
  if (shift == 0) {
    *out_aligned_source = source_vector;
    return iree_ok_status();
  }

  loom_value_id_t control = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_interleave_route_shift_control(state, shift, &control));
  const loom_value_id_t operands[] = {source_vector, source_vector, control};
  return loom_aie2p_vector_packet_emit_descriptor_op(
      state->emitter, AIE2P_CORE_DESCRIPTOR_REF_SHIFT_BYTES_X_CONFIGURED,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      state->emitter->vector_type, /*tied_results=*/NULL,
      /*tied_result_count=*/0, out_aligned_source);
}

static uint64_t loom_aie2p_interleave_byte_mask(uint8_t start, uint8_t length) {
  if (length == 64) {
    return UINT64_MAX;
  }
  return ((UINT64_C(1) << length) - 1u) << start;
}

static loom_aie2p_interleave_route_packet_t
loom_aie2p_interleave_route_packet_initialize(
    const loom_aie2p_interleave_plan_t* plan, uint8_t result_index,
    uint8_t result_packet) {
  return (loom_aie2p_interleave_route_packet_t){
      .result_index = result_index,
      .result_packet = result_packet,
      .live_byte_count =
          (uint8_t)loom_vector_interleave_packet_plan_result_live_byte_count(
              &plan->route, result_index, result_packet),
      .emitted =
          {
              .value = LOOM_VALUE_ID_INVALID,
          },
  };
}

static bool loom_aie2p_interleave_route_packet_is_complete(
    const loom_aie2p_interleave_route_packet_t* packet) {
  return packet->result_packet_byte == packet->live_byte_count;
}

static iree_status_t loom_aie2p_interleave_route_packet_segment(
    loom_aie2p_interleave_route_state_t* state,
    loom_aie2p_interleave_route_packet_t* packet) {
  const uint8_t result_packet_byte = packet->result_packet_byte;
  uint16_t cursor = result_packet_byte;
  loom_vector_interleave_packet_segment_t segment;
  const bool has_segment = loom_vector_interleave_packet_plan_next_segment(
      &state->plan->route, packet->result_index, packet->result_packet, &cursor,
      &segment);
  IREE_ASSERT(has_segment);
  const uint16_t source_byte =
      segment.source_packet * LOOM_AIE2P_INTERLEAVE_PACKET_BYTE_COUNT +
      segment.source_packet_byte_offset;

  if (result_packet_byte == 0 &&
      segment.byte_count == packet->live_byte_count &&
      segment.source_packet_byte_offset == 0 &&
      state->plan->source_carrier_kind == state->plan->result_carrier_kind) {
    IREE_RETURN_IF_ERROR(loom_aie2p_interleave_route_source_native_packet(
        state, segment.source_index, segment.source_packet,
        &packet->emitted.value));
    packet->emitted.is_native = true;
    packet->result_packet_byte = (uint8_t)cursor;
    return iree_ok_status();
  }

  loom_value_id_t aligned_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_interleave_route_aligned_source(
      state, segment.source_index, source_byte, result_packet_byte,
      &aligned_source));
  if (packet->emitted.value == LOOM_VALUE_ID_INVALID) {
    packet->emitted.value = aligned_source;
  } else {
    const uint64_t mask = loom_aie2p_interleave_byte_mask(
        result_packet_byte, (uint8_t)segment.byte_count);
    loom_value_id_t selector = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_byte_selector(
        state->emitter, mask, &selector));
    const loom_value_id_t select_operands[] = {
        packet->emitted.value,
        aligned_source,
        selector,
    };
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_descriptor_op(
        state->emitter, AIE2P_CORE_DESCRIPTOR_REF_SELECT_I8X64, select_operands,
        IREE_ARRAYSIZE(select_operands), loom_named_attr_slice_empty(),
        state->emitter->vector_type,
        /*tied_results=*/NULL, /*tied_result_count=*/0,
        &packet->emitted.value));
  }
  packet->result_packet_byte = (uint8_t)cursor;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_interleave_route_result_packet(
    loom_aie2p_interleave_route_state_t* state, uint8_t result_index,
    uint8_t result_packet, loom_aie2p_interleave_emitted_packet_t* out_packet) {
  loom_aie2p_interleave_route_packet_t packet =
      loom_aie2p_interleave_route_packet_initialize(state->plan, result_index,
                                                    result_packet);
  while (!loom_aie2p_interleave_route_packet_is_complete(&packet)) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_interleave_route_packet_segment(state, &packet));
  }
  *out_packet = packet.emitted;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_interleave_bind_routed_packets(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_aie2p_interleave_plan_t* plan,
    loom_aie2p_interleave_emitted_packet_t* packets,
    loom_value_id_t result_value) {
  loom_value_id_t native_packets[LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint8_t packet = 0; packet < plan->route.result_packet_count; ++packet) {
    if (packets[packet].is_native) {
      native_packets[packet] = packets[packet].value;
    } else {
      IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_write_native(
          emitter, plan->result_carrier_kind, packets[packet].value,
          &native_packets[packet]));
    }
  }
  const loom_aie2p_vector_carrier_t result_carrier = {
      .kind = plan->result_carrier_kind,
      .unit_count = plan->result_carrier_unit_count,
  };
  return loom_aie2p_vector_packet_bind_native_packets(
      emitter, result_carrier, plan->route.result_packet_count, native_packets,
      result_value);
}

static iree_status_t loom_aie2p_emit_zip_plan(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_aie2p_interleave_plan_t* plan, const loom_op_t* source_op) {
  loom_value_id_t low_even = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_odd = LOOM_VALUE_ID_INVALID;
  low_even = loom_low_lower_lookup_value(
      emitter->context, loom_vector_interleave_even(source_op));
  low_odd = loom_low_lower_lookup_value(emitter->context,
                                        loom_vector_interleave_odd(source_op));

  loom_value_id_t controls[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_aie2p_interleave_emit_control(
      emitter, plan->low_control, &controls[0]));
  if (plan->route.result_packet_count > 1) {
    IREE_RETURN_IF_ERROR(loom_aie2p_interleave_emit_control(
        emitter, plan->low_control + 1u, &controls[1]));
  }

  const loom_aie2p_vector_carrier_t source_carrier = {
      .kind = plan->source_carrier_kind,
      .unit_count = plan->source_carrier_unit_count,
  };
  loom_value_id_t result_packets[LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint8_t result_packet = 0;
  for (uint8_t source_packet = 0;
       source_packet < plan->route.source_packet_count; ++source_packet) {
    loom_value_id_t even_packet = LOOM_VALUE_ID_INVALID;
    loom_value_id_t odd_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_vector(
        emitter, low_even, source_carrier, source_packet, &even_packet));
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_vector(
        emitter, low_odd, source_carrier, source_packet, &odd_packet));
    for (uint8_t half = 0;
         half < 2 && result_packet < plan->route.result_packet_count;
         ++half, ++result_packet) {
      IREE_RETURN_IF_ERROR(loom_aie2p_interleave_emit_shuffle(
          emitter, even_packet, odd_packet, controls[half],
          &result_packets[result_packet]));
    }
  }

  const loom_aie2p_vector_carrier_t result_carrier = {
      .kind = plan->result_carrier_kind,
      .unit_count = plan->result_carrier_unit_count,
  };
  return loom_aie2p_vector_packet_bind_vector_packets(
      emitter, result_carrier, plan->route.result_packet_count, result_packets,
      loom_vector_interleave_result(source_op));
}

static iree_status_t loom_aie2p_emit_unzip_plan(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_aie2p_interleave_plan_t* plan, const loom_op_t* source_op) {
  loom_value_id_t low_source = loom_low_lower_lookup_value(
      emitter->context, loom_vector_deinterleave_source(source_op));

  loom_value_id_t controls[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_aie2p_interleave_emit_control(
      emitter, plan->low_control, &controls[0]));
  IREE_RETURN_IF_ERROR(loom_aie2p_interleave_emit_control(
      emitter, plan->low_control + 1u, &controls[1]));

  const loom_aie2p_vector_carrier_t source_carrier = {
      .kind = plan->source_carrier_kind,
      .unit_count = plan->source_carrier_unit_count,
  };
  loom_value_id_t result_packets[2][LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT] = {
      {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
       LOOM_VALUE_ID_INVALID},
      {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
       LOOM_VALUE_ID_INVALID},
  };
  for (uint8_t result_packet = 0;
       result_packet < plan->route.result_packet_count; ++result_packet) {
    const uint8_t low_source_packet = result_packet * 2u;
    loom_value_id_t low_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_vector(
        emitter, low_source, source_carrier, low_source_packet, &low_packet));
    loom_value_id_t high_packet = low_packet;
    if (low_source_packet + 1u < plan->route.source_packet_count) {
      IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_vector(
          emitter, low_source, source_carrier, low_source_packet + 1u,
          &high_packet));
    }
    for (uint8_t result_index = 0; result_index < 2; ++result_index) {
      IREE_RETURN_IF_ERROR(loom_aie2p_interleave_emit_shuffle(
          emitter, low_packet, high_packet, controls[result_index],
          &result_packets[result_index][result_packet]));
    }
  }

  const loom_value_id_t source_results[2] = {
      loom_vector_deinterleave_even(source_op),
      loom_vector_deinterleave_odd(source_op),
  };
  const loom_aie2p_vector_carrier_t result_carrier = {
      .kind = plan->result_carrier_kind,
      .unit_count = plan->result_carrier_unit_count,
  };
  for (uint8_t result_index = 0; result_index < 2; ++result_index) {
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_bind_vector_packets(
        emitter, result_carrier, plan->route.result_packet_count,
        result_packets[result_index], source_results[result_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_emit_zip_block_route_plan(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_aie2p_interleave_plan_t* plan, const loom_op_t* source_op) {
  loom_value_id_t low_even = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_odd = LOOM_VALUE_ID_INVALID;
  low_even = loom_low_lower_lookup_value(
      emitter->context, loom_vector_interleave_even(source_op));
  low_odd = loom_low_lower_lookup_value(emitter->context,
                                        loom_vector_interleave_odd(source_op));

  loom_aie2p_interleave_route_state_t state;
  loom_aie2p_interleave_route_state_initialize(emitter, plan, low_even, low_odd,
                                               &state);
  loom_aie2p_interleave_emitted_packet_t
      result_packets[LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT];
  for (uint8_t packet = 0; packet < plan->route.result_packet_count; ++packet) {
    IREE_RETURN_IF_ERROR(loom_aie2p_interleave_route_result_packet(
        &state, /*result_index=*/0, packet, &result_packets[packet]));
  }
  return loom_aie2p_interleave_bind_routed_packets(
      emitter, plan, result_packets, loom_vector_interleave_result(source_op));
}

static iree_status_t loom_aie2p_emit_unzip_block_route_plan(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_aie2p_interleave_plan_t* plan, const loom_op_t* source_op) {
  loom_value_id_t low_source = loom_low_lower_lookup_value(
      emitter->context, loom_vector_deinterleave_source(source_op));

  loom_aie2p_interleave_route_state_t state;
  loom_aie2p_interleave_route_state_initialize(emitter, plan, low_source,
                                               LOOM_VALUE_ID_INVALID, &state);
  const loom_value_id_t results[2] = {
      loom_vector_deinterleave_even(source_op),
      loom_vector_deinterleave_odd(source_op),
  };
  loom_aie2p_interleave_emitted_packet_t
      result_packets[2][LOOM_AIE2P_INTERLEAVE_MAX_PACKET_COUNT];
  for (uint8_t packet = 0; packet < plan->route.result_packet_count; ++packet) {
    loom_aie2p_interleave_route_packet_t routed_packets[2] = {
        loom_aie2p_interleave_route_packet_initialize(plan, 0, packet),
        loom_aie2p_interleave_route_packet_initialize(plan, 1, packet),
    };
    while (
        !loom_aie2p_interleave_route_packet_is_complete(&routed_packets[0]) ||
        !loom_aie2p_interleave_route_packet_is_complete(&routed_packets[1])) {
      for (uint8_t result_index = 0; result_index < 2; ++result_index) {
        if (!loom_aie2p_interleave_route_packet_is_complete(
                &routed_packets[result_index])) {
          IREE_RETURN_IF_ERROR(loom_aie2p_interleave_route_packet_segment(
              &state, &routed_packets[result_index]));
        }
      }
    }
    result_packets[0][packet] = routed_packets[0].emitted;
    result_packets[1][packet] = routed_packets[1].emitted;
  }
  for (uint8_t result_index = 0; result_index < 2; ++result_index) {
    IREE_RETURN_IF_ERROR(loom_aie2p_interleave_bind_routed_packets(
        emitter, plan, result_packets[result_index], results[result_index]));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_emit_interleave_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  const loom_aie2p_interleave_plan_t* interleave_plan =
      (const loom_aie2p_interleave_plan_t*)plan.target_data;
  loom_aie2p_vector_packet_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emitter_initialize(
      context, source_op, &emitter));
  if (interleave_plan->mechanism ==
      LOOM_AIE2P_INTERLEAVE_MECHANISM_BLOCK_ROUTE) {
    return interleave_plan->route.kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP
               ? loom_aie2p_emit_zip_block_route_plan(&emitter, interleave_plan,
                                                      source_op)
               : loom_aie2p_emit_unzip_block_route_plan(
                     &emitter, interleave_plan, source_op);
  }
  return interleave_plan->route.kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP
             ? loom_aie2p_emit_zip_plan(&emitter, interleave_plan, source_op)
             : loom_aie2p_emit_unzip_plan(&emitter, interleave_plan, source_op);
}
