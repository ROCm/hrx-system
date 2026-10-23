// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/vector_construct.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/vector_packet.h"

enum {
  LOOM_AIE2P_VECTOR_CONSTRUCT_PLAN_PUSH = 0x403,
};

typedef struct loom_aie2p_vector_construct_plan_t {
  // Number of scalar elements in the verified source segment.
  uint16_t element_count;
  // Physical scalar width shared by integer and floating payloads.
  uint8_t element_bit_count;
  // Logical elements represented by one X packet before carrier packing.
  uint8_t packet_lane_count;
  // Number of packets containing logical result elements.
  uint8_t logical_packet_count;
  // Physical representation of the complete result.
  loom_aie2p_vector_carrier_kind_t carrier_kind;
  // Allocation units in the result's physical carrier.
  uint8_t carrier_unit_count;
} loom_aie2p_vector_construct_plan_t;

static bool loom_aie2p_vector_construct_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_aie2p_vector_construct_plan_t* out_plan) {
  if (!loom_vector_from_elements_isa(source_op)) {
    return false;
  }
  const loom_type_t result_type = loom_module_value_type(
      module, loom_vector_from_elements_result(source_op));
  const loom_aie2p_vector_carrier_t carrier =
      loom_aie2p_vector_carrier_for_type(result_type);
  if (carrier.kind == LOOM_AIE2P_VECTOR_CARRIER_NONE) {
    return false;
  }
  const uint16_t element_bit_count = loom_aie2p_scalar_type_physical_bit_count(
      loom_type_element_type(result_type));
  const uint8_t packet_lane_count =
      loom_aie2p_vector_packet_scalar_lane_count(element_bit_count);
  if (packet_lane_count == 0) {
    return false;
  }
  const uint16_t element_count =
      (uint16_t)loom_vector_from_elements_elements(source_op).count;
  *out_plan = (loom_aie2p_vector_construct_plan_t){
      .element_count = element_count,
      .element_bit_count = (uint8_t)element_bit_count,
      .packet_lane_count = packet_lane_count,
      .logical_packet_count =
          (uint8_t)((element_count + packet_lane_count - 1) /
                    packet_lane_count),
      .carrier_kind = carrier.kind,
      .carrier_unit_count = (uint8_t)carrier.unit_count,
  };
  return true;
}

iree_status_t loom_aie2p_query_vector_construct_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();
  if (environment->vector_lane_projection.source_lane_count != 0) {
    return iree_ok_status();
  }
  loom_aie2p_vector_construct_plan_t plan;
  if (loom_aie2p_vector_construct_plan_from_op(environment->module, source_op,
                                               &plan)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

bool loom_aie2p_vector_construct_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AIE2P_VECTOR_CONSTRUCT_PLAN_PUSH;
}

iree_status_t loom_aie2p_select_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  loom_aie2p_vector_construct_plan_t matched_plan;
  if (!loom_aie2p_vector_construct_plan_from_op(
          loom_low_lower_context_module(context), source_op, &matched_plan)) {
    return iree_ok_status();
  }
  loom_aie2p_vector_construct_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = matched_plan;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
      context, retained_plan->carrier_kind, retained_plan->carrier_unit_count,
      &result_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_vector_from_elements_result(source_op), result_type));
  *out_plan = loom_low_lower_plan_make(LOOM_AIE2P_VECTOR_CONSTRUCT_PLAN_PUSH,
                                       retained_plan);
  return iree_ok_status();
}

void loom_aie2p_describe_vector_construct_plan(
    loom_low_lower_plan_report_t* out_report) {
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = IREE_SV("vector-construction.scalar-push"),
  };
}

iree_status_t loom_aie2p_emit_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  const loom_aie2p_vector_construct_plan_t* construction = plan.target_data;
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  loom_aie2p_vector_packet_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emitter_initialize(
      context, source_op, &emitter));
  loom_value_id_t packets[LOOM_AIE2P_VECTOR_PACKET_MAX_COUNT];
  iree_status_t status = iree_ok_status();
  for (uint8_t packet = 0;
       packet < construction->logical_packet_count && iree_status_is_ok(status);
       ++packet) {
    const uint16_t begin = packet * construction->packet_lane_count;
    const uint16_t count = (uint16_t)iree_min(
        construction->packet_lane_count, construction->element_count - begin);
    // Full packets append ascending lanes. Partial packets prepend descending
    // lanes so their logical elements occupy the carrier's low bytes.
    const bool full_packet = count == construction->packet_lane_count;
    const uint16_t seed_index = full_packet ? begin : begin + count - 1;
    const loom_value_id_t seed =
        loom_low_lower_lookup_value(context, elements.values[seed_index]);
    loom_value_id_t result = LOOM_VALUE_ID_INVALID;
    status = loom_aie2p_vector_packet_emit_scalar_splat(
        &emitter, construction->element_bit_count, seed, &result);
    for (uint16_t lane = 1; lane < count && iree_status_is_ok(status); ++lane) {
      const uint16_t element_index =
          full_packet ? begin + lane : begin + count - lane - 1;
      const loom_value_id_t scalar =
          loom_low_lower_lookup_value(context, elements.values[element_index]);
      status = loom_aie2p_vector_packet_emit_scalar_push(
          &emitter, construction->element_bit_count, full_packet, result,
          scalar, &result);
    }
    packets[packet] = result;
  }
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_vector_packet_bind_vector_packets(
        &emitter,
        (loom_aie2p_vector_carrier_t){
            .kind = construction->carrier_kind,
            .unit_count = construction->carrier_unit_count,
        },
        construction->logical_packet_count, packets,
        loom_vector_from_elements_result(source_op));
  }
  return status;
}
