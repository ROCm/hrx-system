// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/transpose.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/transpose.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/vector_packet.h"
#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"

enum {
  LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT = 64,
  LOOM_AIE2P_TRANSPOSE_MAX_LANE_COUNT = 128,
  LOOM_AIE2P_TRANSPOSE_MAX_STAGE_COUNT = 13,
};

typedef enum loom_aie2p_transpose_mechanism_e {
  LOOM_AIE2P_TRANSPOSE_MECHANISM_ALIAS = 0,
  LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_SINGLE = 1,
  LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_PAIR = 2,
  LOOM_AIE2P_TRANSPOSE_MECHANISM_COMPOSED_BYTE_4X4 = 3,
  LOOM_AIE2P_TRANSPOSE_MECHANISM_BENES = 4,
} loom_aie2p_transpose_mechanism_t;

typedef struct loom_aie2p_transpose_native_mode_t {
  // VSHUFFLE control producing the complete or low result packet.
  uint8_t low_control;
  // VSHUFFLE control producing the high result packet, or zero.
  uint8_t high_control;
  // Number of bytes in one matrix cell moved by the mode.
  uint8_t unit_byte_count;
  // Rows in each source matrix tile.
  uint8_t row_count;
  // Columns in each source matrix tile.
  uint8_t column_count;
  // Number of source packets consumed by one application of the mode.
  uint8_t source_packet_count;
} loom_aie2p_transpose_native_mode_t;

// Complete AIE2P VSHUFFLE transpose vocabulary. Single-packet modes repeat
// their smaller tiles across each 512-bit packet. Paired modes transpose one
// complete 1024-bit matrix and return its low and high halves.
static const loom_aie2p_transpose_native_mode_t kTransposeNativeModes[] = {
    {46, 0, 1, 8, 4, 1},   {47, 0, 1, 4, 8, 1},   {35, 0, 1, 8, 8, 1},
    {36, 0, 1, 16, 4, 1},  {37, 0, 1, 4, 16, 1},  {40, 0, 2, 4, 2, 1},
    {41, 0, 2, 2, 4, 1},   {39, 0, 2, 4, 4, 1},   {42, 0, 2, 8, 2, 1},
    {43, 0, 2, 2, 8, 1},   {28, 0, 2, 8, 4, 1},   {29, 0, 2, 4, 8, 1},
    {44, 0, 2, 16, 2, 1},  {45, 0, 2, 2, 16, 1},  {34, 0, 4, 4, 4, 1},
    {0, 1, 1, 64, 2, 2},   {20, 21, 1, 2, 64, 2}, {2, 3, 2, 32, 2, 2},
    {18, 19, 2, 2, 32, 2}, {24, 25, 2, 16, 4, 2}, {26, 27, 2, 4, 16, 2},
    {52, 53, 2, 8, 8, 2},  {4, 5, 4, 16, 2, 2},   {16, 17, 4, 2, 16, 2},
    {30, 31, 4, 8, 4, 2},  {32, 33, 4, 4, 8, 2},  {6, 7, 8, 8, 2, 2},
    {14, 15, 8, 2, 8, 2},  {8, 9, 16, 4, 2, 2},   {12, 13, 16, 2, 4, 2},
    {10, 11, 32, 2, 2, 2},
};

typedef struct loom_aie2p_transpose_plan_layout_t {
  // Number of physical bytes moved as one semantic lane.
  uint8_t lane_byte_count;
  // Base-two logarithm of the padded permutation lane count.
  uint8_t padded_lane_log2;
  // Number of retained Beneš switch masks.
  uint8_t stage_count;
  // Selected native or residual realization.
  uint8_t mechanism;
  // Low or single-result VSHUFFLE control.
  uint8_t low_control;
  // High-result VSHUFFLE control for paired modes.
  uint8_t high_control;
  // Logical 512-bit packets occupied by the padded permutation.
  uint8_t packet_count;
  // Physical carrier family shared by source and result.
  uint8_t carrier_kind;
  // Allocation units in each complete source and result carrier.
  uint8_t carrier_unit_count;
} loom_aie2p_transpose_plan_layout_t;

typedef struct loom_aie2p_transpose_plan_t {
  // Fixed selected layout and carrier facts.
  loom_aie2p_transpose_plan_layout_t layout;
  // Configured pair switches in stage order.
  uint64_t switch_masks[];
} loom_aie2p_transpose_plan_t;

typedef struct loom_aie2p_transpose_match_t {
  // Fixed selected layout and carrier facts.
  loom_aie2p_transpose_plan_layout_t layout;
  // Maximum temporary switch-mask storage used during selection.
  uint64_t switch_masks[LOOM_AIE2P_TRANSPOSE_MAX_STAGE_COUNT];
} loom_aie2p_transpose_match_t;

static bool loom_aie2p_transpose_admitted(const loom_module_t* module,
                                          const loom_op_t* source_op) {
  if (!loom_vector_transpose_isa(source_op)) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transpose_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_transpose_result(source_op));
  return loom_aie2p_vector_carrier_for_type(source_type).kind !=
             LOOM_AIE2P_VECTOR_CARRIER_NONE &&
         loom_aie2p_vector_carrier_for_type(result_type).kind !=
             LOOM_AIE2P_VECTOR_CARRIER_NONE;
}

static void loom_aie2p_transpose_source_lanes(const loom_module_t* module,
                                              const loom_op_t* source_op,
                                              uint16_t lane_count,
                                              uint8_t* out_source_lanes) {
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transpose_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_transpose_result(source_op));
  const loom_attribute_t permutation =
      loom_vector_transpose_permutation(source_op);
  for (uint16_t result_lane = 0; result_lane < lane_count; ++result_lane) {
    out_source_lanes[result_lane] = (uint8_t)loom_vector_transpose_source_lane(
        source_type, result_type, permutation.i64_array, result_lane);
  }
}

static bool loom_aie2p_transpose_is_identity(const uint8_t* source_lanes,
                                             uint16_t lane_count) {
  for (uint16_t lane = 0; lane < lane_count; ++lane) {
    if (source_lanes[lane] != lane) {
      return false;
    }
  }
  return true;
}

static uint16_t loom_aie2p_transpose_native_source_byte(
    const loom_aie2p_transpose_native_mode_t* mode, uint16_t result_byte) {
  const uint16_t packet_base =
      mode->source_packet_count == 1
          ? (result_byte / LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT) *
                LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT
          : 0;
  const uint16_t packet_byte = result_byte - packet_base;
  const uint16_t result_unit = packet_byte / mode->unit_byte_count;
  const uint8_t unit_byte = packet_byte % mode->unit_byte_count;
  const uint16_t tile_unit_count = mode->row_count * mode->column_count;
  const uint16_t tile_base = (result_unit / tile_unit_count) * tile_unit_count;
  const uint16_t tile_unit = result_unit % tile_unit_count;
  const uint16_t source_unit =
      (uint16_t)(tile_base +
                 (tile_unit % mode->row_count) * mode->column_count +
                 tile_unit / mode->row_count);
  return (uint16_t)(packet_base + source_unit * mode->unit_byte_count +
                    unit_byte);
}

static bool loom_aie2p_transpose_native_mode_matches(
    const loom_aie2p_transpose_native_mode_t* mode, const uint8_t* source_lanes,
    uint16_t lane_count, uint8_t lane_byte_count) {
  const uint16_t live_byte_count = lane_count * lane_byte_count;
  if (mode->source_packet_count == 2 &&
      live_byte_count <= LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT) {
    return false;
  }
  for (uint16_t result_byte = 0; result_byte < live_byte_count; ++result_byte) {
    const uint16_t result_lane = result_byte / lane_byte_count;
    const uint8_t lane_byte = result_byte % lane_byte_count;
    const uint16_t expected_source_byte =
        source_lanes[result_lane] * lane_byte_count + lane_byte;
    if (loom_aie2p_transpose_native_source_byte(mode, result_byte) !=
        expected_source_byte) {
      return false;
    }
  }
  return true;
}

static bool loom_aie2p_transpose_is_composed_byte_4x4(
    const uint8_t* source_lanes, uint16_t lane_count, uint8_t lane_byte_count) {
  static const uint8_t kSourceLanes[] = {
      0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
  };
  return lane_byte_count == 1 && lane_count == IREE_ARRAYSIZE(kSourceLanes) &&
         memcmp(source_lanes, kSourceLanes, sizeof(kSourceLanes)) == 0;
}

static uint8_t loom_aie2p_transpose_pair_ordinal(uint16_t lower_lane,
                                                 uint16_t lane_distance) {
  return (uint8_t)((lower_lane / (2u * lane_distance)) * lane_distance +
                   lower_lane % (2u * lane_distance));
}

static void loom_aie2p_transpose_set_switch(uint64_t* switch_masks,
                                            uint8_t stage, uint16_t lower_lane,
                                            uint16_t lane_distance) {
  const uint8_t pair =
      loom_aie2p_transpose_pair_ordinal(lower_lane, lane_distance);
  switch_masks[stage] |= UINT64_C(1) << pair;
}

static void loom_aie2p_transpose_route_subnetwork(
    const uint8_t* output_to_input, uint16_t lane_count, uint8_t depth,
    uint16_t low_bit_prefix, uint8_t total_stage_count,
    uint64_t* switch_masks) {
  if (lane_count == 1) {
    return;
  }
  const uint16_t lane_distance = (uint16_t)(1u << depth);
  if (lane_count == 2) {
    if (output_to_input[0] == 1) {
      loom_aie2p_transpose_set_switch(switch_masks, depth, low_bit_prefix,
                                      lane_distance);
    }
    return;
  }

  uint8_t input_edges[LOOM_AIE2P_TRANSPOSE_MAX_LANE_COUNT / 2][2] = {{0}};
  uint8_t input_edge_counts[LOOM_AIE2P_TRANSPOSE_MAX_LANE_COUNT / 2] = {0};
  for (uint16_t output = 0; output < lane_count; ++output) {
    const uint8_t input_pair = output_to_input[output] >> 1;
    input_edges[input_pair][input_edge_counts[input_pair]++] = (uint8_t)output;
  }

  uint8_t colors[LOOM_AIE2P_TRANSPOSE_MAX_LANE_COUNT];
  memset(colors, UINT8_MAX, lane_count);
  uint8_t pending[LOOM_AIE2P_TRANSPOSE_MAX_LANE_COUNT];
  for (uint16_t first_edge = 0; first_edge < lane_count; ++first_edge) {
    if (colors[first_edge] != UINT8_MAX) {
      continue;
    }
    colors[first_edge] = 0;
    uint8_t pending_count = 1;
    pending[0] = (uint8_t)first_edge;
    while (pending_count != 0) {
      const uint8_t edge = pending[--pending_count];
      const uint8_t expected_color = (uint8_t)(1u - colors[edge]);
      const uint8_t input_pair = output_to_input[edge] >> 1;
      for (uint8_t index = 0; index < 2; ++index) {
        const uint8_t other = input_edges[input_pair][index];
        if (other != edge && colors[other] == UINT8_MAX) {
          colors[other] = expected_color;
          pending[pending_count++] = other;
        }
      }
      const uint8_t other_output = edge ^ 1u;
      if (colors[other_output] == UINT8_MAX) {
        colors[other_output] = expected_color;
        pending[pending_count++] = other_output;
      }
    }
  }

  const uint8_t first_stage = depth;
  const uint8_t last_stage = (uint8_t)(total_stage_count - depth - 1u);
  for (uint8_t input_pair = 0; input_pair < lane_count / 2; ++input_pair) {
    const uint8_t edge = input_edges[input_pair][0];
    if (((output_to_input[edge] & 1u) ^ colors[edge]) != 0) {
      const uint16_t lower_lane =
          (uint16_t)(((uint16_t)2u * input_pair << depth) | low_bit_prefix);
      loom_aie2p_transpose_set_switch(switch_masks, first_stage, lower_lane,
                                      lane_distance);
    }
  }
  for (uint8_t output_pair = 0; output_pair < lane_count / 2; ++output_pair) {
    const uint8_t edge = output_pair * 2u;
    if ((colors[edge] ^ (edge & 1u)) != 0) {
      const uint16_t lower_lane =
          (uint16_t)(((uint16_t)2u * output_pair << depth) | low_bit_prefix);
      loom_aie2p_transpose_set_switch(switch_masks, last_stage, lower_lane,
                                      lane_distance);
    }
  }

  const uint8_t subnetwork_lane_count = (uint8_t)(lane_count / 2u);
  uint8_t subnetworks[2][LOOM_AIE2P_TRANSPOSE_MAX_LANE_COUNT / 2];
  for (uint16_t output = 0; output < lane_count; ++output) {
    const uint8_t color = colors[output];
    subnetworks[color][output >> 1] = output_to_input[output] >> 1;
  }
  for (uint8_t color = 0; color < 2; ++color) {
    loom_aie2p_transpose_route_subnetwork(
        subnetworks[color], subnetwork_lane_count, depth + 1u,
        (uint16_t)(low_bit_prefix | ((uint16_t)color << depth)),
        total_stage_count, switch_masks);
  }
}

static bool loom_aie2p_transpose_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_aie2p_transpose_match_t* out_match) {
  *out_match = (loom_aie2p_transpose_match_t){0};
  if (!loom_aie2p_transpose_admitted(module, source_op)) {
    return false;
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transpose_source(source_op));
  uint64_t wide_lane_count = 0;
  loom_type_static_element_count(source_type, &wide_lane_count);
  const uint16_t lane_count = (uint16_t)wide_lane_count;
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  const uint16_t lane_bit_count =
      element_type == LOOM_SCALAR_TYPE_I1
          ? 8
          : loom_aie2p_scalar_type_physical_bit_count(element_type);
  const uint8_t lane_byte_count = (uint8_t)(lane_bit_count / 8u);
  const loom_aie2p_vector_carrier_t carrier =
      loom_aie2p_vector_carrier_for_type(source_type);

  uint8_t source_lanes[LOOM_AIE2P_TRANSPOSE_MAX_LANE_COUNT] = {0};
  loom_aie2p_transpose_source_lanes(module, source_op, lane_count,
                                    source_lanes);

  uint16_t padded_lane_count = 1;
  uint8_t padded_lane_log2 = 0;
  while (padded_lane_count < lane_count) {
    padded_lane_count *= 2u;
    ++padded_lane_log2;
  }
  const uint16_t padded_byte_count = padded_lane_count * lane_byte_count;
  out_match->layout = (loom_aie2p_transpose_plan_layout_t){
      .lane_byte_count = lane_byte_count,
      .padded_lane_log2 = padded_lane_log2,
      .packet_count = (uint8_t)((padded_byte_count +
                                 LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT - 1u) /
                                LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT),
      .carrier_kind = carrier.kind,
      .carrier_unit_count = (uint8_t)carrier.unit_count,
  };

  if (loom_aie2p_transpose_is_identity(source_lanes, lane_count)) {
    out_match->layout.mechanism = LOOM_AIE2P_TRANSPOSE_MECHANISM_ALIAS;
    return true;
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kTransposeNativeModes); ++i) {
    const loom_aie2p_transpose_native_mode_t* mode = &kTransposeNativeModes[i];
    if (!loom_aie2p_transpose_native_mode_matches(
            mode, source_lanes, lane_count, lane_byte_count)) {
      continue;
    }
    out_match->layout.mechanism =
        mode->source_packet_count == 1
            ? LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_SINGLE
            : LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_PAIR;
    out_match->layout.low_control = mode->low_control;
    out_match->layout.high_control = mode->high_control;
    return true;
  }

  if (loom_aie2p_transpose_is_composed_byte_4x4(source_lanes, lane_count,
                                                lane_byte_count)) {
    out_match->layout.mechanism =
        LOOM_AIE2P_TRANSPOSE_MECHANISM_COMPOSED_BYTE_4X4;
    return true;
  }

  for (uint16_t lane = lane_count; lane < padded_lane_count; ++lane) {
    source_lanes[lane] = (uint8_t)lane;
  }
  const uint8_t stage_count =
      padded_lane_log2 == 0 ? 0 : (uint8_t)(2u * padded_lane_log2 - 1u);
  out_match->layout.mechanism = LOOM_AIE2P_TRANSPOSE_MECHANISM_BENES;
  out_match->layout.stage_count = stage_count;
  loom_aie2p_transpose_route_subnetwork(source_lanes, padded_lane_count,
                                        /*depth=*/0, /*low_bit_prefix=*/0,
                                        stage_count, out_match->switch_masks);
  return true;
}

iree_status_t loom_aie2p_query_transpose_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();
  if (environment->vector_lane_projection.source_lane_count == 0 &&
      loom_aie2p_transpose_admitted(environment->module, source_op)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_select_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  loom_aie2p_transpose_match_t match;
  if (!loom_aie2p_transpose_plan_from_op(loom_low_lower_context_module(context),
                                         source_op, &match)) {
    return iree_ok_status();
  }

  const iree_host_size_t plan_size =
      offsetof(loom_aie2p_transpose_plan_t, switch_masks) +
      match.layout.stage_count * sizeof(match.switch_masks[0]);
  loom_aie2p_transpose_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, plan_size, (void**)&retained_plan));
  retained_plan->layout = match.layout;
  memcpy(retained_plan->switch_masks, match.switch_masks,
         match.layout.stage_count * sizeof(match.switch_masks[0]));
  if (match.layout.mechanism != LOOM_AIE2P_TRANSPOSE_MECHANISM_ALIAS) {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
        context, match.layout.carrier_kind, match.layout.carrier_unit_count,
        &result_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
        context, loom_vector_transpose_result(source_op), result_type));
  }
  *out_plan =
      loom_low_lower_plan_make(LOOM_AIE2P_PLAN_TRANSPOSE_STATIC, retained_plan);
  return iree_ok_status();
}

iree_status_t loom_aie2p_finalize_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  const loom_aie2p_transpose_plan_t* transpose_plan =
      (const loom_aie2p_transpose_plan_t*)plan.target_data;
  if (transpose_plan->layout.mechanism !=
      LOOM_AIE2P_TRANSPOSE_MECHANISM_ALIAS) {
    return iree_ok_status();
  }
  return loom_low_lower_plan_value_type(
      context, loom_vector_transpose_result(source_op),
      loom_low_lower_value_binding_type(
          context, loom_vector_transpose_source(source_op)));
}

void loom_aie2p_mark_transpose_plan_demands(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t plan) {
  (void)plan;
  loom_low_lower_require_source_value_storage(
      context, loom_vector_transpose_source(source_op));
}

void loom_aie2p_describe_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  (void)context;
  (void)source_op;
  const loom_aie2p_transpose_plan_t* transpose_plan =
      (const loom_aie2p_transpose_plan_t*)plan.target_data;
  iree_string_view_t plan_key = iree_string_view_empty();
  switch (transpose_plan->layout.mechanism) {
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_ALIAS:
      plan_key = IREE_SV("transpose.alias");
      break;
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_SINGLE:
      plan_key = IREE_SV("transpose.vshuffle-single");
      break;
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_PAIR:
      plan_key = IREE_SV("transpose.vshuffle-pair");
      break;
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_COMPOSED_BYTE_4X4:
      plan_key = IREE_SV("transpose.byte-4x4");
      break;
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_BENES:
      plan_key = transpose_plan->layout.carrier_kind ==
                         LOOM_AIE2P_VECTOR_CARRIER_PREDICATE
                     ? IREE_SV("transpose.predicate-benes")
                     : IREE_SV("transpose.benes");
      break;
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P transpose mechanism");
      break;
  }
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = plan_key,
  };
}

typedef struct loom_aie2p_transpose_emit_state_t {
  // Shared packet emission helpers and lazily materialized constants.
  loom_aie2p_vector_packet_emitter_t emitter;
  // Selected transpose plan.
  const loom_aie2p_transpose_plan_t* plan;
  // Complete source carrier value.
  loom_value_id_t low_source;
  // Physical carrier shared by source and result.
  loom_aie2p_vector_carrier_t carrier;
  // Lazily materialized VSHIFT controls indexed by byte rotation.
  loom_value_id_t shift_controls[LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT];
} loom_aie2p_transpose_emit_state_t;

static iree_status_t loom_aie2p_transpose_emit_state_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_aie2p_transpose_plan_t* plan,
    loom_aie2p_transpose_emit_state_t* out_state) {
  *out_state = (loom_aie2p_transpose_emit_state_t){
      .plan = plan,
      .low_source = LOOM_VALUE_ID_INVALID,
      .carrier =
          {
              .kind = plan->layout.carrier_kind,
              .unit_count = plan->layout.carrier_unit_count,
          },
  };
  for (uint8_t i = 0; i < LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT; ++i) {
    out_state->shift_controls[i] = LOOM_VALUE_ID_INVALID;
  }
  out_state->low_source = loom_low_lower_lookup_value(
      context, loom_vector_transpose_source(source_op));
  return loom_aie2p_vector_packet_emitter_initialize(context, source_op,
                                                     &out_state->emitter);
}

static iree_status_t loom_aie2p_transpose_emit_control(
    loom_aie2p_transpose_emit_state_t* state, uint8_t value,
    loom_value_id_t* out_control) {
  return loom_aie2p_vector_packet_emit_constant(
      &state->emitter, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_MOVA, value,
      state->emitter.scalar_type, out_control);
}

static iree_status_t loom_aie2p_transpose_emit_shuffle(
    loom_aie2p_transpose_emit_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t control, loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs, control};
  return loom_aie2p_vector_packet_emit_descriptor_op(
      &state->emitter, AIE2P_CORE_DESCRIPTOR_REF_SHUFFLE_X_CONFIGURED, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      state->emitter.vector_type, /*tied_results=*/NULL,
      /*tied_result_count=*/0, out_result);
}

static iree_status_t loom_aie2p_transpose_shift_control(
    loom_aie2p_transpose_emit_state_t* state, uint8_t byte_count,
    loom_value_id_t* out_control) {
  loom_value_id_t* cached = &state->shift_controls[byte_count];
  if (*cached == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_transpose_emit_control(state, byte_count, cached));
  }
  *out_control = *cached;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_transpose_emit_shift(
    loom_aie2p_transpose_emit_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, uint8_t byte_count, loom_value_id_t* out_result) {
  loom_value_id_t control = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_transpose_shift_control(state, byte_count, &control));
  const loom_value_id_t operands[] = {lhs, rhs, control};
  return loom_aie2p_vector_packet_emit_descriptor_op(
      &state->emitter, AIE2P_CORE_DESCRIPTOR_REF_SHIFT_BYTES_X_CONFIGURED,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      state->emitter.vector_type, /*tied_results=*/NULL,
      /*tied_result_count=*/0, out_result);
}

static iree_status_t loom_aie2p_transpose_emit_select(
    loom_aie2p_transpose_emit_state_t* state, loom_value_id_t false_value,
    loom_value_id_t true_value, uint64_t byte_mask,
    loom_value_id_t* out_result) {
  loom_value_id_t selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_byte_selector(
      &state->emitter, byte_mask, &selector));
  const loom_value_id_t operands[] = {false_value, true_value, selector};
  return loom_aie2p_vector_packet_emit_descriptor_op(
      &state->emitter, AIE2P_CORE_DESCRIPTOR_REF_SELECT_I8X64, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      state->emitter.vector_type, /*tied_results=*/NULL,
      /*tied_result_count=*/0, out_result);
}

static iree_status_t loom_aie2p_transpose_read_source_packets(
    loom_aie2p_transpose_emit_state_t* state,
    loom_value_id_t* out_source_packets) {
  for (uint8_t packet = 0; packet < state->plan->layout.packet_count;
       ++packet) {
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_vector(
        &state->emitter, state->low_source, state->carrier, packet,
        &out_source_packets[packet]));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_transpose_bind_vector_packets(
    loom_aie2p_transpose_emit_state_t* state,
    const loom_value_id_t* vector_packets) {
  return loom_aie2p_vector_packet_bind_vector_packets(
      &state->emitter, state->carrier, state->plan->layout.packet_count,
      vector_packets, loom_vector_transpose_result(state->emitter.source_op));
}

static iree_status_t loom_aie2p_transpose_emit_native_single(
    loom_aie2p_transpose_emit_state_t* state) {
  loom_value_id_t source_packets[2] = {LOOM_VALUE_ID_INVALID,
                                       LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_aie2p_transpose_read_source_packets(state, source_packets));
  loom_value_id_t control = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_control(
      state, state->plan->layout.low_control, &control));
  loom_value_id_t result_packets[2] = {LOOM_VALUE_ID_INVALID,
                                       LOOM_VALUE_ID_INVALID};
  for (uint8_t packet = 0; packet < state->plan->layout.packet_count;
       ++packet) {
    IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shuffle(
        state, source_packets[packet], source_packets[packet], control,
        &result_packets[packet]));
  }
  return loom_aie2p_transpose_bind_vector_packets(state, result_packets);
}

static iree_status_t loom_aie2p_transpose_emit_native_pair(
    loom_aie2p_transpose_emit_state_t* state) {
  loom_value_id_t source_packets[2] = {LOOM_VALUE_ID_INVALID,
                                       LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_aie2p_transpose_read_source_packets(state, source_packets));
  loom_value_id_t result_packets[2] = {LOOM_VALUE_ID_INVALID,
                                       LOOM_VALUE_ID_INVALID};
  const uint8_t controls[] = {state->plan->layout.low_control,
                              state->plan->layout.high_control};
  for (uint8_t packet = 0; packet < 2; ++packet) {
    loom_value_id_t control = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_transpose_emit_control(state, controls[packet], &control));
    IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shuffle(
        state, source_packets[0], source_packets[1], control,
        &result_packets[packet]));
  }
  return loom_aie2p_transpose_bind_vector_packets(state, result_packets);
}

static iree_status_t loom_aie2p_transpose_emit_composed_byte_4x4(
    loom_aie2p_transpose_emit_state_t* state) {
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_vector(
      &state->emitter, state->low_source, state->carrier, /*packet_index=*/0,
      &source));
  loom_value_id_t controls[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (uint8_t control = 0; control < 2; ++control) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_transpose_emit_control(state, control, &controls[control]));
  }

  loom_value_id_t packed[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (uint8_t index = 0; index < 2; ++index) {
    IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shuffle(
        state, source, source, controls[index], &packed[index]));
  }
  loom_value_id_t columns[4] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                                LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  const uint8_t packed_indices[] = {0, 0, 1, 1};
  const uint8_t control_indices[] = {0, 1, 0, 1};
  const uint8_t column_indices[] = {0, 2, 1, 3};
  for (uint8_t i = 0; i < 4; ++i) {
    const loom_value_id_t value = packed[packed_indices[i]];
    IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shuffle(
        state, value, value, controls[control_indices[i]],
        &columns[column_indices[i]]));
  }

  loom_value_id_t column_pairs[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  for (uint8_t pair = 0; pair < 2; ++pair) {
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shift(
        state, columns[pair * 2], columns[pair * 2], 4, &shifted));
    IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shift(
        state, shifted, columns[pair * 2 + 1], 60, &column_pairs[pair]));
  }
  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shift(
      state, column_pairs[0], column_pairs[0], 8, &shifted));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shift(
      state, shifted, column_pairs[1], 56, &result));
  return loom_aie2p_transpose_bind_vector_packets(state, &result);
}

static uint64_t loom_aie2p_transpose_lane_byte_mask(uint16_t lane,
                                                    uint8_t lane_byte_count) {
  const uint8_t packet_byte = (uint8_t)((lane * lane_byte_count) %
                                        LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT);
  const uint64_t lane_mask = (UINT64_C(1) << lane_byte_count) - UINT64_C(1);
  return lane_mask << packet_byte;
}

static void loom_aie2p_transpose_stage_packet_masks(
    const loom_aie2p_transpose_plan_t* plan, uint64_t switch_mask,
    uint16_t lane_distance, uint8_t packet, uint64_t* out_lower_mask,
    uint64_t* out_upper_mask) {
  *out_lower_mask = 0;
  *out_upper_mask = 0;
  const uint16_t padded_lane_count =
      (uint16_t)(1u << plan->layout.padded_lane_log2);
  for (uint8_t pair = 0; pair < padded_lane_count / 2u; ++pair) {
    if ((switch_mask & (UINT64_C(1) << pair)) == 0) {
      continue;
    }
    const uint16_t block = pair / lane_distance;
    const uint16_t offset = pair % lane_distance;
    const uint16_t lower_lane = block * 2u * lane_distance + offset;
    const uint16_t upper_lane = lower_lane + lane_distance;
    const uint8_t lower_packet =
        (uint8_t)(lower_lane * plan->layout.lane_byte_count /
                  LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT);
    if (lower_packet != packet) {
      continue;
    }
    *out_lower_mask |= loom_aie2p_transpose_lane_byte_mask(
        lower_lane, plan->layout.lane_byte_count);
    *out_upper_mask |= loom_aie2p_transpose_lane_byte_mask(
        upper_lane, plan->layout.lane_byte_count);
  }
}

static iree_status_t loom_aie2p_transpose_emit_packet_stage(
    loom_aie2p_transpose_emit_state_t* state, loom_value_id_t source,
    uint8_t byte_distance, uint64_t lower_mask, uint64_t upper_mask,
    loom_value_id_t* out_result) {
  loom_value_id_t forward = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shift(
      state, source, source, byte_distance, &forward));
  if (byte_distance == 32) {
    return loom_aie2p_transpose_emit_select(
        state, source, forward, lower_mask | upper_mask, out_result);
  }

  loom_value_id_t backward = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_shift(
      state, source, source,
      LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT - byte_distance, &backward));
  loom_value_id_t lower_selected = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_select(
      state, source, forward, lower_mask, &lower_selected));
  return loom_aie2p_transpose_emit_select(state, lower_selected, backward,
                                          upper_mask, out_result);
}

static iree_status_t loom_aie2p_transpose_emit_cross_packet_stage(
    loom_aie2p_transpose_emit_state_t* state, uint64_t switch_mask,
    loom_value_id_t* packets) {
  uint64_t byte_mask = 0;
  const uint16_t lane_distance = LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT /
                                 state->plan->layout.lane_byte_count;
  for (uint8_t pair = 0; pair < lane_distance; ++pair) {
    if ((switch_mask & (UINT64_C(1) << pair)) != 0) {
      byte_mask |= loom_aie2p_transpose_lane_byte_mask(
          pair, state->plan->layout.lane_byte_count);
    }
  }
  const loom_value_id_t source_packets[] = {packets[0], packets[1]};
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_select(
      state, source_packets[0], source_packets[1], byte_mask, &packets[0]));
  return loom_aie2p_transpose_emit_select(
      state, source_packets[1], source_packets[0], byte_mask, &packets[1]);
}

static iree_status_t loom_aie2p_transpose_emit_benes(
    loom_aie2p_transpose_emit_state_t* state) {
  loom_value_id_t packets[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_aie2p_transpose_read_source_packets(state, packets));
  const uint8_t log_lane_count = state->plan->layout.padded_lane_log2;
  for (uint8_t stage = 0; stage < state->plan->layout.stage_count; ++stage) {
    const uint64_t switch_mask = state->plan->switch_masks[stage];
    if (switch_mask == 0) {
      continue;
    }
    const uint8_t distance_log2 =
        stage < log_lane_count ? stage
                               : (uint8_t)(2u * log_lane_count - stage - 2u);
    const uint16_t lane_distance = (uint16_t)(1u << distance_log2);
    const uint8_t byte_distance =
        (uint8_t)(lane_distance * state->plan->layout.lane_byte_count);
    if (byte_distance == LOOM_AIE2P_TRANSPOSE_PACKET_BYTE_COUNT) {
      IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_cross_packet_stage(
          state, switch_mask, packets));
      continue;
    }

    loom_value_id_t next_packets[2] = {packets[0], packets[1]};
    for (uint8_t packet = 0; packet < state->plan->layout.packet_count;
         ++packet) {
      uint64_t lower_mask = 0;
      uint64_t upper_mask = 0;
      loom_aie2p_transpose_stage_packet_masks(state->plan, switch_mask,
                                              lane_distance, packet,
                                              &lower_mask, &upper_mask);
      if (lower_mask == 0) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_packet_stage(
          state, packets[packet], byte_distance, lower_mask, upper_mask,
          &next_packets[packet]));
    }
    memcpy(packets, next_packets,
           state->plan->layout.packet_count * sizeof(packets[0]));
  }
  return loom_aie2p_transpose_bind_vector_packets(state, packets);
}

iree_status_t loom_aie2p_emit_transpose_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan) {
  const loom_aie2p_transpose_plan_t* transpose_plan =
      (const loom_aie2p_transpose_plan_t*)plan.target_data;
  if (transpose_plan->layout.mechanism ==
      LOOM_AIE2P_TRANSPOSE_MECHANISM_ALIAS) {
    return loom_low_lower_bind_value_alias(
        context, loom_vector_transpose_source(source_op),
        loom_vector_transpose_result(source_op));
  }

  loom_aie2p_transpose_emit_state_t state;
  IREE_RETURN_IF_ERROR(loom_aie2p_transpose_emit_state_initialize(
      context, source_op, transpose_plan, &state));
  switch (transpose_plan->layout.mechanism) {
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_SINGLE:
      return loom_aie2p_transpose_emit_native_single(&state);
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_NATIVE_PAIR:
      return loom_aie2p_transpose_emit_native_pair(&state);
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_COMPOSED_BYTE_4X4:
      return loom_aie2p_transpose_emit_composed_byte_4x4(&state);
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_BENES:
      return loom_aie2p_transpose_emit_benes(&state);
    case LOOM_AIE2P_TRANSPOSE_MECHANISM_ALIAS:
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P transpose mechanism");
      IREE_BUILTIN_UNREACHABLE();
  }
}
