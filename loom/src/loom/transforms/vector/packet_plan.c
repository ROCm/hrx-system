// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/packet_plan.h"

#include <string.h>

#include "loom/analysis/motion.h"
#include "loom/analysis/view_regions.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/combining.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/util/fact_vector_partition.h"

static uint32_t loom_vector_packet_structural_lane_limit(
    const loom_target_vector_packet_policy_t* policy,
    loom_scalar_type_t element_type) {
  for (uint8_t i = 0; i < policy->structural_lane_limit_count; ++i) {
    if (policy->structural_lane_limits[i].element_type == element_type) {
      return policy->structural_lane_limits[i].maximum_lane_count;
    }
  }
  return 0;
}

int32_t loom_vector_packet_element_bit_count(
    const loom_target_vector_packet_policy_t* policy,
    loom_scalar_type_t element_type) {
  if (element_type == LOOM_SCALAR_TYPE_INDEX && policy->index_bit_count != 0) {
    return policy->index_bit_count;
  }
  if (element_type == LOOM_SCALAR_TYPE_OFFSET &&
      policy->offset_bit_count != 0) {
    return policy->offset_bit_count;
  }
  return loom_scalar_type_bitwidth(element_type);
}

static uint32_t loom_vector_packet_lane_count_for_bit_count(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint16_t packet_bit_count) {
  const loom_scalar_type_t element_type = loom_type_element_type(vector_type);
  const int32_t element_bit_count =
      loom_vector_packet_element_bit_count(policy, element_type);
  if (element_bit_count <= 0 || packet_bit_count == 0 ||
      packet_bit_count % (uint32_t)element_bit_count != 0) {
    return 0;
  }
  uint32_t packet_lane_count = packet_bit_count / (uint32_t)element_bit_count;
  const uint32_t lane_limit =
      loom_vector_packet_structural_lane_limit(policy, element_type);
  if (lane_limit != 0) {
    packet_lane_count = iree_min(packet_lane_count, lane_limit);
  }
  return packet_lane_count;
}

bool loom_vector_packet_static_lane_count(loom_type_t vector_type,
                                          uint32_t* out_lane_count) {
  *out_lane_count = 0;
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1 ||
      !loom_type_is_all_static(vector_type)) {
    return false;
  }
  const int64_t lane_count_i64 = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count_i64 < 1 || lane_count_i64 > UINT32_MAX) {
    return false;
  }
  *out_lane_count = (uint32_t)lane_count_i64;
  return true;
}

static bool loom_vector_packet_payload_bit_count(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint32_t lane_count, uint64_t* out_bit_count) {
  *out_bit_count = 0;
  const int32_t element_bit_count = loom_vector_packet_element_bit_count(
      policy, loom_type_element_type(vector_type));
  if (element_bit_count <= 0) {
    return false;
  }
  *out_bit_count = (uint64_t)lane_count * (uint32_t)element_bit_count;
  return true;
}

static bool loom_vector_packet_select_native_chunk_lane_count(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint32_t maximum_lane_count, uint32_t* out_chunk_lane_count) {
  *out_chunk_lane_count = 0;
  uint32_t selected_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_bit_count_count; ++i) {
    const uint32_t packet_lane_count =
        loom_vector_packet_lane_count_for_bit_count(
            policy, vector_type, policy->native_bit_counts[i]);
    if (packet_lane_count == 0) {
      continue;
    }
    if (packet_lane_count <= maximum_lane_count) {
      selected_lane_count = iree_max(selected_lane_count, packet_lane_count);
    }
  }
  *out_chunk_lane_count = selected_lane_count;
  return selected_lane_count != 0;
}

bool loom_vector_packet_maximum_chunk_lane_count(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint32_t* out_chunk_lane_count) {
  *out_chunk_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_bit_count_count; ++i) {
    const uint32_t packet_lane_count =
        loom_vector_packet_lane_count_for_bit_count(
            policy, vector_type, policy->native_bit_counts[i]);
    if (packet_lane_count == 0) {
      continue;
    }
    *out_chunk_lane_count = iree_max(*out_chunk_lane_count, packet_lane_count);
  }
  return *out_chunk_lane_count != 0;
}

bool loom_vector_packet_shape_from_type(
    const loom_target_legalization_context_t* context,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* out_shape) {
  *out_shape = (loom_vector_packet_shape_t){0};
  uint32_t lane_count = 0;
  uint32_t chunk_lane_count = 0;
  if (!loom_vector_packet_static_lane_count(vector_type, &lane_count) ||
      (loom_target_legalization_has_source_vector_carrier(
           context, vector_type, LOOM_VALUE_ID_INVALID) &&
       !loom_target_vector_packet_policy_requires_packets(policy,
                                                          vector_type)) ||
      !loom_vector_packet_select_native_chunk_lane_count(
          policy, vector_type, lane_count - 1u, &chunk_lane_count)) {
    return false;
  }
  const uint32_t chunk_count = (lane_count - 1u) / chunk_lane_count + 1u;
  *out_shape = (loom_vector_packet_shape_t){
      .lane_count = lane_count,
      .chunk_lane_count = chunk_lane_count,
      .chunk_count = chunk_count,
  };
  return true;
}

static iree_status_t loom_vector_packet_component_query(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    uint32_t source_lane_count, uint32_t packet_lane_count,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_vector_lane_projection_t projection = {
      .source_lane_count = source_lane_count,
      .projected_lane_count = packet_lane_count,
  };
  return loom_target_legalization_query_contract_with_vector_lane_projection(
      context, op, projection, out_result);
}

static bool loom_vector_packet_component_has_carriers(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    uint32_t packet_lane_count) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(context->module, operands[i]);
    if (!loom_type_is_vector(type)) {
      continue;
    }
    const loom_type_t packet_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(type),
                            loom_dim_pack_static(packet_lane_count),
                            loom_type_rank(type) == 1 ? type.encoding_id : 0);
    if (!loom_target_legalization_has_source_vector_carrier(
            context, packet_type, LOOM_VALUE_ID_INVALID)) {
      return false;
    }
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(context->module, results[i]);
    if (!loom_type_is_vector(type)) {
      continue;
    }
    const loom_type_t packet_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(type),
                            loom_dim_pack_static(packet_lane_count),
                            loom_type_rank(type) == 1 ? type.encoding_id : 0);
    if (!loom_target_legalization_has_source_vector_carrier(
            context, packet_type, LOOM_VALUE_ID_INVALID)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_vector_packet_component_candidate_bits(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    uint32_t source_lane_count, uint64_t* out_candidate_bits,
    uint64_t* out_producer_closure_candidate_bits) {
  *out_candidate_bits = 0;
  *out_producer_closure_candidate_bits = 0;
  const loom_target_vector_packet_policy_t* policy =
      context->vector_packet_policy;
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t packet_lane_count = policy->native_lane_counts[i];
    if (packet_lane_count == 0 || packet_lane_count >= source_lane_count) {
      continue;
    }
    loom_target_contract_query_result_t packet_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_vector_packet_component_query(
        context, op, source_lane_count, packet_lane_count, &packet_result));
    if (packet_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL ||
        !loom_vector_packet_component_has_carriers(context, op,
                                                   packet_lane_count)) {
      continue;
    }
    loom_target_contract_projection_flags_t projection_flags =
        packet_result.projection_flags;
    const uint32_t tail_lane_count = source_lane_count % packet_lane_count;
    if (tail_lane_count != 0) {
      loom_target_contract_query_result_t tail_result =
          loom_target_contract_query_result_empty();
      IREE_RETURN_IF_ERROR(loom_vector_packet_component_query(
          context, op, source_lane_count, tail_lane_count, &tail_result));
      if (tail_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL ||
          !loom_vector_packet_component_has_carriers(context, op,
                                                     tail_lane_count)) {
        continue;
      }
      projection_flags |= tail_result.projection_flags;
    }
    *out_candidate_bits |= (uint64_t)1u << i;
    if (iree_any_bit_set(
            projection_flags,
            LOOM_TARGET_CONTRACT_PROJECTION_FLAG_ROOT_PRODUCER_CLOSURE)) {
      *out_producer_closure_candidate_bits |= (uint64_t)1u << i;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_memory_chunk_lane_count_select_impl(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint32_t maximum_lane_count, uint32_t* out_lane_count,
    bool* out_contract_handled_candidate) {
  *out_lane_count = 0;
  if (out_contract_handled_candidate != NULL) {
    *out_contract_handled_candidate = false;
  }
  uint32_t authored_lane_count = 0;
  if (!loom_vector_packet_static_lane_count(vector_type,
                                            &authored_lane_count)) {
    return iree_ok_status();
  }

  uint32_t selected_lane_count = 0;
  uint32_t policy_lane_count = 0;
  bool contract_handled_candidate = false;
  bool maximum_candidate_queried = false;
  for (uint8_t i = 0; i < policy->native_bit_count_count; ++i) {
    const uint32_t candidate_lane_count =
        loom_vector_packet_lane_count_for_bit_count(
            policy, vector_type, policy->native_bit_counts[i]);
    if (candidate_lane_count == 0 ||
        candidate_lane_count > maximum_lane_count) {
      continue;
    }
    maximum_candidate_queried |= candidate_lane_count == maximum_lane_count;
    policy_lane_count = iree_max(policy_lane_count, candidate_lane_count);
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = authored_lane_count,
        .projected_lane_count = candidate_lane_count,
    };
    loom_target_contract_query_result_t query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(
        loom_target_legalization_query_contract_with_vector_lane_projection(
            context, op, projection, &query_result));
    contract_handled_candidate |=
        query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED;
    if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      selected_lane_count = iree_max(selected_lane_count, candidate_lane_count);
    }
  }
  // Component planning may select a contract-native subregister width that is
  // intentionally absent from the target's preferred physical register bit
  // counts. Honor that exact upper bound when the target admits both its
  // carrier and projected memory operation; otherwise retain the policy-width
  // selection and custom-planner fallback above.
  if (maximum_lane_count != 0 && !maximum_candidate_queried &&
      loom_target_legalization_has_source_vector_carrier(
          context, loom_vector_packet_type(vector_type, maximum_lane_count),
          LOOM_VALUE_ID_INVALID)) {
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = authored_lane_count,
        .projected_lane_count = maximum_lane_count,
    };
    loom_target_contract_query_result_t query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(
        loom_target_legalization_query_contract_with_vector_lane_projection(
            context, op, projection, &query_result));
    contract_handled_candidate |=
        query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED;
    if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      selected_lane_count = iree_max(selected_lane_count, maximum_lane_count);
    }
  }
  if (selected_lane_count != 0) {
    *out_lane_count = selected_lane_count;
  } else if (!contract_handled_candidate) {
    // Some targets retain structural memory selection in a custom Low planner
    // instead of exposing candidate rows through the query contract. Preserve
    // their declared packet policy only when the contract had no opinion about
    // any candidate. One handled answer makes the contract authoritative.
    *out_lane_count = policy_lane_count;
  }
  if (out_contract_handled_candidate != NULL) {
    *out_contract_handled_candidate = contract_handled_candidate;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_packet_memory_chunk_lane_count_select(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint32_t maximum_lane_count, uint32_t* out_lane_count) {
  return loom_vector_packet_memory_chunk_lane_count_select_impl(
      context, op, policy, vector_type, maximum_lane_count, out_lane_count,
      /*out_contract_handled_candidate=*/NULL);
}

iree_status_t loom_vector_packet_load_candidate_bits(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint64_t* out_candidate_bits) {
  *out_candidate_bits = 0;
  loom_vector_memory_footprint_t footprint = {0};
  uint32_t source_lane_count = 0;
  if (!loom_vector_memory_footprint_describe(
          loom_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE ||
      !loom_type_equal(footprint.vector_type, vector_type) ||
      !loom_vector_packet_static_lane_count(vector_type, &source_lane_count)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t packet_lane_count = policy->native_lane_counts[i];
    if (packet_lane_count == 0 || packet_lane_count >= source_lane_count) {
      continue;
    }
    uint32_t selected_lane_count = 0;
    IREE_RETURN_IF_ERROR(loom_vector_packet_memory_chunk_lane_count_select(
        context, op, policy, vector_type, packet_lane_count,
        &selected_lane_count));
    if (selected_lane_count != packet_lane_count) {
      continue;
    }
    const uint32_t tail_lane_count = source_lane_count % packet_lane_count;
    if (tail_lane_count != 0) {
      IREE_RETURN_IF_ERROR(loom_vector_packet_memory_chunk_lane_count_select(
          context, op, policy, vector_type, tail_lane_count,
          &selected_lane_count));
      if (selected_lane_count != tail_lane_count) {
        continue;
      }
    }
    const loom_vector_packet_shape_t shape = {
        .lane_count = source_lane_count,
        .chunk_lane_count = packet_lane_count,
        .chunk_count = (source_lane_count - 1u) / packet_lane_count + 1u,
    };
    if (!loom_vector_packet_memory_can_build_chunk_origins(context, &footprint,
                                                           &shape)) {
      continue;
    }
    *out_candidate_bits |= (uint64_t)1u << i;
  }
  return iree_ok_status();
}

// Constrains a geometric packet shape with target contract answers for this
// memory operation. The authored access retains its address and view facts
// while the scoped query projects matching vector values to each candidate
// width.
iree_status_t loom_vector_packet_memory_shape_constrain(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* inout_shape, bool* out_supported) {
  *out_supported = false;
  uint32_t lane_count = 0;
  if (!loom_vector_packet_static_lane_count(vector_type, &lane_count) ||
      lane_count != inout_shape->lane_count || lane_count <= 1) {
    return iree_ok_status();
  }

  uint32_t selected_lane_count = 0;
  bool contract_handled_candidate = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_memory_chunk_lane_count_select_impl(
      context, op, policy, vector_type,
      iree_min(inout_shape->chunk_lane_count, lane_count - 1u),
      &selected_lane_count, &contract_handled_candidate));
  if (selected_lane_count == 0) {
    // Preserve the caller's geometric width for targets whose custom Low
    // planner does not expose any candidate contract rows. Exact component
    // planning uses the public selector above and instead treats a suffix with
    // no policy width as a scalar tail.
    *out_supported = !contract_handled_candidate;
    return iree_ok_status();
  }

  inout_shape->chunk_lane_count = selected_lane_count;
  inout_shape->chunk_count = (lane_count - 1u) / selected_lane_count + 1u;
  *out_supported = true;
  return iree_ok_status();
}

// Constrains a geometric packet shape to widths whose projected reduction is
// target-native. The full packets dominate reduction cost; a shorter terminal
// packet retains the existing bounded reference fallback when no exact target
// recipe covers its lane count.
iree_status_t loom_vector_packet_reduce_shape_constrain(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* inout_shape, bool* out_supported) {
  *out_supported = false;
  uint32_t lane_count = 0;
  if (!loom_vector_packet_static_lane_count(vector_type, &lane_count) ||
      lane_count != inout_shape->lane_count) {
    return iree_ok_status();
  }

  uint32_t selected_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_bit_count_count; ++i) {
    const uint32_t candidate_lane_count =
        loom_vector_packet_lane_count_for_bit_count(
            policy, vector_type, policy->native_bit_counts[i]);
    if (candidate_lane_count == 0 || candidate_lane_count >= lane_count ||
        candidate_lane_count > inout_shape->chunk_lane_count) {
      continue;
    }
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = lane_count,
        .projected_lane_count = candidate_lane_count,
    };
    loom_target_contract_query_result_t query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(
        loom_target_legalization_query_contract_with_vector_lane_projection(
            context, op, projection, &query_result));
    if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      selected_lane_count = iree_max(selected_lane_count, candidate_lane_count);
    }
  }
  if (selected_lane_count == 0) {
    return iree_ok_status();
  }

  inout_shape->chunk_lane_count = selected_lane_count;
  inout_shape->chunk_count = (lane_count - 1u) / selected_lane_count + 1u;
  *out_supported = true;
  return iree_ok_status();
}

// Queries the target contract for the lanewise fold that would combine packed
// reduction leaves. The prototype borrows the reduction input for its three
// value slots so the existing lane projection supplies the candidate type
// without adding speculative IR to the module.
static iree_status_t loom_vector_packet_query_combining_contract(
    loom_target_legalization_context_t* context, const loom_op_t* reduce_op,
    loom_target_contract_vector_lane_projection_t projection,
    loom_target_contract_query_result_t* out_result) {
  typedef struct loom_vector_binary_op_prototype_t {
    // Stack-owned operation header presented to the target contract.
    loom_op_t op;
    // Two operands followed by one result in loom_op_t trailing storage order.
    loom_value_id_t values[3];
    // Detached operand slots; contract queries never publish use-list entries.
    loom_use_index_t operand_use_indices[2];
  } loom_vector_binary_op_prototype_t;

  loom_op_kind_t op_kind = LOOM_OP_KIND_UNKNOWN;
  if (!loom_vector_combining_kind_op(loom_vector_reduce_kind(reduce_op),
                                     &op_kind)) {
    *out_result = loom_target_contract_query_result_empty();
    return iree_ok_status();
  }
  loom_vector_binary_op_prototype_t prototype = {
      .op =
          {
              .kind = op_kind,
              .operand_count = 2,
              .result_count = 1,
              .location = reduce_op->location,
              .instance_flags = loom_vector_reduce_fastmath(reduce_op),
              .block_ordinal = reduce_op->block_ordinal,
              .parent_op = reduce_op->parent_op,
              .parent_block = reduce_op->parent_block,
          },
  };
  const loom_op_vtable_t* prototype_vtable =
      loom_op_vtable(context->module, &prototype.op);
  IREE_ASSERT(prototype_vtable != NULL);
  prototype.op.traits = prototype_vtable->traits;
  const loom_value_id_t input = loom_vector_reduce_input(reduce_op);
  loom_op_operands(&prototype.op)[0] = input;
  loom_op_operands(&prototype.op)[1] = input;
  loom_op_results(&prototype.op)[0] = input;
  loom_op_operand_use_indices(&prototype.op)[0] = LOOM_USE_INDEX_INVALID;
  loom_op_operand_use_indices(&prototype.op)[1] = LOOM_USE_INDEX_INVALID;
  return loom_target_legalization_query_contract_with_vector_lane_projection(
      context, &prototype.op, projection, out_result);
}

// Returns whether a power-of-two leaf can be folded to two lanes using only
// target-native lanewise combining operations. The final pair is handled by
// scalar extraction, so a target does not need a synthetic one-lane vector
// contract to complete the fold.
static iree_status_t loom_vector_packet_reduction_leaf_can_fold(
    loom_target_legalization_context_t* context, const loom_op_t* reduce_op,
    uint32_t input_lane_count, uint32_t leaf_lane_count, bool* out_supported) {
  *out_supported = false;
  if (leaf_lane_count < 2 || (leaf_lane_count & (leaf_lane_count - 1u)) != 0) {
    return iree_ok_status();
  }
  while (leaf_lane_count > 2) {
    leaf_lane_count /= 2u;
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = input_lane_count,
        .projected_lane_count = leaf_lane_count,
    };
    loom_target_contract_query_result_t query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_vector_packet_query_combining_contract(
        context, reduce_op, projection, &query_result));
    if (query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      return iree_ok_status();
    }
  }
  *out_supported = true;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_reduce_combine_shape_constrain(
    loom_target_legalization_context_t* context, const loom_op_t* reduce_op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* inout_shape, bool* out_supported) {
  *out_supported = false;
  uint32_t lane_count = 0;
  int64_t identity = 0;
  double float_identity = 0.0;
  const loom_combining_kind_t kind = loom_vector_reduce_kind(reduce_op);
  const bool can_reassociate =
      loom_combining_kind_accepts_integer(kind) ||
      iree_any_bit_set(loom_vector_reduce_fastmath(reduce_op),
                       LOOM_VECTOR_FASTMATHFLAGS_REASSOC);
  const bool has_identity =
      loom_combining_kind_integer_identity(
          kind, loom_type_element_type(vector_type), &identity) ||
      (can_reassociate &&
       loom_combining_kind_float_identity(kind, &float_identity));
  const bool can_repeat =
      loom_combining_kind_is_idempotent(kind) && can_reassociate;
  if (!loom_vector_packet_static_lane_count(vector_type, &lane_count) ||
      lane_count != inout_shape->lane_count || (!has_identity && !can_repeat)) {
    return iree_ok_status();
  }

  uint32_t selected_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_bit_count_count; ++i) {
    const uint32_t candidate_lane_count =
        loom_vector_packet_lane_count_for_bit_count(
            policy, vector_type, policy->native_bit_counts[i]);
    if (candidate_lane_count < 2 || candidate_lane_count >= lane_count ||
        candidate_lane_count > inout_shape->chunk_lane_count) {
      continue;
    }
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = lane_count,
        .projected_lane_count = candidate_lane_count,
    };
    loom_target_contract_query_result_t query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_vector_packet_query_combining_contract(
        context, reduce_op, projection, &query_result));
    if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      selected_lane_count = iree_max(selected_lane_count, candidate_lane_count);
    }
  }
  if (selected_lane_count == 0) {
    return iree_ok_status();
  }

  inout_shape->chunk_lane_count = selected_lane_count;
  inout_shape->chunk_count = (lane_count - 1u) / selected_lane_count + 1u;
  *out_supported = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_partition_vector_reductions_are_native(
    loom_target_legalization_context_t* context, const loom_op_t* reduce_op,
    loom_value_fact_vector_partition_t partition, uint32_t input_lane_count,
    bool* out_native) {
  *out_native = true;
  uint32_t previous_lane_count = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    if (!loom_type_is_vector(component_type) ||
        component.lane_count == previous_lane_count) {
      continue;
    }
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = input_lane_count,
        .projected_lane_count = component.lane_count,
    };
    loom_target_contract_query_result_t query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(
        loom_target_legalization_query_contract_with_vector_lane_projection(
            context, reduce_op, projection, &query_result));
    if (query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      *out_native = false;
      return iree_ok_status();
    }
    previous_lane_count = component.lane_count;
  }
  return iree_ok_status();
}

static bool loom_vector_packet_partition_packed_work_plan(
    const loom_target_legalization_context_t* context,
    loom_value_fact_vector_partition_t partition, uint32_t candidate_lane_count,
    uint32_t scalar_count, bool combining_is_idempotent,
    bool has_combining_identity, bool reduction_is_native,
    uint32_t* out_packed_value_count, uint32_t* out_scalar_pack_lane_count) {
  *out_packed_value_count = 0;
  *out_scalar_pack_lane_count = 0;
  iree_host_size_t packet_count = 0;
  iree_host_size_t operation_count = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    if (loom_type_is_scalar(loom_module_value_type(
            context->module, partition.components[i].value))) {
      continue;
    }
    const uint32_t component_lane_count = partition.components[i].lane_count;
    const iree_host_size_t component_packet_count =
        (component_lane_count - 1u) / candidate_lane_count + 1u;
    if (!iree_host_size_checked_add(packet_count, component_packet_count,
                                    &packet_count)) {
      return false;
    }
    // Components larger than a leaf require one slice per packet. A short
    // terminal packet also requires a concat and, when repetition is not
    // algebraically valid, an identity constant.
    if (component_lane_count > candidate_lane_count &&
        !iree_host_size_checked_add(operation_count, component_packet_count,
                                    &operation_count)) {
      return false;
    }
    const uint32_t terminal_lane_count =
        component_lane_count % candidate_lane_count;
    if (terminal_lane_count != 0) {
      const bool can_repeat_component =
          combining_is_idempotent &&
          candidate_lane_count % terminal_lane_count == 0;
      if (!can_repeat_component && !has_combining_identity) {
        return false;
      }
      // Target legalization lowers a repeated multi-input concat to a binary
      // tree. Charge that final shape here instead of treating an arbitrarily
      // wide authored concat as one operation.
      const iree_host_size_t normalization_op_count =
          can_repeat_component ? candidate_lane_count / terminal_lane_count - 1u
                               : 2u;
      if (!iree_host_size_checked_add(operation_count, normalization_op_count,
                                      &operation_count)) {
        return false;
      }
    }
  }
  iree_host_size_t terminal_op_count = 1;
  if (!reduction_is_native) {
    // Each vector fold level builds two half-width slices and one combine. The
    // final pair needs two extracts, one scalar combine, and one accumulator
    // combine.
    terminal_op_count = 4;
    for (uint32_t lane_count = candidate_lane_count; lane_count > 2;
         lane_count /= 2u) {
      if (!iree_host_size_checked_add(terminal_op_count, 3u,
                                      &terminal_op_count)) {
        return false;
      }
    }
  }

  if (combining_is_idempotent && scalar_count != 0) {
    // Two-lane scalar groups map the second value to the target's cheap
    // zero-index insert and expose independent packets to the balanced fold.
    // Increase the group width only when that parallel form would cross the
    // shared static expansion limit.
    const uint32_t first_group_lane_count = scalar_count == 1 ? 1u : 2u;
    const uint32_t maximum_group_lane_count =
        iree_min(scalar_count, candidate_lane_count);
    for (uint32_t group_lane_count = first_group_lane_count;
         group_lane_count <= maximum_group_lane_count; ++group_lane_count) {
      const iree_host_size_t scalar_packet_count =
          (scalar_count + group_lane_count - 1u) / group_lane_count;
      iree_host_size_t candidate_packet_count = 0;
      iree_host_size_t candidate_operation_count = operation_count;
      if (!iree_host_size_checked_add(packet_count, scalar_packet_count,
                                      &candidate_packet_count) ||
          !iree_host_size_checked_add(candidate_operation_count, scalar_count,
                                      &candidate_operation_count) ||
          !iree_host_size_checked_add(candidate_operation_count,
                                      candidate_packet_count - 1u,
                                      &candidate_operation_count) ||
          !iree_host_size_checked_add(candidate_operation_count,
                                      terminal_op_count,
                                      &candidate_operation_count)) {
        return false;
      }
      if (candidate_operation_count <= LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
        *out_packed_value_count = (uint32_t)candidate_packet_count;
        *out_scalar_pack_lane_count = group_lane_count;
        return true;
      }
    }
    return false;
  }

  const iree_host_size_t scalar_packet_count = scalar_count != 0 ? 1u : 0u;
  if (!iree_host_size_checked_add(packet_count, scalar_packet_count,
                                  &packet_count) ||
      packet_count == 0) {
    return false;
  }
  if (scalar_count != 0 &&
      !iree_host_size_checked_add(operation_count, scalar_count + 1u,
                                  &operation_count)) {
    return false;
  }
  if (!iree_host_size_checked_add(operation_count, packet_count - 1u,
                                  &operation_count)) {
    return false;
  }
  if (!iree_host_size_checked_add(operation_count, terminal_op_count,
                                  &operation_count) ||
      operation_count > LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
    return false;
  }
  *out_packed_value_count = (uint32_t)packet_count;
  return true;
}

iree_status_t loom_vector_packet_partition_reduce_plan_select(
    loom_target_legalization_context_t* context, const loom_op_t* reduce_op,
    const loom_target_vector_packet_policy_t* policy,
    loom_value_fact_vector_partition_t partition,
    loom_vector_packet_partition_reduce_plan_t* out_plan) {
  *out_plan = (loom_vector_packet_partition_reduce_plan_t){0};
  const loom_value_id_t input = loom_vector_reduce_input(reduce_op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  uint32_t input_lane_count = 0;
  if (partition.count == 0 ||
      loom_value_fact_vector_partition_has_uniform_components(partition) ||
      !loom_vector_packet_static_lane_count(input_type, &input_lane_count) ||
      partition.lane_count != input_lane_count) {
    return iree_ok_status();
  }

  uint32_t scalar_count = 0;
  bool has_vector_component = false;
  uint32_t maximum_vector_lane_count = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    if (loom_type_is_scalar(component_type)) {
      if (component.lane_count != 1 || loom_type_element_type(component_type) !=
                                           loom_type_element_type(input_type)) {
        return iree_ok_status();
      }
      ++scalar_count;
      continue;
    }
    uint32_t component_lane_count = 0;
    if (!loom_vector_packet_static_lane_count(component_type,
                                              &component_lane_count) ||
        component_lane_count != component.lane_count ||
        !loom_type_element_type_equals(component_type, input_type)) {
      return iree_ok_status();
    }
    has_vector_component = true;
    maximum_vector_lane_count =
        iree_max(maximum_vector_lane_count, component_lane_count);
  }
  const loom_combining_kind_t kind = loom_vector_reduce_kind(reduce_op);
  const bool combining_accepts_integer =
      loom_combining_kind_accepts_integer(kind);
  if (!combining_accepts_integer && !loom_combining_kind_accepts_float(kind)) {
    return iree_ok_status();
  }
  if (!has_vector_component) {
    return iree_ok_status();
  }
  if (scalar_count == 0 &&
      partition.count <= LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
    bool vector_reductions_are_native = false;
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_partition_vector_reductions_are_native(
            context, reduce_op, partition, input_lane_count,
            &vector_reductions_are_native));
    if (vector_reductions_are_native) {
      // A vector-only partition needs one native reduction per component.
      // Normalizing unequal components to a shared leaf cannot use fewer
      // operations: it adds at least one normalization or padding operation
      // before the packet combines and terminal reduction.
      out_plan->mode = LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_HYBRID;
      return iree_ok_status();
    }
  }
  const bool can_reassociate =
      combining_accepts_integer ||
      iree_any_bit_set(loom_vector_reduce_fastmath(reduce_op),
                       LOOM_VECTOR_FASTMATHFLAGS_REASSOC);
  int64_t identity = 0;
  double float_identity = 0.0;
  const bool has_combining_identity =
      (combining_accepts_integer &&
       loom_combining_kind_integer_identity(
           kind, loom_type_element_type(input_type), &identity)) ||
      (can_reassociate &&
       loom_combining_kind_float_identity(kind, &float_identity));
  const bool combining_is_idempotent = loom_combining_kind_is_idempotent(kind);
  const bool can_pack =
      has_combining_identity || (combining_is_idempotent && can_reassociate);
  const uint32_t maximum_candidate_lane_count = maximum_vector_lane_count;
  uint32_t selected_lane_count = 0;
  for (uint8_t candidate_index = 0;
       can_pack && candidate_index < policy->native_bit_count_count;
       ++candidate_index) {
    const uint32_t candidate_lane_count =
        loom_vector_packet_lane_count_for_bit_count(
            policy, input_type, policy->native_bit_counts[candidate_index]);
    if (candidate_lane_count < 2 || candidate_lane_count >= input_lane_count ||
        candidate_lane_count > maximum_candidate_lane_count) {
      continue;
    }
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = input_lane_count,
        .projected_lane_count = candidate_lane_count,
    };
    loom_target_contract_query_result_t combining_query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_vector_packet_query_combining_contract(
        context, reduce_op, projection, &combining_query_result));
    if (combining_query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      continue;
    }

    loom_target_contract_query_result_t reduction_query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(
        loom_target_legalization_query_contract_with_vector_lane_projection(
            context, reduce_op, projection, &reduction_query_result));
    const bool reduction_is_native =
        reduction_query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL;
    if (!reduction_is_native) {
      bool leaf_can_fold = false;
      IREE_RETURN_IF_ERROR(loom_vector_packet_reduction_leaf_can_fold(
          context, reduce_op, input_lane_count, candidate_lane_count,
          &leaf_can_fold));
      if (!leaf_can_fold) {
        continue;
      }
    }
    uint32_t packed_value_count = 0;
    uint32_t scalar_pack_lane_count = 0;
    if (!loom_vector_packet_partition_packed_work_plan(
            context, partition, candidate_lane_count, scalar_count,
            combining_is_idempotent, has_combining_identity,
            reduction_is_native, &packed_value_count,
            &scalar_pack_lane_count)) {
      continue;
    }
    if (out_plan->leaf_reduction_is_native && !reduction_is_native) {
      continue;
    }
    if (out_plan->leaf_reduction_is_native == reduction_is_native &&
        candidate_lane_count <= selected_lane_count) {
      continue;
    }
    selected_lane_count = candidate_lane_count;
    out_plan->packed_value_count = packed_value_count;
    out_plan->scalar_pack_lane_count = scalar_pack_lane_count;
    out_plan->leaf_reduction_is_native = reduction_is_native;
  }
  if (selected_lane_count == 0) {
    bool vector_reductions_are_native = false;
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_partition_vector_reductions_are_native(
            context, reduce_op, partition, input_lane_count,
            &vector_reductions_are_native));
    if (partition.count <= LOOM_VECTOR_PACKET_STATIC_OP_LIMIT &&
        vector_reductions_are_native) {
      out_plan->mode = LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_HYBRID;
    }
    return iree_ok_status();
  }
  out_plan->mode = LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_PACKED;
  out_plan->leaf_shape = (loom_vector_packet_shape_t){
      .lane_count = input_lane_count,
      .chunk_lane_count = selected_lane_count,
      .chunk_count = (input_lane_count - 1u) / selected_lane_count + 1u,
  };
  return iree_ok_status();
}

// Uniform producers do not need axis-aware slicing, so their packet plan can
// cover any static logical shape in row-major lane order.
static bool loom_vector_packet_uniform_shape_from_type(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* out_shape) {
  *out_shape = (loom_vector_packet_shape_t){0};
  uint64_t lane_count_u64 = 0;
  if (!loom_type_is_vector(vector_type) ||
      !loom_type_static_element_count(vector_type, &lane_count_u64) ||
      lane_count_u64 < 1 || lane_count_u64 > UINT32_MAX) {
    return false;
  }
  const uint32_t lane_count = (uint32_t)lane_count_u64;
  uint32_t chunk_lane_count = 0;
  if (!loom_target_vector_packet_policy_requires_packets(policy, vector_type) ||
      !loom_vector_packet_select_native_chunk_lane_count(
          policy, vector_type, lane_count - 1u, &chunk_lane_count)) {
    return false;
  }
  *out_shape = (loom_vector_packet_shape_t){
      .lane_count = lane_count,
      .chunk_lane_count = chunk_lane_count,
      .chunk_count = (lane_count - 1u) / chunk_lane_count + 1u,
  };
  return true;
}

static bool loom_vector_packet_shape_constrain(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* inout_shape) {
  uint32_t lane_count = 0;
  uint32_t chunk_lane_count = 0;
  if (!loom_vector_packet_static_lane_count(vector_type, &lane_count) ||
      lane_count != inout_shape->lane_count ||
      !loom_vector_packet_maximum_chunk_lane_count(policy, vector_type,
                                                   &chunk_lane_count)) {
    return false;
  }
  inout_shape->chunk_lane_count =
      iree_min(inout_shape->chunk_lane_count, chunk_lane_count);
  inout_shape->chunk_count =
      (lane_count - 1u) / inout_shape->chunk_lane_count + 1u;
  return true;
}

bool loom_vector_packet_table_lookup_shape(
    const loom_target_vector_packet_policy_t* policy, loom_type_t index_type,
    loom_type_t result_type, loom_vector_packet_shape_t* out_shape) {
  *out_shape = (loom_vector_packet_shape_t){0};
  uint32_t index_lane_count = 0;
  uint32_t result_lane_count = 0;
  uint32_t index_packet_lane_count = 0;
  uint32_t result_packet_lane_count = 0;
  if (!loom_vector_packet_static_lane_count(index_type, &index_lane_count) ||
      !loom_vector_packet_static_lane_count(result_type, &result_lane_count) ||
      index_lane_count != result_lane_count ||
      !loom_vector_packet_maximum_chunk_lane_count(policy, index_type,
                                                   &index_packet_lane_count) ||
      !loom_vector_packet_maximum_chunk_lane_count(policy, result_type,
                                                   &result_packet_lane_count)) {
    return false;
  }
  const uint32_t packet_lane_count =
      iree_min(index_packet_lane_count, result_packet_lane_count);
  if (result_lane_count <= packet_lane_count) {
    return false;
  }
  *out_shape = (loom_vector_packet_shape_t){
      .lane_count = result_lane_count,
      .chunk_lane_count = packet_lane_count,
      .chunk_count = (result_lane_count - 1u) / packet_lane_count + 1u,
  };
  return true;
}

loom_type_t loom_vector_packet_type(loom_type_t vector_type,
                                    uint32_t lane_count) {
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(vector_type),
      loom_dim_pack_static(lane_count), vector_type.encoding_id);
}

static bool loom_vector_packet_memory_find_dynamic_axis_index(
    loom_attribute_t static_indices, iree_host_size_t axis,
    iree_host_size_t* out_dynamic_index) {
  iree_host_size_t dynamic_index = 0;
  for (iree_host_size_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (i == axis) {
      *out_dynamic_index = dynamic_index;
      return true;
    }
    ++dynamic_index;
  }
  return false;
}

bool loom_vector_packet_memory_can_build_chunk_origins(
    const loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape) {
  if (shape->chunk_count <= 1) {
    return true;
  }
  if (footprint->static_indices.count == 0) {
    return false;
  }

  const uint32_t max_lane_offset =
      (shape->chunk_count - 1u) * shape->chunk_lane_count;
  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  const int64_t last_static_index =
      footprint->static_indices.i64_array[last_axis];
  if (last_static_index != INT64_MIN) {
    int64_t last_chunk_static_index = 0;
    return iree_checked_add_i64(last_static_index, (int64_t)max_lane_offset,
                                &last_chunk_static_index);
  }

  iree_host_size_t dynamic_index = 0;
  if (!loom_vector_packet_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index) ||
      dynamic_index >= footprint->dynamic_indices.count) {
    return false;
  }
  const loom_value_id_t dynamic_value =
      footprint->dynamic_indices.values[dynamic_index];
  const loom_type_t dynamic_type =
      loom_module_value_type(context->module, dynamic_value);
  return loom_type_is_scalar(dynamic_type) &&
         loom_type_element_type(dynamic_type) == LOOM_SCALAR_TYPE_INDEX;
}

iree_status_t loom_vector_packet_memory_build_chunk_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* source_op,
    uint32_t chunk_lane_offset, const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count, bool* out_built) {
  *out_dynamic_indices = footprint->dynamic_indices.values;
  *out_dynamic_index_count = footprint->dynamic_indices.count;
  *out_static_indices = footprint->static_indices.i64_array;
  *out_static_index_count = footprint->static_indices.count;
  *out_built = true;
  if (chunk_lane_offset == 0) {
    return iree_ok_status();
  }
  if (footprint->static_indices.count == 0) {
    *out_built = false;
    return iree_ok_status();
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, footprint->static_indices.count,
      sizeof(*static_indices), (void**)&static_indices));
  memcpy(static_indices, footprint->static_indices.i64_array,
         footprint->static_indices.count * sizeof(*static_indices));

  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  if (static_indices[last_axis] != INT64_MIN) {
    if (!iree_checked_add_i64(static_indices[last_axis],
                              (int64_t)chunk_lane_offset,
                              &static_indices[last_axis])) {
      *out_built = false;
      return iree_ok_status();
    }
    *out_static_indices = static_indices;
    return iree_ok_status();
  }

  iree_host_size_t dynamic_index = 0;
  if (!loom_vector_packet_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index) ||
      dynamic_index >= footprint->dynamic_indices.count) {
    *out_built = false;
    return iree_ok_status();
  }
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, footprint->dynamic_indices.count,
      sizeof(*dynamic_indices), (void**)&dynamic_indices));
  memcpy(dynamic_indices, footprint->dynamic_indices.values,
         footprint->dynamic_indices.count * sizeof(*dynamic_indices));

  const loom_value_id_t dynamic_value = dynamic_indices[dynamic_index];
  const loom_type_t dynamic_type =
      loom_module_value_type(context->module, dynamic_value);
  if (!loom_type_is_scalar(dynamic_type) ||
      loom_type_element_type(dynamic_type) != LOOM_SCALAR_TYPE_INDEX) {
    *out_built = false;
    return iree_ok_status();
  }
  loom_op_t* offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(chunk_lane_offset),
      dynamic_type, source_op->location, &offset_op));
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(&context->rewriter->builder, dynamic_value,
                           loom_index_constant_result(offset_op), dynamic_type,
                           source_op->location, &add_op));
  dynamic_indices[dynamic_index] = loom_index_add_result(add_op);

  *out_dynamic_indices = dynamic_indices;
  *out_static_indices = static_indices;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_memory_build_dynamic_chunk_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* source_op,
    loom_value_id_t lane_offset, const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count) {
  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  const int64_t last_static_index =
      footprint->static_indices.i64_array[last_axis];
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  if (last_static_index != INT64_MIN) {
    int64_t* static_indices = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, footprint->static_indices.count,
        sizeof(*static_indices), (void**)&static_indices));
    memcpy(static_indices, footprint->static_indices.i64_array,
           footprint->static_indices.count * sizeof(*static_indices));
    static_indices[last_axis] = INT64_MIN;

    const iree_host_size_t dynamic_index_count =
        footprint->dynamic_indices.count + 1u;
    loom_value_id_t* dynamic_indices = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, dynamic_index_count, sizeof(*dynamic_indices),
        (void**)&dynamic_indices));
    if (footprint->dynamic_indices.count != 0) {
      memcpy(dynamic_indices, footprint->dynamic_indices.values,
             footprint->dynamic_indices.count * sizeof(*dynamic_indices));
    }
    loom_value_id_t dynamic_last_index = lane_offset;
    if (last_static_index != 0) {
      loom_op_t* base_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_index_constant_build(builder, loom_attr_i64(last_static_index),
                                    index_type, source_op->location, &base_op));
      loom_op_t* add_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, loom_index_constant_result(base_op), lane_offset, index_type,
          source_op->location, &add_op));
      dynamic_last_index = loom_index_add_result(add_op);
    }
    dynamic_indices[dynamic_index_count - 1u] = dynamic_last_index;

    *out_dynamic_indices = dynamic_indices;
    *out_dynamic_index_count = dynamic_index_count;
    *out_static_indices = static_indices;
    *out_static_index_count = footprint->static_indices.count;
    return iree_ok_status();
  }

  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, footprint->dynamic_indices.count,
      sizeof(*dynamic_indices), (void**)&dynamic_indices));
  memcpy(dynamic_indices, footprint->dynamic_indices.values,
         footprint->dynamic_indices.count * sizeof(*dynamic_indices));
  iree_host_size_t dynamic_index = 0;
  const bool found_dynamic_index =
      loom_vector_packet_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index);
  IREE_ASSERT_TRUE(found_dynamic_index);
  (void)found_dynamic_index;
  IREE_ASSERT_LT(dynamic_index, footprint->dynamic_indices.count);
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(builder, dynamic_indices[dynamic_index], lane_offset,
                           index_type, source_op->location, &add_op));
  dynamic_indices[dynamic_index] = loom_index_add_result(add_op);

  *out_dynamic_indices = dynamic_indices;
  *out_dynamic_index_count = footprint->dynamic_indices.count;
  *out_static_indices = footprint->static_indices.i64_array;
  *out_static_index_count = footprint->static_indices.count;
  return iree_ok_status();
}

const loom_fact_context_t* loom_vector_packet_fact_context(
    const loom_target_legalization_context_t* context) {
  return context->fact_table ? &context->fact_table->context : NULL;
}

static bool loom_vector_packet_type_shape_matches(
    loom_type_t type, const loom_vector_packet_shape_t* shape) {
  uint32_t lane_count = 0;
  return loom_vector_packet_static_lane_count(type, &lane_count) &&
         lane_count == shape->lane_count;
}

void loom_vector_packetization_initialize(
    loom_target_legalization_context_t* context,
    const loom_target_vector_packet_policy_t* policy,
    loom_vector_packetization_t* out_packetization) {
  *out_packetization = (loom_vector_packetization_t){
      .context = context,
      .policy = policy,
  };
}

// Each branch consumes one source-ID bit, bounding lookup to 29 records even
// for sparse identities. Only selected producers occupy the index: unrelated
// function values require neither storage nor initialization per packet root.
static uint32_t* loom_vector_packet_find_index(
    loom_vector_packetization_t* packetization, loom_value_id_t source) {
  const uint32_t root_mask = (1u << LOOM_VECTOR_PACKET_INDEX_ROOT_BITS) - 1u;
  uint32_t* index = &packetization->roots[source & root_mask];
  uint32_t remaining_bits = source >> LOOM_VECTOR_PACKET_INDEX_ROOT_BITS;
  while (*index != 0) {
    loom_vector_packetized_value_t* value = &packetization->values[*index - 1u];
    if (value->source == source) {
      break;
    }
    index = &value->children[remaining_bits & 1u];
    remaining_bits >>= 1;
  }
  return index;
}

loom_vector_packetized_value_t* loom_vector_packet_find(
    loom_vector_packetization_t* packetization, loom_value_id_t source) {
  const uint32_t value_index =
      *loom_vector_packet_find_index(packetization, source);
  return value_index == 0 ? NULL : &packetization->values[value_index - 1u];
}

static iree_status_t loom_vector_packet_reserve(
    loom_vector_packetization_t* packetization, uint32_t capacity) {
  if (capacity <= packetization->value_capacity) {
    return iree_ok_status();
  }
  uint32_t new_capacity =
      packetization->value_capacity == 0 ? 8u : packetization->value_capacity;
  while (new_capacity < capacity) {
    if (new_capacity > UINT32_MAX / 2u) {
      new_capacity = capacity;
      break;
    }
    new_capacity *= 2u;
  }
  loom_vector_packetized_value_t* new_values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(packetization->context->arena, new_capacity,
                                sizeof(*new_values), (void**)&new_values));
  if (packetization->value_count != 0) {
    memcpy(new_values, packetization->values,
           packetization->value_count * sizeof(*new_values));
  }
  packetization->values = new_values;
  packetization->value_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_record(
    loom_vector_packetization_t* packetization, loom_value_id_t source,
    loom_type_t source_type) {
  IREE_RETURN_IF_ERROR(loom_vector_packet_reserve(
      packetization, packetization->value_count + 1u));
  uint32_t* index = loom_vector_packet_find_index(packetization, source);
  IREE_ASSERT_EQ(*index, 0u);
  const uint32_t value_index = packetization->value_count++;
  packetization->values[value_index] = (loom_vector_packetized_value_t){
      .source = source,
      .source_type = source_type,
      .packet =
          (loom_vector_packet_value_t){
              .vector = LOOM_VALUE_ID_INVALID,
              .scalar = LOOM_VALUE_ID_INVALID,
          },
  };
  *index = value_index + 1u;
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_record_captured(
    loom_vector_packetization_t* packetization, loom_value_id_t source,
    loom_type_t source_type, bool* out_selected) {
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_record(packetization, source, source_type));
  loom_vector_packetized_value_t* packetized_value =
      loom_vector_packet_find(packetization, source);
  IREE_ASSERT(packetized_value != NULL);
  packetized_value->source_mode = LOOM_VECTOR_PACKET_SOURCE_MODE_CAPTURED;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_select_value_shape(
    loom_vector_packetization_t* packetization, loom_value_id_t source,
    loom_vector_packet_shape_t* inout_shape, bool* out_selected);

static iree_status_t loom_vector_packet_select_memory_load_shape(
    const loom_vector_packetization_t* packetization, const loom_op_t* op,
    loom_vector_packet_shape_t* inout_shape, bool* out_selected) {
  *out_selected = false;
  loom_target_legalization_context_t* context = packetization->context;
  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE ||
      !loom_type_equal(footprint.vector_type,
                       loom_module_value_type(context->module,
                                              loom_vector_load_result(op)))) {
    return iree_ok_status();
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  // Scalar packet materialization uses view.load and wraps the result in a
  // one-lane vector. It does not require a native vector.load contract.
  if (inout_shape->chunk_lane_count == 1u) {
    *out_selected = true;
    return iree_ok_status();
  }
  return loom_vector_packet_memory_shape_constrain(
      context, op, packetization->policy, footprint.vector_type, inout_shape,
      out_selected);
}

// Selects a retained regular vector partition as an already captured packet
// snapshot. Constraining the root packet width to divide the authored width
// makes every later packet an O(1) lookup or a static slice of one partition.
static bool loom_vector_packet_select_snapshot_shape(
    const loom_vector_packetization_t* packetization,
    loom_value_id_t source_value, loom_vector_packet_shape_t* inout_shape,
    loom_value_fact_vector_partition_t* out_partition) {
  *out_partition = loom_value_fact_table_query_vector_partition(
      packetization->context->fact_table, source_value);
  if (out_partition->snapshot_count == 0) {
    *out_partition = (loom_value_fact_vector_partition_t){0};
    return false;
  }
  const uint32_t packet_lane_count = out_partition->snapshot_lane_count;
  uint32_t chunk_lane_count = inout_shape->chunk_lane_count;
  if (chunk_lane_count > packet_lane_count ||
      packet_lane_count % chunk_lane_count != 0) {
    const loom_type_t result_type =
        loom_module_value_type(packetization->context->module, source_value);
    const int32_t element_bit_count = loom_vector_packet_element_bit_count(
        packetization->policy, loom_type_element_type(result_type));
    if (element_bit_count <= 0) {
      return false;
    }
    chunk_lane_count = 0;
    for (uint8_t i = 0; i < packetization->policy->native_bit_count_count;
         ++i) {
      const uint16_t bit_count = packetization->policy->native_bit_counts[i];
      if (bit_count == 0 || bit_count % (uint32_t)element_bit_count != 0) {
        continue;
      }
      const uint32_t candidate_lane_count =
          bit_count / (uint32_t)element_bit_count;
      if (candidate_lane_count <= inout_shape->chunk_lane_count &&
          candidate_lane_count <= packet_lane_count &&
          packet_lane_count % candidate_lane_count == 0) {
        chunk_lane_count = iree_max(chunk_lane_count, candidate_lane_count);
      }
    }
    if (chunk_lane_count == 0) {
      return false;
    }
  }

  inout_shape->chunk_lane_count = chunk_lane_count;
  inout_shape->chunk_count =
      (inout_shape->lane_count - 1u) / chunk_lane_count + 1u;
  return true;
}

static iree_status_t loom_vector_packet_select_decomposable_op_shape(
    loom_vector_packetization_t* packetization, const loom_op_t* op,
    loom_vector_packet_shape_t* inout_shape, bool* out_selected) {
  *out_selected = false;
  const loom_target_legalization_context_t* context = packetization->context;
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  if (!loom_vector_packet_type_shape_matches(result_type, inout_shape)) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_type_t operand_type =
        loom_module_value_type(context->module, operands[i]);
    if (!loom_vector_packet_type_shape_matches(operand_type, inout_shape)) {
      return iree_ok_status();
    }
    bool operand_selected = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_select_value_shape(
        packetization, operands[i], inout_shape, &operand_selected));
    if (!operand_selected) {
      return iree_ok_status();
    }
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_select_value_shape(
    loom_vector_packetization_t* packetization, loom_value_id_t source,
    loom_vector_packet_shape_t* inout_shape, bool* out_selected) {
  *out_selected = false;
  if (loom_vector_packet_find(packetization, source) != NULL) {
    *out_selected = true;
    return iree_ok_status();
  }
  const loom_target_legalization_context_t* context = packetization->context;
  const loom_value_t* value = loom_module_value(context->module, source);
  if (value == NULL) {
    return iree_ok_status();
  }
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  if (!loom_vector_packet_type_shape_matches(source_type, inout_shape)) {
    return iree_ok_status();
  }
  if (loom_value_is_block_arg(value)) {
    if (!iree_any_bit_set(packetization->selection_flags,
                          LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_EXTERNAL)) {
      return iree_ok_status();
    }
    if (!loom_vector_packet_shape_constrain(packetization->policy, source_type,
                                            inout_shape)) {
      return iree_ok_status();
    }
    return loom_vector_packet_record_captured(packetization, source,
                                              source_type, out_selected);
  }
  if (!loom_vector_packet_shape_constrain(packetization->policy, source_type,
                                          inout_shape)) {
    return iree_ok_status();
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (op == NULL) {
    return iree_ok_status();
  }
  loom_value_fact_vector_partition_t source_partition = {0};
  if (loom_vector_packet_select_snapshot_shape(
          packetization, source, inout_shape, &source_partition)) {
    *out_selected = true;
  } else if (loom_vector_load_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_select_memory_load_shape(
        packetization, op, inout_shape, out_selected));
  } else if (loom_vector_concat_isa(op)) {
    // Irregular concatenations are not replayable packet snapshots.
  } else if (loom_vector_from_elements_isa(op) ||
             loom_vector_constant_isa(op) || loom_vector_poison_isa(op) ||
             loom_vector_splat_isa(op)) {
    *out_selected = true;
  } else {
    if (iree_any_bit_set(packetization->selection_flags,
                         LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_EXTERNAL) &&
        loom_scf_for_isa(op)) {
      return loom_vector_packet_record_captured(packetization, source,
                                                source_type, out_selected);
    }
    const loom_trait_flags_t traits =
        loom_op_effective_traits(context->module, op);
    if (!iree_all_bits_set(traits, LOOM_TRAIT_DECOMPOSABLE)) {
      if (!iree_any_bit_set(
              packetization->selection_flags,
              LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_NON_DECOMPOSABLE)) {
        return iree_ok_status();
      }
      if (loom_vector_slice_isa(op)) {
        bool dynamic_window_owns_consumer_chain = false;
        IREE_RETURN_IF_ERROR(
            loom_vector_packet_dynamic_window_owns_consumer_chain(
                packetization->context, packetization->policy, op,
                &dynamic_window_owns_consumer_chain));
        if (dynamic_window_owns_consumer_chain) {
          return iree_ok_status();
        }
      }
      return loom_vector_packet_record_captured(packetization, source,
                                                source_type, out_selected);
    }
    IREE_RETURN_IF_ERROR(loom_vector_packet_select_decomposable_op_shape(
        packetization, op, inout_shape, out_selected));
  }
  if (*out_selected) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_record(packetization, source, source_type));
    if (source_partition.snapshot_count != 0) {
      loom_vector_packetized_value_t* packetized_value =
          loom_vector_packet_find(packetization, source);
      IREE_ASSERT(packetized_value != NULL);
      packetized_value->source_mode = LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT;
      packetized_value->source_partition = source_partition;
    } else if (loom_vector_from_elements_isa(op)) {
      loom_vector_packetized_value_t* packetized_value =
          loom_vector_packet_find(packetization, source);
      IREE_ASSERT(packetized_value != NULL);
      packetized_value->source_mode =
          LOOM_VECTOR_PACKET_SOURCE_MODE_STATIC_ELEMENTS;
    }
  }
  return iree_ok_status();
}

static loom_vector_packetized_value_t* loom_vector_packet_find_op_result(
    loom_vector_packetization_t* packetization, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_vector_packetized_value_t* packetized_value =
        loom_vector_packet_find(packetization, results[i]);
    if (packetized_value != NULL) {
      return packetized_value;
    }
  }
  return NULL;
}

static bool loom_vector_packet_value_is_exclusive(
    loom_vector_packetization_t* packetization,
    const loom_vector_packetized_value_t* packetized_value,
    const loom_op_t* root_op) {
  const loom_module_t* module = packetization->context->module;
  const loom_value_id_t source = packetized_value->source;
  const loom_value_t* value = loom_module_value(module, source);
  if (loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(module, source)) {
    return false;
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (user_op != root_op &&
        loom_vector_packet_find_op_result(packetization, user_op) == NULL) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_packet_graph_is_exclusive(
    loom_vector_packetization_t* packetization, const loom_op_t* root_op) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    if (!loom_vector_packet_value_is_exclusive(
            packetization, &packetization->values[i], root_op)) {
      return false;
    }
  }
  return true;
}

// Returns true when materializing a shared snapshot's authored packet tuple
// once for each direct consumer would exceed the static expansion bound. This
// lower bound is independent of the consumers' element types and producer
// graphs; work within the current graph remains covered by the ordinary static
// expansion check.
bool loom_vector_packet_shared_snapshot_requires_staging(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape, const loom_op_t* root_op) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    const loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i];
    if (packetized_value->source_mode !=
            LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT ||
        loom_vector_packet_value_is_exclusive(packetization, packetized_value,
                                              root_op)) {
      continue;
    }

    const loom_value_t* value = loom_module_value(
        packetization->context->module, packetized_value->source);
    // One packet tuple plus one per distinct consumer must fit the budget.
    // Keep only distinct users and stop as soon as that bound is exceeded;
    // repeated operand uses never enlarge the search domain.
    const uint32_t tuple_limit =
        LOOM_VECTOR_PACKET_STATIC_OP_LIMIT / shape->chunk_count;
    if (tuple_limit == 0) {
      return true;
    }
    const loom_op_t* consumers[LOOM_VECTOR_PACKET_STATIC_OP_LIMIT];
    uint32_t consumer_count = 0;
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t use_index = 0; use_index < value->use_count; ++use_index) {
      const loom_op_t* consumer = loom_use_user_op(uses[use_index]);
      bool first_use = true;
      for (uint32_t previous_index = 0; previous_index < consumer_count;
           ++previous_index) {
        if (consumers[previous_index] == consumer) {
          first_use = false;
          break;
        }
      }
      if (first_use) {
        if (consumer_count + 1u >= tuple_limit) {
          return true;
        }
        consumers[consumer_count++] = consumer;
      }
    }
  }
  return false;
}

// Classifies load leaves that may be streamed at the root instead of captured
// at their authored position. A shared producer graph may replay a load whose
// value has no use outside that graph; other shared loads retain one authored
// snapshot. Every replay uses the function's indexed motion barriers instead
// of rescanning the block for each consumer.
iree_status_t loom_vector_packet_classify_memory_loads(
    loom_vector_packetization_t* packetization, const loom_op_t* root_op,
    bool allow_streaming) {
  uint32_t remaining_load_count = 0;
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i];
    const loom_value_t* value = loom_module_value(
        packetization->context->module, packetized_value->source);
    if (!loom_value_is_block_arg(value) &&
        loom_vector_load_isa(loom_value_def_op(value))) {
      packetized_value->source_mode =
          LOOM_VECTOR_PACKET_SOURCE_MODE_PRESERVE_LOAD;
      ++remaining_load_count;
    }
  }
  if (remaining_load_count == 0 || !allow_streaming) {
    return iree_ok_status();
  }

  const loom_module_t* module = packetization->context->module;
  const bool graph_is_exclusive =
      loom_vector_packet_graph_is_exclusive(packetization, root_op);
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i];
    if (packetized_value->source_mode !=
        LOOM_VECTOR_PACKET_SOURCE_MODE_PRESERVE_LOAD) {
      continue;
    }
    const loom_value_t* value =
        loom_module_value(module, packetized_value->source);
    loom_op_t* load_op = loom_value_def_op(value);
    if (!loom_motion_op_is_ordinary_load(module, load_op)) {
      continue;
    }
    if (!graph_is_exclusive && !loom_vector_packet_value_is_exclusive(
                                   packetization, packetized_value, root_op)) {
      continue;
    }
    if (packetization->context->read_motion_barriers == NULL) {
      IREE_RETURN_IF_ERROR(loom_motion_read_barrier_table_allocate(
          module, packetization->context->arena,
          &packetization->context->read_motion_barriers));
    }
    bool can_cross = false;
    IREE_RETURN_IF_ERROR(loom_motion_read_barrier_table_can_cross(
        packetization->context->read_motion_barriers, load_op, root_op,
        &can_cross));
    if (can_cross) {
      packetized_value->source_mode = LOOM_VECTOR_PACKET_SOURCE_MODE_STREAMED;
    }
  }
  return iree_ok_status();
}

static bool loom_vector_packet_can_materialize_memory_load(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_vector_packet_shape_t* shape) {
  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE ||
      !loom_type_equal(footprint.vector_type,
                       loom_module_value_type(context->module,
                                              loom_vector_load_result(op)))) {
    return false;
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  return loom_vector_memory_cache_policy_from_op(context->module, op,
                                                 &cache_policy) &&
         loom_vector_packet_memory_can_build_chunk_origins(context, &footprint,
                                                           shape);
}

bool loom_vector_packet_can_materialize(
    const loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    const loom_vector_packetized_value_t* value = &packetization->values[i];
    const loom_value_t* source_value =
        loom_module_value(packetization->context->module, value->source);
    if (loom_value_is_block_arg(source_value)) {
      continue;
    }
    const loom_op_t* op = loom_value_def_op(source_value);
    if (loom_vector_load_isa(op) &&
        !loom_vector_packet_can_materialize_memory_load(packetization->context,
                                                        op, shape)) {
      return false;
    }
    if (value->source_mode == LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT) {
      const uint32_t packet_lane_count =
          value->source_partition.snapshot_lane_count;
      if (packet_lane_count < shape->chunk_lane_count ||
          packet_lane_count % shape->chunk_lane_count != 0) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_vector_packet_footprints_match(
    const loom_vector_memory_footprint_t* left,
    const loom_vector_memory_footprint_t* right) {
  return left->view == right->view &&
         loom_type_equal(left->vector_type, right->vector_type) &&
         loom_attribute_equal(&left->static_indices, &right->static_indices) &&
         left->dynamic_indices.count == right->dynamic_indices.count &&
         (left->dynamic_indices.count == 0 ||
          memcmp(left->dynamic_indices.values, right->dynamic_indices.values,
                 left->dynamic_indices.count * sizeof(loom_value_id_t)) == 0);
}

static bool loom_vector_packet_views_are_disjoint(
    const loom_target_legalization_context_t* context, loom_value_id_t left,
    loom_value_id_t right) {
  loom_value_fact_view_reference_t left_reference = {0};
  loom_value_fact_view_reference_t right_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table, left),
          &left_reference) ||
      !loom_value_facts_query_view_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table, right),
          &right_reference)) {
    return false;
  }
  if (loom_value_fact_reference_origins_are_disjoint(left_reference.origin,
                                                     right_reference.origin)) {
    return true;
  }
  if (left_reference.root_value_id == LOOM_VALUE_ID_INVALID ||
      right_reference.root_value_id == LOOM_VALUE_ID_INVALID ||
      left_reference.root_value_id == right_reference.root_value_id) {
    return false;
  }
  return loom_view_memory_spaces_are_disjoint(left_reference.memory_space,
                                              right_reference.memory_space) ||
         (left_reference.alias_scope_id !=
              LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE &&
          right_reference.alias_scope_id !=
              LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE &&
          left_reference.alias_scope_id != right_reference.alias_scope_id);
}

bool loom_vector_packet_store_can_interleave(
    const loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    const loom_vector_packetized_value_t* value = &packetization->values[i];
    const loom_value_t* source_value =
        loom_module_value(packetization->context->module, value->source);
    const loom_op_t* op = loom_value_def_op(source_value);
    if (!loom_vector_load_isa(op)) {
      continue;
    }

    loom_vector_memory_footprint_t load_footprint = {0};
    const bool footprint_described = loom_vector_memory_footprint_describe(
        loom_vector_packet_fact_context(packetization->context),
        packetization->context->module, op, &load_footprint);
    IREE_ASSERT_TRUE(footprint_described);
    if (!loom_vector_packet_footprints_match(&load_footprint,
                                             store_footprint) &&
        !loom_vector_packet_views_are_disjoint(packetization->context,
                                               load_footprint.view,
                                               store_footprint->view)) {
      return false;
    }
  }
  return true;
}
