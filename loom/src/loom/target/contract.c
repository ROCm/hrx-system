// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/contract.h"

#include <stddef.h>
#include <stdint.h>

#include "loom/ir/scalar_type.h"

bool loom_target_vector_packet_policy_requires_packets(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type) {
  if (policy == NULL || !loom_type_is_vector(vector_type) ||
      !loom_type_is_all_static(vector_type)) {
    return false;
  }
  uint64_t element_count = 0;
  if (!loom_type_static_element_count(vector_type, &element_count) ||
      element_count == 0 || element_count > UINT32_MAX) {
    return false;
  }

  const loom_scalar_type_t element_type = loom_type_element_type(vector_type);
  int32_t element_bit_count = loom_scalar_type_bitwidth(element_type);
  if (element_type == LOOM_SCALAR_TYPE_INDEX && policy->index_bit_count != 0) {
    element_bit_count = policy->index_bit_count;
  } else if (element_type == LOOM_SCALAR_TYPE_OFFSET &&
             policy->offset_bit_count != 0) {
    element_bit_count = policy->offset_bit_count;
  }
  if (element_bit_count <= 0) {
    return false;
  }

  for (uint8_t i = 0; i < policy->structural_lane_limit_count; ++i) {
    const loom_target_vector_packet_lane_limit_t* limit =
        &policy->structural_lane_limits[i];
    if (limit->element_type == element_type &&
        element_count > limit->maximum_lane_count) {
      return true;
    }
  }
  return policy->maximum_unpacketized_bit_count !=
             LOOM_TARGET_VECTOR_PACKET_UNBOUNDED_BIT_COUNT &&
         element_count * (uint32_t)element_bit_count >
             policy->maximum_unpacketized_bit_count;
}

loom_type_t loom_target_contract_query_projected_value_type(
    loom_target_contract_vector_lane_projection_t projection,
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_type_t authored_type = loom_module_value_type(module, value_id);
  uint64_t lane_count = 0;
  if (!loom_type_is_vector(authored_type) ||
      !loom_type_static_element_count(authored_type, &lane_count) ||
      lane_count > UINT32_MAX) {
    return authored_type;
  }
  const uint32_t projected_lane_count =
      loom_target_contract_query_vector_lane_count(projection,
                                                   (uint32_t)lane_count);
  if (projected_lane_count == lane_count) {
    return authored_type;
  }
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(authored_type),
      loom_dim_pack_static(projected_lane_count),
      loom_type_rank(authored_type) == 1 ? authored_type.encoding_id : 0);
}

iree_status_t loom_target_contract_query_resolve_function_analysis(
    loom_target_contract_query_environment_t* environment) {
  if (environment->view_regions != NULL ||
      environment->function_analysis.fn == NULL) {
    return iree_ok_status();
  }
  return environment->function_analysis.fn(
      environment->function_analysis.user_data, &environment->value_domain,
      &environment->view_regions);
}

iree_status_t loom_target_contract_query_get_or_allocate_target_state(
    const loom_target_contract_query_environment_t* environment,
    const void* key, iree_host_size_t data_length, void** out_data) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(key);
  IREE_ASSERT_GT(data_length, 0);
  IREE_ASSERT_ARGUMENT(out_data);
  *out_data = NULL;
  if (environment->target_state_allocator.fn == NULL) {
    return iree_ok_status();
  }
  return environment->target_state_allocator.fn(
      environment->target_state_allocator.user_data, key, data_length,
      out_data);
}
