// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/storage.h"

#include "loom/ir/scalar_type.h"

uint32_t loom_vector_static_rank1_lane_count(loom_type_t type,
                                             loom_scalar_type_t element_type,
                                             uint32_t maximum_lane_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type) ||
      loom_type_element_type(type) != element_type) {
    return 0;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > (int64_t)maximum_lane_count) {
    return 0;
  }
  return (uint32_t)lane_count;
}

bool loom_vector_packed_integer_storage_shape(
    loom_type_t type, uint32_t storage_unit_bit_count,
    uint32_t maximum_storage_unit_count,
    loom_vector_packed_integer_storage_shape_t* out_shape) {
  if (out_shape == NULL) {
    return false;
  }
  *out_shape = (loom_vector_packed_integer_storage_shape_t){
      .type = type,
      .storage_unit_bit_count = storage_unit_bit_count,
  };
  if (storage_unit_bit_count == 0 || maximum_storage_unit_count == 0 ||
      !loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }

  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > UINT32_MAX) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_integer(element_type)) {
    return false;
  }
  const int32_t element_bit_count = loom_scalar_type_bitwidth(element_type);
  if (element_bit_count <= 0) {
    return false;
  }

  int64_t payload_bit_count = 0;
  if (!iree_checked_mul_i64(lane_count, element_bit_count,
                            &payload_bit_count) ||
      payload_bit_count <= 0 || payload_bit_count > UINT32_MAX) {
    return false;
  }
  const int64_t storage_unit_count =
      (payload_bit_count + storage_unit_bit_count - 1) / storage_unit_bit_count;
  if (storage_unit_count < 1 ||
      storage_unit_count > (int64_t)maximum_storage_unit_count) {
    return false;
  }

  *out_shape = (loom_vector_packed_integer_storage_shape_t){
      .type = type,
      .element_type = element_type,
      .lane_count = (uint32_t)lane_count,
      .element_bit_count = (uint32_t)element_bit_count,
      .payload_bit_count = (uint32_t)payload_bit_count,
      .storage_unit_bit_count = storage_unit_bit_count,
      .storage_unit_count = (uint32_t)storage_unit_count,
  };
  return true;
}

bool loom_vector_packed_integer_payload_from_lanes_match(
    loom_type_t source_type, loom_type_t result_type, uint32_t width,
    uint32_t storage_unit_bit_count, uint32_t maximum_result_storage_unit_count,
    loom_vector_packed_integer_payload_from_lanes_match_t* out_match) {
  if (out_match != NULL) {
    *out_match = (loom_vector_packed_integer_payload_from_lanes_match_t){0};
  }
  if (width == 0 || storage_unit_bit_count == 0 ||
      (storage_unit_bit_count % width) != 0) {
    return false;
  }

  loom_vector_packed_integer_storage_shape_t source_shape = {0};
  loom_vector_packed_integer_storage_shape_t result_shape = {0};
  if (!loom_vector_packed_integer_storage_shape(
          source_type, storage_unit_bit_count, UINT32_MAX, &source_shape) ||
      !loom_vector_packed_integer_storage_shape(
          result_type, storage_unit_bit_count,
          maximum_result_storage_unit_count, &result_shape) ||
      width > source_shape.element_bit_count) {
    return false;
  }

  const uint64_t expected_payload_bit_count =
      (uint64_t)source_shape.lane_count * width;
  if (expected_payload_bit_count > UINT32_MAX ||
      expected_payload_bit_count != result_shape.payload_bit_count) {
    return false;
  }

  if (out_match != NULL) {
    *out_match = (loom_vector_packed_integer_payload_from_lanes_match_t){
        .source_shape = source_shape,
        .result_shape = result_shape,
        .width = width,
    };
  }
  return true;
}

bool loom_vector_packed_integer_lanes_from_payload_match(
    loom_type_t source_type, loom_type_t result_type, uint32_t width,
    uint32_t storage_unit_bit_count, uint32_t maximum_source_storage_unit_count,
    uint32_t maximum_lane_count,
    loom_vector_packed_integer_lanes_from_payload_match_t* out_match) {
  if (out_match != NULL) {
    *out_match = (loom_vector_packed_integer_lanes_from_payload_match_t){0};
  }
  if (width == 0 || maximum_lane_count == 0 || storage_unit_bit_count == 0 ||
      (storage_unit_bit_count % width) != 0) {
    return false;
  }

  loom_vector_packed_integer_storage_shape_t source_shape = {0};
  loom_vector_packed_integer_storage_shape_t result_shape = {0};
  if (!loom_vector_packed_integer_storage_shape(
          source_type, storage_unit_bit_count,
          maximum_source_storage_unit_count, &source_shape) ||
      !loom_vector_packed_integer_storage_shape(
          result_type, storage_unit_bit_count, UINT32_MAX, &result_shape) ||
      (source_shape.payload_bit_count % width) != 0 ||
      width > result_shape.element_bit_count) {
    return false;
  }

  const uint32_t lane_count = source_shape.payload_bit_count / width;
  if (lane_count == 0 || lane_count > maximum_lane_count ||
      result_shape.lane_count != lane_count) {
    return false;
  }

  if (out_match != NULL) {
    *out_match = (loom_vector_packed_integer_lanes_from_payload_match_t){
        .source_shape = source_shape,
        .result_shape = result_shape,
        .width = width,
        .lane_count = lane_count,
    };
  }
  return true;
}
