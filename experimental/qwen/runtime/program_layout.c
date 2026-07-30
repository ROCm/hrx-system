// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/program_layout.h"

#include <string.h>

#include "experimental/qwen/runtime/model_shape.h"

#define QWEN_PROGRAM_STORAGE_ALIGNMENT 256

static iree_status_t qwen_program_layout_checked_product(
    iree_device_size_t lhs, iree_device_size_t rhs,
    iree_device_size_t* out_value) {
  if (!iree_device_size_checked_mul(lhs, rhs, out_value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen transient tensor byte length overflows");
  }
  return iree_ok_status();
}

static iree_status_t qwen_program_layout_tensor_byte_length(
    iree_host_size_t token_count, iree_device_size_t elements_per_token,
    iree_device_size_t element_byte_length,
    iree_device_size_t* out_byte_length) {
  iree_device_size_t element_count = 0;
  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      (iree_device_size_t)token_count, elements_per_token, &element_count));
  return qwen_program_layout_checked_product(element_count, element_byte_length,
                                             out_byte_length);
}

static iree_status_t qwen_program_layout_append(iree_device_size_t length,
                                                iree_device_size_t* cursor,
                                                qwen_program_span_t* out_span) {
  iree_device_size_t aligned_cursor = 0;
  if (!iree_device_size_checked_add(*cursor, QWEN_PROGRAM_STORAGE_ALIGNMENT - 1,
                                    &aligned_cursor)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen transient span alignment overflows");
  }
  aligned_cursor &= ~(QWEN_PROGRAM_STORAGE_ALIGNMENT - 1);
  iree_device_size_t next_cursor = 0;
  if (!iree_device_size_checked_add(aligned_cursor, length, &next_cursor)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen transient layout byte length overflows");
  }
  *out_span = (qwen_program_span_t){
      .offset = aligned_cursor,
      .length = length,
  };
  *cursor = next_cursor;
  return iree_ok_status();
}

iree_status_t qwen_layer_program_layout_calculate(
    iree_host_size_t token_count, qwen_layer_program_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));
  iree_device_size_t hidden_state_byte_length = 0;
  IREE_RETURN_IF_ERROR(qwen_model_hidden_state_byte_length(
      token_count, &hidden_state_byte_length));

  iree_device_size_t cursor = 0;
  iree_device_size_t byte_length = 0;

  // Large prefill projections consume materialized F32 RMSNorm output. This is
  // also sufficient for the smaller Q8_1 x4 representation used by decode.
  IREE_RETURN_IF_ERROR(
      qwen_program_layout_append(hidden_state_byte_length, &cursor,
                                 &out_layout->attention_projection_input));

  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE,
      sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(
      qwen_program_layout_append(byte_length, &cursor, &out_layout->raw_query));

  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_KEY_VALUE_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE,
      sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(
      qwen_program_layout_append(byte_length, &cursor, &out_layout->raw_key));
  IREE_RETURN_IF_ERROR(
      qwen_program_layout_append(byte_length, &cursor, &out_layout->raw_value));

  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE,
      sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(byte_length, &cursor,
                                                  &out_layout->rotated_query));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->attention_output));

  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      hidden_state_byte_length, &cursor, &out_layout->feed_forward_norm));

  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_EXPERT_COUNT, sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(byte_length, &cursor,
                                                  &out_layout->router_logits));

  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_ROUTE_COUNT, sizeof(int32_t), &byte_length));
  IREE_RETURN_IF_ERROR(
      qwen_program_layout_append(byte_length, &cursor, &out_layout->route_ids));
  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_ROUTE_COUNT, sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(byte_length, &cursor,
                                                  &out_layout->route_weights));

  // One count and token_count assignment slots for every expert.
  const iree_device_size_t expert_table_element_count =
      QWEN_MODEL_EXPERT_COUNT * (1 + (iree_device_size_t)token_count);
  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      expert_table_element_count, sizeof(int32_t), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(byte_length, &cursor,
                                                  &out_layout->expert_table));

  // One descriptor count followed by the producer's conservative capacity:
  // ceil(assignments / 32) assignment partitions plus one tail partition per
  // expert.
  const iree_device_size_t assignment_count =
      (iree_device_size_t)token_count * QWEN_MODEL_ROUTE_COUNT;
  const iree_device_size_t assignment_partition_count =
      (assignment_count + 31) / 32;
  const iree_device_size_t partition_descriptor_count =
      assignment_partition_count + QWEN_MODEL_EXPERT_COUNT;
  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      1 + partition_descriptor_count, sizeof(int32_t), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->partition_table));

  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_ROUTE_COUNT * QWEN_MODEL_EXPERT_INTERMEDIATE_SIZE,
      sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(
      qwen_program_layout_append(byte_length, &cursor, &out_layout->swiglu));

  IREE_RETURN_IF_ERROR(qwen_program_layout_tensor_byte_length(
      token_count, QWEN_MODEL_ROUTE_COUNT * QWEN_MODEL_HIDDEN_SIZE,
      /*F16 byte length=*/2, &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(byte_length, &cursor,
                                                  &out_layout->routed_down));

  out_layout->transient_byte_length = cursor;
  return iree_ok_status();
}
