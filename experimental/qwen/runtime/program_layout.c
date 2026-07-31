// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/program_layout.h"

#include <string.h>

#include "experimental/qwen/runtime/model_shape.h"

#define QWEN_PROGRAM_STORAGE_ALIGNMENT 256
#define QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT 128
#define QWEN_PROGRAM_Q8_1_X4_GROUP_BYTE_LENGTH 144

static_assert(QWEN_MODEL_HIDDEN_SIZE %
                      QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT ==
                  0,
              "Qwen hidden rows must contain complete GGML Q8_1 x4 groups");
static_assert((QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE) %
                      QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT ==
                  0,
              "Qwen attention rows must contain complete GGML Q8_1 x4 groups");

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

static iree_status_t qwen_program_layout_rebase_layer(
    iree_device_size_t base_offset, qwen_layer_program_layout_t* layout) {
  qwen_program_span_t* spans[] = {
      &layout->projection_input_scratch,
      &layout->raw_query,
      &layout->raw_key,
      &layout->raw_value,
      &layout->rotated_query,
      &layout->attention_output,
      &layout->feed_forward_norm,
      &layout->router_logits,
      &layout->route_ids,
      &layout->route_weights,
      &layout->expert_table,
      &layout->partition_table,
      &layout->swiglu,
      &layout->routed_projection_scratch,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(spans); ++i) {
    iree_device_size_t rebased_offset = 0;
    if (!iree_device_size_checked_add(base_offset, spans[i]->offset,
                                      &rebased_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen nested transient span offset overflows");
    }
    spans[i]->offset = rebased_offset;
  }
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

  iree_device_size_t quantized_attention_byte_length = 0;
  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      token_count,
      (QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE) /
          QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT,
      &quantized_attention_byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      quantized_attention_byte_length, QWEN_PROGRAM_Q8_1_X4_GROUP_BYTE_LENGTH,
      &quantized_attention_byte_length));
  const iree_device_size_t projection_input_scratch_byte_length =
      iree_max(hidden_state_byte_length, quantized_attention_byte_length);
  IREE_RETURN_IF_ERROR(
      qwen_program_layout_append(projection_input_scratch_byte_length, &cursor,
                                 &out_layout->projection_input_scratch));

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
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->routed_projection_scratch));

  out_layout->transient_byte_length = cursor;
  return iree_ok_status();
}

iree_status_t qwen_full_program_layout_calculate(
    iree_host_size_t token_count, qwen_full_program_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  IREE_RETURN_IF_ERROR(
      qwen_layer_program_layout_calculate(token_count, &out_layout->layer));
  IREE_RETURN_IF_ERROR(
      qwen_layer_program_layout_calculate(1, &out_layout->terminal_layer));

  iree_device_size_t cursor = out_layout->layer.transient_byte_length;
  qwen_program_span_t terminal_layer_storage;
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      out_layout->terminal_layer.transient_byte_length, &cursor,
      &terminal_layer_storage));
  IREE_RETURN_IF_ERROR(qwen_program_layout_rebase_layer(
      terminal_layer_storage.offset, &out_layout->terminal_layer));

  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      QWEN_MODEL_HIDDEN_SIZE, sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->final_normalized_hidden_state));

  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      QWEN_MODEL_HIDDEN_SIZE / QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT,
      QWEN_PROGRAM_Q8_1_X4_GROUP_BYTE_LENGTH, &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->final_quantized_hidden_state));

  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      QWEN_MODEL_VOCABULARY_SIZE, sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->vocabulary_logits));

  out_layout->transient_byte_length = cursor;
  return iree_ok_status();
}
