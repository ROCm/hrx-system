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
#define QWEN_PROGRAM_SPLIT_ATTENTION_KEY_VALUE_BLOCK_LENGTH 64
#define QWEN_PROGRAM_SPLIT_ATTENTION_QUERY_ROW_CAPACITY 16
#define QWEN_PROGRAM_DECODE_QKV_COMPLETION_COUNTER_COUNT \
  (QWEN_MODEL_QUERY_HEAD_COUNT + 2 * QWEN_MODEL_KEY_VALUE_HEAD_COUNT)
#define QWEN_PROGRAM_DECODE_GATE_UP_COMPLETION_COUNTER_COUNT       \
  (QWEN_MODEL_ROUTE_COUNT * (QWEN_MODEL_EXPERT_INTERMEDIATE_SIZE / \
                             QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT))

static_assert(QWEN_MODEL_HIDDEN_SIZE %
                      QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT ==
                  0,
              "Qwen hidden rows must contain complete GGML Q8_1 x4 groups");
static_assert((QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE) %
                      QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT ==
                  0,
              "Qwen attention rows must contain complete GGML Q8_1 x4 groups");
static_assert(
    (QWEN_MODEL_KEY_VALUE_HEAD_COUNT * sizeof(int32_t)) % 16 == 0,
    "Qwen split-attention counters must align grouped-stage counters");
static_assert(
    (QWEN_PROGRAM_DECODE_QKV_COMPLETION_COUNTER_COUNT * sizeof(int32_t)) % 16 ==
        0,
    "Qwen QKV-head counters must align grouped-stage counters");
static_assert((QWEN_PROGRAM_DECODE_GATE_UP_COMPLETION_COUNTER_COUNT *
               sizeof(int32_t)) %
                      16 ==
                  0,
              "Qwen gate/up counters must align the shared stage counter");

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
    iree_host_size_t token_count, iree_host_size_t context_count,
    qwen_full_program_layout_flags_t flags,
    qwen_full_program_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));
  const qwen_full_program_layout_flags_t supported_flags =
      QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_SPLIT_ATTENTION |
      QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_FUSED_STAGE_COMPLETION;
  if (iree_any_bit_set(flags, ~supported_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen full-program layout flags are unsupported");
  }
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(
      qwen_model_layer_cache_byte_length(context_count, &byte_length));

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

  if (iree_any_bit_set(flags,
                       QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_SPLIT_ATTENTION)) {
    const iree_device_size_t key_value_block_count =
        ((iree_device_size_t)context_count +
         QWEN_PROGRAM_SPLIT_ATTENTION_KEY_VALUE_BLOCK_LENGTH - 1) /
        QWEN_PROGRAM_SPLIT_ATTENTION_KEY_VALUE_BLOCK_LENGTH;
    iree_device_size_t partial_row_count = 0;
    IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
        QWEN_MODEL_KEY_VALUE_HEAD_COUNT, key_value_block_count,
        &partial_row_count));
    IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
        partial_row_count, QWEN_PROGRAM_SPLIT_ATTENTION_QUERY_ROW_CAPACITY,
        &partial_row_count));
    IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
        partial_row_count, sizeof(float), &byte_length));
    IREE_RETURN_IF_ERROR(qwen_program_layout_append(
        byte_length, &cursor, &out_layout->attention_partial_maximums));
    IREE_RETURN_IF_ERROR(qwen_program_layout_append(
        byte_length, &cursor, &out_layout->attention_partial_sums));

    IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
        partial_row_count, QWEN_MODEL_HEAD_SIZE, &byte_length));
    IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
        byte_length, /*F16 byte length=*/2, &byte_length));
    IREE_RETURN_IF_ERROR(qwen_program_layout_append(
        byte_length, &cursor, &out_layout->attention_partial_outputs));
  }

  const bool reserves_attention_completion = iree_any_bit_set(
      flags, QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_SPLIT_ATTENTION);
  const bool reserves_fused_stage_completion = iree_any_bit_set(
      flags, QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_FUSED_STAGE_COMPLETION);
  if (reserves_attention_completion || reserves_fused_stage_completion) {
    iree_device_size_t attention_byte_length = 0;
    if (reserves_attention_completion) {
      IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
          QWEN_MODEL_KEY_VALUE_HEAD_COUNT, sizeof(int32_t),
          &attention_byte_length));
    }
    iree_device_size_t grouped_stage_byte_length = 0;
    if (reserves_fused_stage_completion) {
      const iree_device_size_t grouped_stage_counter_count =
          iree_max(QWEN_PROGRAM_DECODE_QKV_COMPLETION_COUNTER_COUNT,
                   QWEN_PROGRAM_DECODE_GATE_UP_COMPLETION_COUNTER_COUNT);
      IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
          grouped_stage_counter_count, sizeof(int32_t),
          &grouped_stage_byte_length));
    }
    const iree_device_size_t shared_byte_length =
        reserves_fused_stage_completion ? sizeof(int32_t) : 0;
    iree_device_size_t initialization_byte_length = 0;
    if (!iree_device_size_checked_add(attention_byte_length,
                                      grouped_stage_byte_length,
                                      &initialization_byte_length) ||
        !iree_device_size_checked_add(initialization_byte_length,
                                      shared_byte_length,
                                      &initialization_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen completion storage length overflows");
    }
    IREE_RETURN_IF_ERROR(qwen_program_layout_append(
        initialization_byte_length, &cursor,
        &out_layout->decode_completion.initialization));
    out_layout->decode_completion.attention = (qwen_program_span_t){
        .offset = out_layout->decode_completion.initialization.offset,
        .length = attention_byte_length,
    };
    out_layout->decode_completion.grouped_stage = (qwen_program_span_t){
        .offset = out_layout->decode_completion.initialization.offset +
                  attention_byte_length,
        .length = grouped_stage_byte_length,
    };
    out_layout->decode_completion.shared = (qwen_program_span_t){
        .offset = out_layout->decode_completion.grouped_stage.offset +
                  grouped_stage_byte_length,
        .length = shared_byte_length,
    };
  }

  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      QWEN_MODEL_HIDDEN_SIZE / QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT,
      QWEN_PROGRAM_Q8_1_X4_GROUP_BYTE_LENGTH, &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->final_quantized_hidden_state));

  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      QWEN_MODEL_VOCABULARY_PARTIAL_COUNT, sizeof(float), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->vocabulary_argmax.partial_logits));

  IREE_RETURN_IF_ERROR(qwen_program_layout_checked_product(
      QWEN_MODEL_VOCABULARY_PARTIAL_COUNT, sizeof(int32_t), &byte_length));
  IREE_RETURN_IF_ERROR(qwen_program_layout_append(
      byte_length, &cursor, &out_layout->vocabulary_argmax.partial_ids));

  out_layout->transient_byte_length = cursor;
  return iree_ok_status();
}
