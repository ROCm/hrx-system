// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/request_state.h"

#include <string.h>

#include "experimental/qwen/runtime/model_shape.h"

#define QWEN_REQUEST_STORAGE_ALIGNMENT 256

static iree_status_t qwen_request_storage_append(
    iree_device_size_t length, iree_device_size_t alignment,
    iree_device_size_t* cursor, qwen_request_span_t* out_span) {
  iree_device_size_t aligned_cursor = 0;
  if (!iree_device_size_checked_add(*cursor, alignment - 1, &aligned_cursor)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen request storage alignment overflows");
  }
  aligned_cursor &= ~(alignment - 1);
  iree_device_size_t next_cursor = 0;
  if (!iree_device_size_checked_add(aligned_cursor, length, &next_cursor)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen request storage byte length overflows");
  }
  *out_span = (qwen_request_span_t){
      .offset = aligned_cursor,
      .length = length,
  };
  *cursor = next_cursor;
  return iree_ok_status();
}

iree_status_t qwen_request_storage_layout_calculate(
    iree_host_size_t token_count, iree_host_size_t context_capacity,
    qwen_request_storage_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));
  if (token_count > context_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen token count %" PRIhsz
                            " exceeds request context capacity %" PRIhsz,
                            token_count, context_capacity);
  }

  iree_device_size_t hidden_state_byte_length = 0;
  IREE_RETURN_IF_ERROR(qwen_model_hidden_state_byte_length(
      token_count, &hidden_state_byte_length));
  IREE_RETURN_IF_ERROR(qwen_model_layer_cache_byte_length(
      context_capacity, &out_layout->layer_cache_byte_length));

  iree_device_size_t cursor = 0;
  IREE_RETURN_IF_ERROR(qwen_request_storage_append(
      hidden_state_byte_length, QWEN_REQUEST_STORAGE_ALIGNMENT, &cursor,
      &out_layout->hidden_state));

  iree_device_size_t token_ids_byte_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)token_count,
                                    sizeof(int32_t), &token_ids_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen token-ID storage byte length overflows");
  }
  IREE_RETURN_IF_ERROR(qwen_request_storage_append(token_ids_byte_length,
                                                   _Alignof(int32_t), &cursor,
                                                   &out_layout->token_ids));
  IREE_RETURN_IF_ERROR(
      qwen_request_storage_append(sizeof(int32_t), _Alignof(int32_t), &cursor,
                                  &out_layout->selected_token));
  IREE_RETURN_IF_ERROR(qwen_request_storage_append(
      sizeof(qwen_request_control_t), _Alignof(qwen_request_control_t), &cursor,
      &out_layout->control));
  out_layout->reset_upload_byte_length = cursor;

  iree_device_size_t positions_byte_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)token_count,
                                    sizeof(int32_t), &positions_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen position storage byte length overflows");
  }
  IREE_RETURN_IF_ERROR(qwen_request_storage_append(positions_byte_length,
                                                   _Alignof(int32_t), &cursor,
                                                   &out_layout->positions));

  iree_device_size_t indices_byte_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)token_count,
                                    sizeof(int64_t), &indices_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen cache-index storage byte length overflows");
  }
  IREE_RETURN_IF_ERROR(
      qwen_request_storage_append(indices_byte_length, _Alignof(int64_t),
                                  &cursor, &out_layout->key_cache_indices));
  IREE_RETURN_IF_ERROR(
      qwen_request_storage_append(indices_byte_length, _Alignof(int64_t),
                                  &cursor, &out_layout->value_cache_indices));

  iree_device_size_t mask_element_count = 0;
  iree_device_size_t mask_byte_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)token_count,
                                    (iree_device_size_t)context_capacity,
                                    &mask_element_count) ||
      !iree_device_size_checked_mul(mask_element_count,
                                    /*F16 byte length=*/2, &mask_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen attention-mask byte length overflows");
  }
  IREE_RETURN_IF_ERROR(qwen_request_storage_append(
      mask_byte_length, /*FlashAttention vector alignment=*/16, &cursor,
      &out_layout->attention_mask));
  out_layout->dispatch_state_byte_length = cursor;

  iree_device_size_t all_layer_cache_byte_length = 0;
  if (!iree_device_size_checked_mul(out_layout->layer_cache_byte_length,
                                    QWEN_MODEL_LAYER_COUNT,
                                    &all_layer_cache_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen all-layer cache byte length overflows");
  }
  IREE_RETURN_IF_ERROR(qwen_request_storage_append(
      all_layer_cache_byte_length, QWEN_REQUEST_STORAGE_ALIGNMENT, &cursor,
      &out_layout->key_cache));
  IREE_RETURN_IF_ERROR(qwen_request_storage_append(
      all_layer_cache_byte_length, QWEN_REQUEST_STORAGE_ALIGNMENT, &cursor,
      &out_layout->value_cache));
  out_layout->persistent_byte_length = cursor;
  return iree_ok_status();
}
