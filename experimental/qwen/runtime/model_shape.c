// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/model_shape.h"

static iree_status_t qwen_model_shape_checked_byte_length(
    iree_device_size_t element_count, iree_device_size_t element_byte_length,
    iree_device_size_t* out_byte_length) {
  if (!iree_device_size_checked_mul(element_count, element_byte_length,
                                    out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen tensor byte length overflows device size");
  }
  return iree_ok_status();
}

iree_status_t qwen_model_hidden_state_byte_length(
    iree_host_size_t token_count, iree_device_size_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(out_byte_length);
  *out_byte_length = 0;
  if (token_count == 0 || token_count > QWEN_MODEL_MAX_PHYSICAL_TOKEN_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen physical token count %" PRIhsz
                            " is outside [1, %d]",
                            token_count, QWEN_MODEL_MAX_PHYSICAL_TOKEN_COUNT);
  }
  iree_device_size_t element_count = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)token_count,
                                    QWEN_MODEL_HIDDEN_SIZE, &element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen hidden-state element count overflows");
  }
  return qwen_model_shape_checked_byte_length(element_count, sizeof(float),
                                              out_byte_length);
}

iree_status_t qwen_model_layer_cache_byte_length(
    iree_host_size_t context_capacity, iree_device_size_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(out_byte_length);
  *out_byte_length = 0;
  if (context_capacity == 0 ||
      context_capacity > QWEN_MODEL_MAX_CONTEXT_CAPACITY) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen context capacity %" PRIhsz
                            " is outside [1, %d]",
                            context_capacity, QWEN_MODEL_MAX_CONTEXT_CAPACITY);
  }

  iree_device_size_t element_count = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)context_capacity,
                                    QWEN_MODEL_KEY_VALUE_HEAD_COUNT,
                                    &element_count) ||
      !iree_device_size_checked_mul(element_count, QWEN_MODEL_HEAD_SIZE,
                                    &element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen layer cache element count overflows");
  }
  return qwen_model_shape_checked_byte_length(
      element_count, /*F16 byte length=*/2, out_byte_length);
}

iree_status_t qwen_model_all_cache_byte_length(
    iree_host_size_t context_capacity, iree_device_size_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(out_byte_length);
  *out_byte_length = 0;
  iree_device_size_t layer_cache_byte_length = 0;
  IREE_RETURN_IF_ERROR(qwen_model_layer_cache_byte_length(
      context_capacity, &layer_cache_byte_length));
  iree_device_size_t all_cache_byte_length = 0;
  if (!iree_device_size_checked_mul(layer_cache_byte_length,
                                    QWEN_MODEL_LAYER_COUNT,
                                    &all_cache_byte_length) ||
      !iree_device_size_checked_mul(all_cache_byte_length,
                                    /*K and V=*/2, &all_cache_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen complete K/V cache byte length overflows");
  }
  *out_byte_length = all_cache_byte_length;
  return iree_ok_status();
}
