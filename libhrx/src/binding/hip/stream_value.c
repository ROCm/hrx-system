// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/stream_value.h"

#include <string.h>

iree_status_t iree_hip_stream_write_value_params_initialize(
    uint64_t value, unsigned int flags, iree_hal_atomic_width_t width,
    iree_hip_stream_value_write_params_t* out_params) {
  IREE_ASSERT_ARGUMENT(out_params);
  memset(out_params, 0, sizeof(*out_params));

  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      value &= UINT32_MAX;
      break;
    case IREE_HAL_ATOMIC_WIDTH_64:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported HIP stream value width %u", width);
  }

  switch (flags) {
    case IREE_HIP_STREAM_WRITE_VALUE_DEFAULT:
      out_params->operation = IREE_HIP_STREAM_VALUE_WRITE_OPERATION_STORE;
      out_params->store = (iree_hal_atomic_store_params_t){
          .value = value,
          .flags = IREE_HAL_ATOMIC_FLAG_RELEASE,
          .width = width,
      };
      break;
    case IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT:
    case IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT:
      out_params->operation =
          flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT
              ? IREE_HIP_STREAM_VALUE_WRITE_OPERATION_ADD
              : IREE_HIP_STREAM_VALUE_WRITE_OPERATION_SUBTRACT;
      out_params->update = (iree_hal_atomic_rmw_params_t){
          .operand = value,
          .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
          .width = width,
          .operation = flags == IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT
                           ? IREE_HAL_ATOMIC_RMW_OPERATION_ADD
                           : IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
      };
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported HIP stream write flags 0x%x", flags);
  }
  return iree_ok_status();
}

iree_status_t iree_hip_stream_wait_value_params_initialize(
    uint64_t value, uint64_t mask, unsigned int flags,
    iree_hal_atomic_width_t width, iree_hal_atomic_wait_params_t* out_params) {
  IREE_ASSERT_ARGUMENT(out_params);
  memset(out_params, 0, sizeof(*out_params));

  uint64_t width_mask = 0;
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      width_mask = UINT32_MAX;
      break;
    case IREE_HAL_ATOMIC_WIDTH_64:
      width_mask = UINT64_MAX;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported HIP stream value width %u", width);
  }

  value &= width_mask;
  mask &= width_mask;
  *out_params = (iree_hal_atomic_wait_params_t){
      .value = value,
      .mask = mask,
      .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE,
      .width = width,
  };
  switch (flags) {
    case IREE_HIP_STREAM_WAIT_VALUE_GTE:
      out_params->condition =
          IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
      break;
    case IREE_HIP_STREAM_WAIT_VALUE_EQ:
      out_params->condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL;
      break;
    case IREE_HIP_STREAM_WAIT_VALUE_AND:
      // ((observed & mask) & value) != 0
      out_params->value = 0;
      out_params->mask = mask & value;
      out_params->condition = IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL;
      break;
    case IREE_HIP_STREAM_WAIT_VALUE_NOR: {
      // A NOR wait completes when at least one masked bit is clear in both the
      // observed value and the requested value.
      const uint64_t required_zero_bits = (~value) & mask & width_mask;
      out_params->value = required_zero_bits;
      out_params->mask = required_zero_bits;
      out_params->condition = IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL;
      break;
    }
    default:
      memset(out_params, 0, sizeof(*out_params));
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported HIP stream wait condition 0x%x",
                              flags);
  }
  return iree_ok_status();
}
