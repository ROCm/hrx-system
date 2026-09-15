// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_STREAM_VALUE_H_
#define HRX_BINDING_HIP_STREAM_VALUE_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  IREE_HIP_STREAM_WAIT_VALUE_GTE = 0x0,
  IREE_HIP_STREAM_WAIT_VALUE_EQ = 0x1,
  IREE_HIP_STREAM_WAIT_VALUE_AND = 0x2,
  IREE_HIP_STREAM_WAIT_VALUE_NOR = 0x3,
  IREE_HIP_STREAM_WRITE_VALUE_DEFAULT = 0x0,
  IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT = 0x1000,
  IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT = 0x1001,
};

typedef enum iree_hip_stream_value_write_operation_e {
  IREE_HIP_STREAM_VALUE_WRITE_OPERATION_STORE = 0,
  IREE_HIP_STREAM_VALUE_WRITE_OPERATION_ADD = 1,
  IREE_HIP_STREAM_VALUE_WRITE_OPERATION_SUBTRACT = 2,
} iree_hip_stream_value_write_operation_t;

typedef struct iree_hip_stream_value_write_params_t {
  // Selects which translated parameter record is active.
  iree_hip_stream_value_write_operation_t operation;
  // Parameters used when |operation| is STORE.
  iree_hal_atomic_store_params_t store;
  // Parameters used when |operation| is ADD or SUBTRACT.
  iree_hal_atomic_rmw_params_t update;
} iree_hip_stream_value_write_params_t;

// Translates one HIP write operation to generic HAL atomic parameters.
iree_status_t iree_hip_stream_write_value_params_initialize(
    uint64_t value, unsigned int flags, iree_hal_atomic_width_t width,
    iree_hip_stream_value_write_params_t* out_params);

// Translates one HIP wait predicate to the generic HAL atomic-wait form.
// Values and masks are narrowed to |width| before the predicate is built.
iree_status_t iree_hip_stream_wait_value_params_initialize(
    uint64_t value, uint64_t mask, unsigned int flags,
    iree_hal_atomic_width_t width, iree_hal_atomic_wait_params_t* out_params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_STREAM_VALUE_H_
