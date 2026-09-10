// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_STREAM_VALUE_H_
#define IREE_EXPERIMENTAL_STREAMING_STREAM_VALUE_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;

// Operation represented by one stream-value batch entry.
typedef enum iree_hal_streaming_value_operation_kind_e {
  IREE_HAL_STREAMING_VALUE_OPERATION_WAIT = 0,
  IREE_HAL_STREAMING_VALUE_OPERATION_STORE = 1,
  IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE = 2,
} iree_hal_streaming_value_operation_kind_t;

// Parameters for one stream-value batch entry.
typedef union iree_hal_streaming_value_operation_params_u {
  // Parameters used by IREE_HAL_STREAMING_VALUE_OPERATION_WAIT.
  iree_hal_atomic_wait_params_t wait;
  // Parameters used by IREE_HAL_STREAMING_VALUE_OPERATION_STORE.
  iree_hal_atomic_store_params_t store;
  // Parameters used by IREE_HAL_STREAMING_VALUE_OPERATION_UPDATE.
  iree_hal_atomic_rmw_params_t update;
} iree_hal_streaming_value_operation_params_t;

// Fully resolved operation submitted at one position in a stream-value batch.
typedef struct iree_hal_streaming_value_operation_t {
  // Operation performed by this entry.
  iree_hal_streaming_value_operation_kind_t kind;
  // HAL allocation containing the naturally aligned target cell.
  iree_hal_buffer_t* target_buffer;
  // Byte offset of the target cell in |target_buffer|.
  iree_device_size_t target_offset;
  // Parameters selected by |kind|.
  iree_hal_streaming_value_operation_params_t params;
} iree_hal_streaming_value_operation_t;

// Returns true when |family_spec| can dedicate an exact queue to each logical
// stream that uses a memory wait. Waits must consume no dispatch resources so
// kernels that satisfy their predicates can continue to execute.
bool iree_hal_streaming_queue_family_supports_value_waits(
    const iree_hal_queue_family_spec_t* family_spec);

// Enqueues |operations| as one ordered stream transaction. All resources and
// command storage are prepared before the transaction is submitted. Batches
// containing a wait use an independently progressing queue owned by |stream|.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_value_operations(
    iree_hal_streaming_stream_t* stream, iree_host_size_t operation_count,
    const iree_hal_streaming_value_operation_t* operations);

// Enqueues a device-side value wait at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_wait_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_wait_params_t params);

// Enqueues a device-side atomic store at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_store_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_store_params_t params);

// Enqueues a device-side atomic update at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_update_value(
    iree_hal_streaming_stream_t* stream, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_rmw_params_t params);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_STREAM_VALUE_H_
